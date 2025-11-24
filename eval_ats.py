#
# Evaluation script to compare VLM performance with and without ATS (Adaptive Token Sampling)
# Measures TTFT (Time To First Token) and generation speed
#
import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

import torch
from PIL import Image

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def run_inference(args, use_ats: bool, top_k_tokens: int = 200):
    """Run inference with or without ATS"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model (only once)
    if not hasattr(run_inference, 'model_cache') or run_inference.model_cache is None:
        disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, args.model_base, model_name, device=device.type
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        run_inference.model_cache = (tokenizer, model, image_processor, device)
    else:
        tokenizer, model, image_processor, device = run_inference.model_cache
    
    # Construct prompt
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    
    # Load and preprocess image
    image = Image.open(args.image_file).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_inputs = image_tensor.unsqueeze(0).to(device=device, dtype=model.dtype)
    
    # ATS processing if enabled
    pre_encoded_features = None
    num_visual_tokens = None
    
    if use_ats and top_k_tokens > 0:
        vision_tower = model.get_model().get_vision_tower()
        with torch.inference_mode():
            result = vision_tower.forward_images(
                image_inputs, return_attention_maps=True
            )
            if isinstance(result, tuple):
                vision_features_full, attention_maps = result
            else:
                vision_features_full = result
                attention_maps = None
        
        if attention_maps is not None:
            # Process attention maps
            attn_avg = attention_maps.mean(dim=1)[0]  # (N, N)
            token_importance = attn_avg.mean(dim=0)  # (N,)
            
            num_tokens = token_importance.shape[0]
            # k = min(top_k_tokens, int(num_tokens * 0.5))
            k = top_k_tokens
            
            _, top_k_indices = torch.topk(token_importance, k, dim=0)
            top_k_indices = top_k_indices.sort().values
            
            # Filter features
            if vision_features_full.dim() == 3:
                pre_encoded_features = vision_features_full[:, top_k_indices, :]
            elif vision_features_full.dim() == 2:
                pre_encoded_features = vision_features_full[top_k_indices, :]
            
            num_visual_tokens = k
        else:
            use_ats = False
    
    if not use_ats or pre_encoded_features is None:
        # Compute full visual tokens
        with torch.inference_mode():
            vision_features = model.encode_images(image_inputs)
        if vision_features.dim() == 3:
            num_visual_tokens = vision_features.shape[1]
        elif vision_features.dim() == 2:
            num_visual_tokens = vision_features.shape[0]
    
    # Warmup run (discard)
    with torch.inference_mode():
        _ = model.generate(
            input_ids,
            images=image_inputs,
            image_sizes=[image.size],
            max_new_tokens=1,
            use_cache=True,
            pre_encoded_features=pre_encoded_features if use_ats else None
        )
    
    # Measure TTFT: time to generate first token
    ttft_start = time.perf_counter()
    with torch.inference_mode():
        _ = model.generate(
            input_ids,
            images=image_inputs,
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1,  # Generate only first token for TTFT
            use_cache=True,
            pre_encoded_features=pre_encoded_features if use_ats else None
        )
    ttft = time.perf_counter() - ttft_start
    
    # Measure total generation time
    generation_start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_inputs,
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pre_encoded_features=pre_encoded_features if use_ats else None
        )
    generation_end = time.perf_counter()
    
    # Decode output
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # Calculate metrics
    total_time = generation_end - generation_start
    num_generated_tokens = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_second = num_generated_tokens / total_time if total_time > 0 else 0
    
    return {
        'ttft': ttft,
        'total_time': total_time,
        'num_visual_tokens': num_visual_tokens,
        'num_generated_tokens': num_generated_tokens,
        'tokens_per_second': tokens_per_second,
        'output': outputs,
        'use_ats': use_ats,
        'top_k_tokens': top_k_tokens if use_ats else None
    }


def evaluate_ats(args):
    """Run evaluation comparing with and without ATS"""
    results = {
        'image_file': args.image_file,
        'prompt': args.prompt,
        'model_path': args.model_path,
        'max_new_tokens': args.max_new_tokens,
        'runs': []
    }
    
    print(f"\n{'='*60}")
    print(f"Evaluating ATS Performance")
    print(f"{'='*60}")
    print(f"Image: {args.image_file}")
    print(f"Prompt: {args.prompt}")
    print(f"Model: {args.model_path}")
    print(f"Number of runs: {args.num_runs}")
    print(f"{'='*60}\n")
    
    # Run without ATS
    print("Running WITHOUT ATS (baseline)...")
    baseline_results = []
    for i in range(args.num_runs):
        print(f"  Run {i+1}/{args.num_runs}...", end=' ', flush=True)
        result = run_inference(args, use_ats=False)
        baseline_results.append(result)
        print(f"TTFT: {result['ttft']*1000:.2f}ms, Tokens: {result['num_visual_tokens']}")
    results['baseline'] = baseline_results
    
    # Run with ATS
    print(f"\nRunning WITH ATS (top-{args.top_k_tokens} tokens)...")
    ats_results = []
    for i in range(args.num_runs):
        print(f"  Run {i+1}/{args.num_runs}...", end=' ', flush=True)
        result = run_inference(args, use_ats=True, top_k_tokens=args.top_k_tokens)
        ats_results.append(result)
        print(f"TTFT: {result['ttft']*1000:.2f}ms, Tokens: {result['num_visual_tokens']}")
    results['ats'] = ats_results
    
    # Calculate statistics
    baseline_ttfts = [r['ttft'] for r in baseline_results if r['ttft'] is not None]
    ats_ttfts = [r['ttft'] for r in ats_results if r['ttft'] is not None]
    
    baseline_total_times = [r['total_time'] for r in baseline_results]
    ats_total_times = [r['total_time'] for r in ats_results]
    
    baseline_tps = [r['tokens_per_second'] for r in baseline_results]
    ats_tps = [r['tokens_per_second'] for r in ats_results]
    
    stats = {
        'baseline': {
            'ttft_mean': statistics.mean(baseline_ttfts) if baseline_ttfts else None,
            'ttft_std': statistics.stdev(baseline_ttfts) if len(baseline_ttfts) > 1 else None,
            'ttft_min': min(baseline_ttfts) if baseline_ttfts else None,
            'ttft_max': max(baseline_ttfts) if baseline_ttfts else None,
            'total_time_mean': statistics.mean(baseline_total_times),
            'total_time_std': statistics.stdev(baseline_total_times) if len(baseline_total_times) > 1 else 0,
            'tokens_per_second_mean': statistics.mean(baseline_tps),
            'num_visual_tokens': baseline_results[0]['num_visual_tokens'] if baseline_results else None,
        },
        'ats': {
            'ttft_mean': statistics.mean(ats_ttfts) if ats_ttfts else None,
            'ttft_std': statistics.stdev(ats_ttfts) if len(ats_ttfts) > 1 else None,
            'ttft_min': min(ats_ttfts) if ats_ttfts else None,
            'ttft_max': max(ats_ttfts) if ats_ttfts else None,
            'total_time_mean': statistics.mean(ats_total_times),
            'total_time_std': statistics.stdev(ats_total_times) if len(ats_total_times) > 1 else 0,
            'tokens_per_second_mean': statistics.mean(ats_tps),
            'num_visual_tokens': ats_results[0]['num_visual_tokens'] if ats_results else None,
        }
    }
    
    # Calculate improvements
    if stats['baseline']['ttft_mean'] and stats['ats']['ttft_mean']:
        ttft_improvement = ((stats['baseline']['ttft_mean'] - stats['ats']['ttft_mean']) / stats['baseline']['ttft_mean']) * 100
        stats['improvement'] = {
            'ttft_percent': ttft_improvement,
            'ttft_absolute_ms': (stats['baseline']['ttft_mean'] - stats['ats']['ttft_mean']) * 1000,
            'token_reduction_percent': ((stats['baseline']['num_visual_tokens'] - stats['ats']['num_visual_tokens']) / stats['baseline']['num_visual_tokens']) * 100 if stats['baseline']['num_visual_tokens'] else None,
        }
    
    results['statistics'] = stats
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\nBaseline (no ATS):")
    print(f"  Visual tokens: {stats['baseline']['num_visual_tokens']}")
    print(f"  TTFT: {stats['baseline']['ttft_mean']*1000:.2f}ms ± {stats['baseline']['ttft_std']*1000:.2f}ms" if stats['baseline']['ttft_std'] else f"  TTFT: {stats['baseline']['ttft_mean']*1000:.2f}ms")
    print(f"  Total time: {stats['baseline']['total_time_mean']:.3f}s ± {stats['baseline']['total_time_std']:.3f}s")
    print(f"  Tokens/sec: {stats['baseline']['tokens_per_second_mean']:.2f}")
    
    print(f"\nATS (top-{args.top_k_tokens} tokens):")
    print(f"  Visual tokens: {stats['ats']['num_visual_tokens']}")
    print(f"  TTFT: {stats['ats']['ttft_mean']*1000:.2f}ms ± {stats['ats']['ttft_std']*1000:.2f}ms" if stats['ats']['ttft_std'] else f"  TTFT: {stats['ats']['ttft_mean']*1000:.2f}ms")
    print(f"  Total time: {stats['ats']['total_time_mean']:.3f}s ± {stats['ats']['total_time_std']:.3f}s")
    print(f"  Tokens/sec: {stats['ats']['tokens_per_second_mean']:.2f}")
    
    if 'improvement' in stats:
        print(f"\nImprovement:")
        print(f"  TTFT: {stats['improvement']['ttft_percent']:.1f}% faster ({stats['improvement']['ttft_absolute_ms']:.2f}ms reduction)")
        if stats['improvement']['token_reduction_percent']:
            print(f"  Visual tokens: {stats['improvement']['token_reduction_percent']:.1f}% reduction")
    
    print(f"{'='*60}\n")
    
    # Print generated outputs for comparison
    print(f"{'='*60}")
    print("GENERATED OUTPUTS COMPARISON")
    print(f"{'='*60}")
    
    # Show baseline outputs
    print(f"\n📝 Baseline Outputs (no ATS):")
    print(f"{'-'*60}")
    for i, result in enumerate(baseline_results[:3], 1):  # Show first 3 runs
        output_preview = result['output'][:200] + "..." if len(result['output']) > 200 else result['output']
        print(f"\nRun {i}:")
        print(f"  {output_preview}")
    
    # Show ATS outputs
    print(f"\n📝 ATS Outputs (top-{args.top_k_tokens} tokens):")
    print(f"{'-'*60}")
    for i, result in enumerate(ats_results[:3], 1):  # Show first 3 runs
        output_preview = result['output'][:200] + "..." if len(result['output']) > 200 else result['output']
        print(f"\nRun {i}:")
        print(f"  {output_preview}")
    
    # Show full outputs if only one run
    if args.num_runs == 1:
        print(f"\n{'='*60}")
        print("FULL OUTPUTS")
        print(f"{'='*60}")
        print(f"\n📝 Baseline (no ATS) - Full Output:")
        print(f"{'-'*60}")
        print(baseline_results[0]['output'])
        print(f"\n📝 ATS (top-{args.top_k_tokens} tokens) - Full Output:")
        print(f"{'-'*60}")
        print(ats_results[0]['output'])
    
    print(f"\n{'='*60}\n")
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ATS performance on VLM")
    parser.add_argument("--model-path", type=str, default="checkpoints/llava-fastvithd_0.5b_stage3")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="img/banana.jpg", help="Location of image file")
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt for VLM")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--top_k_tokens", type=int, default=256, help="Number of top visual tokens to keep with ATS")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs for each configuration")
    parser.add_argument("--output-file", type=str, default="eval_results.json", help="Output file for results (JSON)")
    args = parser.parse_args()
    
    # Initialize model cache
    run_inference.model_cache = None
    
    evaluate_ats(args)


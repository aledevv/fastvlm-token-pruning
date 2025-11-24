#!/usr/bin/env python3
#
# Evaluation script to compare different token pruning methods for VLM optimization
# Tests: Attention-based, Similarity-based (PruMerge Lite), and Norm-based pruning
# Measures TTFT, generation speed, and output quality
#
import os
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics

import torch
import torch.nn.functional as F
from PIL import Image

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


# ============================================================================
# PRUNING METHODS
# ============================================================================

def attention_based_pruning(vision_features, attention_maps, keep_ratio):
    """
    Attention-Based Pruning (ATS method)
    Keeps tokens with highest attention importance
    
    Args:
        vision_features: tensor of shape [batch, num_tokens, features]
        attention_maps: attention weights from vision tower
        keep_ratio: ratio of tokens to keep (0.0 to 1.0)
    
    Returns:
        pruned_features: filtered tokens
        kept_indices: indices of kept tokens
    """
    if attention_maps is None:
        return vision_features, None
    
    # Average attention across heads and compute token importance
    attn_avg = attention_maps.mean(dim=1)[0]  # (N, N)
    token_importance = attn_avg.mean(dim=0)  # (N,)
    
    num_tokens = token_importance.shape[0]
    k = max(1, int(num_tokens * keep_ratio))
    
    # Select top-k most important tokens
    _, top_k_indices = torch.topk(token_importance, k, dim=0)
    top_k_indices = top_k_indices.sort().values
    
    # Filter features
    if vision_features.dim() == 3:
        pruned_features = vision_features[:, top_k_indices, :]
    elif vision_features.dim() == 2:
        pruned_features = vision_features[top_k_indices, :]
    
    return pruned_features, top_k_indices


def similarity_based_pruning(vision_features, keep_ratio, similarity_threshold=0.90):
    """
    Similarity-Based Pruning (PruMerge Lite)
    Merges similar tokens to reduce redundancy
    
    Args:
        vision_features: tensor of shape [batch, num_tokens, features]
        keep_ratio: target ratio of tokens to keep
        similarity_threshold: cosine similarity threshold for clustering
    
    Returns:
        pruned_features: merged tokens
        kept_indices: None (merging doesn't preserve exact indices)
    """
    if vision_features.dim() == 3:
        batch_size, num_tokens, feat_dim = vision_features.shape
        features = vision_features[0]  # Process first batch item
    else:
        num_tokens, feat_dim = vision_features.shape
        features = vision_features
        batch_size = 1
    
    # Normalize features for cosine similarity
    features_norm = F.normalize(features, p=2, dim=-1)
    
    # Compute pairwise cosine similarity matrix
    similarity_matrix = torch.mm(features_norm, features_norm.t())  # (N, N)
    
    # Greedy clustering based on similarity
    merged_tokens = []
    processed = torch.zeros(num_tokens, dtype=torch.bool, device=features.device)
    
    target_num_tokens = max(1, int(num_tokens * keep_ratio))
    
    for i in range(num_tokens):
        if processed[i]:
            continue
        
        # Find all tokens similar to current token
        similar_mask = similarity_matrix[i] > similarity_threshold
        similar_mask[processed] = False  # Exclude already processed tokens
        similar_indices = similar_mask.nonzero(as_tuple=True)[0]
        
        if len(similar_indices) > 0:
            # Average pool similar tokens
            cluster_tokens = features[similar_indices]
            merged_token = cluster_tokens.mean(dim=0)
            merged_tokens.append(merged_token)
            processed[similar_indices] = True
        else:
            # Keep token as is
            merged_tokens.append(features[i])
            processed[i] = True
        
        # Stop if we've reached target number of tokens
        if len(merged_tokens) >= target_num_tokens:
            break
    
    # If we have too many tokens, select by norm
    if len(merged_tokens) > target_num_tokens:
        merged_tensor = torch.stack(merged_tokens)
        norms = torch.norm(merged_tensor, p=2, dim=-1)
        _, top_indices = torch.topk(norms, target_num_tokens)
        top_indices = top_indices.sort().values
        merged_tensor = merged_tensor[top_indices]
    else:
        merged_tensor = torch.stack(merged_tokens)
    
    # Reshape to match expected output format
    if batch_size > 1 or vision_features.dim() == 3:
        merged_tensor = merged_tensor.unsqueeze(0)
    
    return merged_tensor, None


def norm_based_pruning(vision_features, keep_ratio):
    """
    Norm-Based Pruning (Low Magnitude)
    Keeps tokens with highest L2 norm
    
    Args:
        vision_features: tensor of shape [batch, num_tokens, features]
        keep_ratio: ratio of tokens to keep (0.0 to 1.0)
    
    Returns:
        pruned_features: filtered tokens
        kept_indices: indices of kept tokens
    """
    if vision_features.dim() == 3:
        batch_size, num_tokens, feat_dim = vision_features.shape
        features = vision_features[0]  # Process first batch item
    else:
        num_tokens, feat_dim = vision_features.shape
        features = vision_features
        batch_size = 1
    
    # Compute L2 norm of each token
    token_norms = torch.norm(features, p=2, dim=-1)  # (N,)
    
    # Select top-k tokens with highest norm
    k = max(1, int(num_tokens * keep_ratio))
    _, top_k_indices = torch.topk(token_norms, k, dim=0)
    top_k_indices = top_k_indices.sort().values
    
    # Filter features
    pruned_features = features[top_k_indices]
    
    # Reshape to match expected output format
    if batch_size > 1 or vision_features.dim() == 3:
        pruned_features = pruned_features.unsqueeze(0)
    
    return pruned_features, top_k_indices


# ============================================================================
# INFERENCE AND EVALUATION
# ============================================================================

def run_inference(args, pruning_method='none', keep_ratio=1.0):
    """Run inference with specified pruning method"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model (cached)
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
    
    # Extract vision features
    pre_encoded_features = None
    num_visual_tokens_original = None
    num_visual_tokens_pruned = None
    pruning_time = 0.0
    
    if pruning_method != 'none':
        vision_tower = model.get_model().get_vision_tower()
        
        with torch.inference_mode():
            # Get vision features and attention maps
            result = vision_tower.forward_images(
                image_inputs, return_attention_maps=(pruning_method == 'attention')
            )
            if isinstance(result, tuple):
                vision_features_full, attention_maps = result
            else:
                vision_features_full = result
                attention_maps = None
        
        num_visual_tokens_original = vision_features_full.shape[1] if vision_features_full.dim() == 3 else vision_features_full.shape[0]
        
        # Apply pruning
        pruning_start = time.perf_counter()
        
        if pruning_method == 'attention':
            pre_encoded_features, _ = attention_based_pruning(
                vision_features_full, attention_maps, keep_ratio
            )
        elif pruning_method == 'similarity':
            pre_encoded_features, _ = similarity_based_pruning(
                vision_features_full, keep_ratio
            )
        elif pruning_method == 'norm':
            pre_encoded_features, _ = norm_based_pruning(
                vision_features_full, keep_ratio
            )
        
        pruning_time = time.perf_counter() - pruning_start
        
        num_visual_tokens_pruned = pre_encoded_features.shape[1] if pre_encoded_features.dim() == 3 else pre_encoded_features.shape[0]
    else:
        # Baseline: no pruning
        with torch.inference_mode():
            vision_features = model.encode_images(image_inputs)
        num_visual_tokens_original = vision_features.shape[1] if vision_features.dim() == 3 else vision_features.shape[0]
        num_visual_tokens_pruned = num_visual_tokens_original
    
    # Warmup run
    with torch.inference_mode():
        _ = model.generate(
            input_ids,
            images=image_inputs,
            image_sizes=[image.size],
            max_new_tokens=1,
            use_cache=True,
            pre_encoded_features=pre_encoded_features if pruning_method != 'none' else None
        )
    
    # Measure TTFT
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
            max_new_tokens=1,
            use_cache=True,
            pre_encoded_features=pre_encoded_features if pruning_method != 'none' else None
        )
    ttft = time.perf_counter() - ttft_start
    
    # Full generation
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
            pre_encoded_features=pre_encoded_features if pruning_method != 'none' else None
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
        'pruning_time': pruning_time,
        'num_visual_tokens_original': num_visual_tokens_original,
        'num_visual_tokens_pruned': num_visual_tokens_pruned,
        'token_reduction_percent': ((num_visual_tokens_original - num_visual_tokens_pruned) / num_visual_tokens_original * 100) if num_visual_tokens_original else 0,
        'num_generated_tokens': num_generated_tokens,
        'tokens_per_second': tokens_per_second,
        'output': outputs,
        'pruning_method': pruning_method,
        'keep_ratio': keep_ratio
    }


def evaluate_all_methods(args):
    """Run comprehensive evaluation of all pruning methods"""
    results = {
        'metadata': {
            'image_file': args.image_file,
            'prompt': args.prompt,
            'model_path': args.model_path,
            'max_new_tokens': args.max_new_tokens,
            'num_runs': args.num_runs,
            'retention_ratios': args.retention_ratios
        },
        'experiments': []
    }
    
    print(f"\n{'='*80}")
    print(f"🔬 PRUNING METHODS EVALUATION")
    print(f"{'='*80}")
    print(f"Image: {args.image_file}")
    print(f"Prompt: {args.prompt}")
    print(f"Model: {args.model_path}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Retention ratios: {args.retention_ratios}")
    print(f"{'='*80}\n")
    
    # Baseline: No pruning
    print("📊 Running BASELINE (no pruning)...")
    baseline_runs = []
    for i in range(args.num_runs):
        print(f"  Run {i+1}/{args.num_runs}...", end=' ', flush=True)
        result = run_inference(args, pruning_method='none', keep_ratio=1.0)
        baseline_runs.append(result)
        print(f"TTFT: {result['ttft']*1000:.2f}ms, Visual Tokens: {result['num_visual_tokens_pruned']}")
    
    results['experiments'].append({
        'method': 'baseline',
        'keep_ratio': 1.0,
        'runs': baseline_runs
    })
    
    # Test each pruning method with different retention ratios
    methods = ['attention', 'similarity', 'norm']
    method_names = {
        'attention': 'Attention-Based (ATS)',
        'similarity': 'Similarity-Based (PruMerge Lite)',
        'norm': 'Norm-Based (Low Magnitude)'
    }
    
    for method in methods:
        for keep_ratio in args.retention_ratios:
            print(f"\n📊 Running {method_names[method].upper()} (keep {keep_ratio*100:.0f}%)...")
            method_runs = []
            
            for i in range(args.num_runs):
                print(f"  Run {i+1}/{args.num_runs}...", end=' ', flush=True)
                result = run_inference(args, pruning_method=method, keep_ratio=keep_ratio)
                method_runs.append(result)
                print(f"TTFT: {result['ttft']*1000:.2f}ms, Tokens: {result['num_visual_tokens_original']}→{result['num_visual_tokens_pruned']}")
            
            results['experiments'].append({
                'method': method,
                'keep_ratio': keep_ratio,
                'runs': method_runs
            })
    
    # Calculate statistics and comparisons
    print(f"\n{'='*80}")
    print("📈 COMPUTING STATISTICS...")
    print(f"{'='*80}\n")
    
    summary = compute_summary_statistics(results)
    results['summary'] = summary
    
    # Print results table
    print_results_table(summary)
    
    # Print output quality comparison
    print_output_comparison(results)
    
    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✅ Results saved to: {output_path}\n")
    
    return results


def compute_summary_statistics(results):
    """Compute summary statistics for all experiments"""
    summary = []
    
    for exp in results['experiments']:
        runs = exp['runs']
        
        ttfts = [r['ttft'] for r in runs]
        total_times = [r['total_time'] for r in runs]
        tps = [r['tokens_per_second'] for r in runs]
        
        stats = {
            'method': exp['method'],
            'method_name': {
                'baseline': 'Baseline (No Pruning)',
                'attention': 'Attention-Based',
                'similarity': 'Similarity-Based',
                'norm': 'Norm-Based'
            }.get(exp['method'], exp['method']),
            'keep_ratio': exp['keep_ratio'],
            'ttft_mean': statistics.mean(ttfts),
            'ttft_std': statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
            'total_time_mean': statistics.mean(total_times),
            'tokens_per_second_mean': statistics.mean(tps),
            'num_visual_tokens_original': runs[0]['num_visual_tokens_original'],
            'num_visual_tokens_pruned': runs[0]['num_visual_tokens_pruned'],
            'token_reduction_percent': runs[0]['token_reduction_percent'],
            'sample_output': runs[0]['output'][:150] + "..." if len(runs[0]['output']) > 150 else runs[0]['output']
        }
        
        summary.append(stats)
    
    # Add relative improvements vs baseline
    baseline_stats = summary[0]
    for i in range(1, len(summary)):
        stats = summary[i]
        stats['ttft_improvement_percent'] = ((baseline_stats['ttft_mean'] - stats['ttft_mean']) / baseline_stats['ttft_mean']) * 100
        stats['total_time_improvement_percent'] = ((baseline_stats['total_time_mean'] - stats['total_time_mean']) / baseline_stats['total_time_mean']) * 100
        stats['speedup_factor'] = baseline_stats['total_time_mean'] / stats['total_time_mean']
    
    return summary


def print_results_table(summary):
    """Print formatted results table"""
    print(f"{'='*130}")
    print(f"{'Method':<25} {'Keep%':<7} {'Tokens':<12} {'TTFT (ms)':<12} {'Total (s)':<11} {'Tok/s':<8} {'Speedup':<8} {'Quality Hint':<20}")
    print(f"{'='*130}")
    
    for stats in summary:
        method_name = stats['method_name']
        keep_pct = f"{stats['keep_ratio']*100:.0f}%"
        tokens = f"{stats['num_visual_tokens_pruned']}/{stats['num_visual_tokens_original']}"
        ttft = f"{stats['ttft_mean']*1000:.1f}±{stats['ttft_std']*1000:.1f}" if stats['ttft_std'] > 0 else f"{stats['ttft_mean']*1000:.1f}"
        total_time = f"{stats['total_time_mean']:.2f}"
        tps = f"{stats['tokens_per_second_mean']:.1f}"
        
        if 'speedup_factor' in stats:
            speedup = f"{stats['speedup_factor']:.2f}x"
            ttft_impr = stats['ttft_improvement_percent']
            quality_hint = "✅ Good" if ttft_impr > 20 else "⚠️ Moderate" if ttft_impr > 0 else "❌ Slower"
        else:
            speedup = "-"
            quality_hint = "🔵 Reference"
        
        print(f"{method_name:<25} {keep_pct:<7} {tokens:<12} {ttft:<12} {total_time:<11} {tps:<8} {speedup:<8} {quality_hint:<20}")
    
    print(f"{'='*130}\n")


def print_output_comparison(results):
    """Print side-by-side output comparison"""
    print(f"{'='*80}")
    print(f"📝 OUTPUT QUALITY COMPARISON")
    print(f"{'='*80}\n")
    
    # Get baseline output
    baseline_exp = results['experiments'][0]
    baseline_output = baseline_exp['runs'][0]['output']
    
    print(f"🔵 BASELINE (No Pruning)")
    print(f"{'-'*80}")
    print(f"{baseline_output}\n")
    
    # Show outputs for each method at different retention ratios
    methods = {}
    for exp in results['experiments'][1:]:
        method = exp['method']
        if method not in methods:
            methods[method] = []
        methods[method].append(exp)
    
    method_names = {
        'attention': '🎯 ATTENTION-BASED (ATS)',
        'similarity': '🔗 SIMILARITY-BASED (PruMerge Lite)',
        'norm': '📊 NORM-BASED (Low Magnitude)'
    }
    
    for method, exps in methods.items():
        print(f"\n{method_names.get(method, method)}")
        print(f"{'-'*80}")
        
        for exp in exps:
            keep_ratio = exp['keep_ratio']
            output = exp['runs'][0]['output']
            tokens = f"{exp['runs'][0]['num_visual_tokens_pruned']}/{exp['runs'][0]['num_visual_tokens_original']}"
            
            print(f"\n  📌 Keep {keep_ratio*100:.0f}% (Tokens: {tokens}):")
            print(f"  {output[:200]}{'...' if len(output) > 200 else ''}\n")
    
    print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate different token pruning methods for VLM optimization")
    parser.add_argument("--model-path", type=str, default="checkpoints/llava-fastvithd_0.5b_stage3")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="img/netflix.jpg", help="Location of image file")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="Prompt for VLM")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--retention-ratios", type=float, nargs='+', default=[0.3, 0.5, 0.7], 
                        help="Token retention ratios to test (e.g., 0.3 0.5 0.7)")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--output-file", type=str, default="pruning_results.json", help="Output file for results (JSON)")
    args = parser.parse_args()
    
    # Initialize model cache
    run_inference.model_cache = None
    
    # Run evaluation
    evaluate_all_methods(args)

#!/usr/bin/env python3
import os
import argparse
import time
import json
import glob
from pathlib import Path
import statistics
import random

import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Import LLAVA modules
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# ============================================================================
# GESTORE COCO DATASET
# ============================================================================

class COCOLoader:
    def __init__(self, annotation_path, image_dir):
        print(f"📂 Loading COCO annotations from {annotation_path}...")
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"❌ File non trovato: {annotation_path}")
            exit(1)
        
        # Crea mappa image_filename -> list_of_captions
        self.img_map = {}
        id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id in id_to_filename:
                fname = id_to_filename[img_id]
                if fname not in self.img_map:
                    self.img_map[fname] = []
                self.img_map[fname].append(ann['caption'])
        
        self.image_dir = image_dir
        self.available_images = list(self.img_map.keys())
        print(f"✅ Loaded {len(self.available_images)} images with captions.")

    def get_ground_truth(self, filename):
        return self.img_map.get(filename, [])

# ============================================================================
# METRICHE E MODELLI
# ============================================================================

class Evaluator:
    def __init__(self):
        print("📥 Loading sentence-transformer...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_similarity(self, generated_text, reference_texts):
        if isinstance(reference_texts, str):
            reference_texts = [reference_texts]
            
        emb_gen = self.model.encode(generated_text, convert_to_tensor=True)
        emb_refs = self.model.encode(reference_texts, convert_to_tensor=True)
        
        cosine_scores = util.pytorch_cos_sim(emb_gen, emb_refs)
        best_score = torch.max(cosine_scores).item()
        return best_score

# ============================================================================
# PRUNING METHODS
# ============================================================================

def attention_based_pruning(vision_features, attention_maps, keep_ratio):
    if attention_maps is None: return vision_features, None
    attn_avg = attention_maps.mean(dim=1)[0]
    token_importance = attn_avg.mean(dim=0)
    k = max(1, int(token_importance.shape[0] * keep_ratio))
    _, top_k_indices = torch.topk(token_importance, k, dim=0)
    top_k_indices = top_k_indices.sort().values
    if vision_features.dim() == 3: pruned_features = vision_features[:, top_k_indices, :]
    else: pruned_features = vision_features[top_k_indices, :]
    return pruned_features, top_k_indices

def similarity_based_pruning(vision_features, keep_ratio, similarity_threshold=0.90):
    if vision_features.dim() == 3:
        batch_size, num_tokens, feat_dim = vision_features.shape
        features = vision_features[0]
    else:
        num_tokens, feat_dim = vision_features.shape
        features = vision_features
        batch_size = 1
    
    features_norm = F.normalize(features, p=2, dim=-1)
    similarity_matrix = torch.mm(features_norm, features_norm.t())
    
    merged_tokens = []
    processed = torch.zeros(num_tokens, dtype=torch.bool, device=features.device)
    target_num_tokens = max(1, int(num_tokens * keep_ratio))
    
    for i in range(num_tokens):
        if processed[i]: continue
        
        similar_mask = similarity_matrix[i] > similarity_threshold
        similar_mask[processed] = False
        similar_indices = similar_mask.nonzero(as_tuple=True)[0]
        
        if len(similar_indices) > 0:
            cluster_tokens = features[similar_indices]
            merged_token = cluster_tokens.mean(dim=0)
            merged_tokens.append(merged_token)
            processed[similar_indices] = True
        else:
            merged_tokens.append(features[i])
            processed[i] = True
        
        if len(merged_tokens) >= target_num_tokens: break
    
    if len(merged_tokens) > target_num_tokens:
        merged_tensor = torch.stack(merged_tokens)
        norms = torch.norm(merged_tensor, p=2, dim=-1)
        _, top_indices = torch.topk(norms, target_num_tokens)
        top_indices = top_indices.sort().values
        merged_tensor = merged_tensor[top_indices]
    else:
        merged_tensor = torch.stack(merged_tokens)
    
    if batch_size > 1 or vision_features.dim() == 3:
        merged_tensor = merged_tensor.unsqueeze(0)
    
    return merged_tensor, None

def norm_based_pruning(vision_features, keep_ratio):
    if vision_features.dim() == 3: features = vision_features[0]
    else: features = vision_features
    token_norms = torch.norm(features, p=2, dim=-1)
    k = max(1, int(token_norms.shape[0] * keep_ratio))
    _, top_k_indices = torch.topk(token_norms, k, dim=0)
    top_k_indices = top_k_indices.sort().values
    pruned_features = features[top_k_indices]
    if vision_features.dim() == 3: pruned_features = pruned_features.unsqueeze(0)
    return pruned_features, top_k_indices

# ============================================================================
# INFERENCE ENGINE
# ============================================================================

def run_inference_single(model_data, image_path, args, pruning_method='none', keep_ratio=1.0):
    tokenizer, model, image_processor, device = model_data
    
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device) 
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_inputs = image_tensor.unsqueeze(0).to(device=device, dtype=model.dtype)
    
    pre_encoded_features = None
    
    if pruning_method != 'none':
        vision_tower = model.get_model().get_vision_tower()
        with torch.inference_mode():
            result = vision_tower.forward_images(image_inputs, return_attention_maps=(pruning_method == 'attention'))
            if isinstance(result, tuple): vf, attn = result
            else: vf, attn = result, None
            
        if pruning_method == 'attention':
            pre_encoded_features, _ = attention_based_pruning(vf, attn, keep_ratio)
        elif pruning_method == 'similarity':
            pre_encoded_features, _ = similarity_based_pruning(vf, keep_ratio)
        elif pruning_method == 'norm':
            pre_encoded_features, _ = norm_based_pruning(vf, keep_ratio)
    
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, 
            images=image_inputs, 
            image_sizes=[image.size],
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature, 
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pre_encoded_features=pre_encoded_features if pruning_method != 'none' else None
        )
    total_time = time.perf_counter() - start
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    n_orig = pre_encoded_features.shape[1] if pre_encoded_features is not None else None
    n_pruned = pre_encoded_features.shape[1] if pre_encoded_features is not None else None
    
    if n_orig is None or n_pruned is None:
        with torch.inference_mode():
            vf_full = model.encode_images(image_inputs)
        n_orig = vf_full.shape[1]
        n_pruned = n_orig

    return {'output': outputs, 'time': total_time, 'n_orig': n_orig, 'n_pruned': n_pruned}

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_coco(args):
    # Init Components
    coco = COCOLoader(args.coco_json, args.image_dir)
    evaluator = Evaluator()
    
    # Init Model
    disable_torch_init()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, get_model_name_from_path(args.model_path), device=device.type
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model_data = (tokenizer, model, image_processor, device)

    # Select random subset
    image_list = coco.available_images
    if args.max_images and len(image_list) > args.max_images:
        image_list = random.sample(image_list, args.max_images)

    results = []
    
    # [MODIFICA] Lista per salvare i log dettagliati delle caption
    detailed_logs = []
    
    print(f"\n🚀 Starting Evaluation on {len(image_list)} images...")
    
    method_info = {
        'attention': {'emoji': '🎯', 'name': 'Attention'},
        'similarity': {'emoji': '🔗', 'name': 'Similarity'},
        'norm': {'emoji': '📊', 'name': 'Norm-Based'}
    }

    pbar = tqdm(image_list, unit="img")
    
    for idx, img_name in enumerate(pbar):
        img_path = os.path.join(args.image_dir, img_name)
        if not os.path.exists(img_path): continue
        
        pbar.set_description(f"Processing {img_name}")
        ground_truths = coco.get_ground_truth(img_name)
        
        # [MODIFICA] Struttura per il log JSON di questa singola immagine
        img_log = {
            'image_id': img_name,
            'ground_truth': ground_truths,
            'baseline': {},
            'experiments': []
        }
        
        tqdm.write(f"\n[{idx+1}/{len(image_list)}] Immagine: {img_name}")
        
        # --- 1. BASELINE RUN ---
        res_base = run_inference_single(model_data, img_path, args, pruning_method='none')
        score_base_vs_gt = evaluator.compute_similarity(res_base['output'], ground_truths)

        tqdm.write(
            f"  🔵 Baseline (R=1.0) "
            f"| Time: {res_base['time']:.3f}s "
            f"| GT-Acc: {score_base_vs_gt:.3f}"
        )
        
        # [MODIFICA] Salvo dati baseline nel log JSON
        img_log['baseline'] = {
            'caption': res_base['output'],
            'time': res_base['time'],
            'accuracy': score_base_vs_gt
        }
        
        # [MODIFICA] Aggiungo generated_caption e ground_truth al CSV results
        results.append({
            'image': img_name,
            'method': 'baseline',
            'ratio': 1.0,
            'speedup': 1.0,
            'sim_vs_baseline': 1.0,
            'sim_vs_gt': score_base_vs_gt,
            'baseline_acc': score_base_vs_gt,
            'time': res_base['time'],
            'generated_caption': res_base['output'],
            'ground_truth': " | ".join(ground_truths) # Flatten list for CSV
        })
        
        # --- 2. PRUNING RUNS ---
        methods = ['attention', 'similarity', 'norm']
        
        for method in methods:
            info = method_info[method]
            
            for ratio in args.retention_ratios:
                res_pruned = run_inference_single(model_data, img_path, args, method, ratio)
                
                score_pruned_vs_base = evaluator.compute_similarity(res_pruned['output'], res_base['output'])
                score_pruned_vs_gt = evaluator.compute_similarity(res_pruned['output'], ground_truths)
                
                speedup = res_base['time'] / res_pruned['time']
                tqdm.write(
                    f"  {info['emoji']} {info['name']} (R={ratio:.2f}) "
                    f"| GT-Acc: {score_pruned_vs_gt:.3f}"
                )
                
                # [MODIFICA] Aggiungo esperimento al log JSON
                img_log['experiments'].append({
                    'method': method,
                    'ratio': ratio,
                    'caption': res_pruned['output'],
                    'time': res_pruned['time'],
                    'speedup': speedup,
                    'accuracy_vs_gt': score_pruned_vs_gt,
                    'similarity_vs_baseline': score_pruned_vs_base
                })

                results.append({
                    'image': img_name,
                    'method': method,
                    'ratio': ratio,
                    'speedup': speedup,
                    'sim_vs_baseline': score_pruned_vs_base,
                    'sim_vs_gt': score_pruned_vs_gt,
                    'baseline_acc': score_base_vs_gt,
                    'time': res_pruned['time'],
                    'generated_caption': res_pruned['output'],
                    'ground_truth': " | ".join(ground_truths)
                })

        # Aggiungi il log dell'immagine corrente alla lista generale
        detailed_logs.append(img_log)

    # Export & Plot
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Salva CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, 'coco_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Risultati CSV salvati in {csv_path}")
    
    # 2. Salva JSON Dettagliato (Captions)
    json_path = os.path.join(args.output_dir, 'captions_comparison.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_logs, f, indent=4, ensure_ascii=False)
    print(f"📝 Captions JSON salvate in {json_path}")
    
    plot_results(df, args.output_dir)


def plot_results(df, out_dir):
    """Genera i grafici richiesti a partire dal dataframe dei risultati."""
    sns.set_theme(style="whitegrid")
    os.makedirs(out_dir, exist_ok=True)

    baseline_rows = df[df['method'] == 'baseline']
    baseline_acc = baseline_rows['sim_vs_gt'].mean() if not baseline_rows.empty else None
    df_methods = df[df['method'] != 'baseline'].copy() # Usa .copy() per evitare SettingWithCopyWarning

    palette = {"attention": "blue", "similarity": "orange", "norm": "green"}

    # --- 1. Original Similarity vs Retention ---
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sns.lineplot(data=df_methods, x='ratio', y='sim_vs_gt', hue='method', style='method', markers=True, dashes=False, palette=palette, ax=ax)
    if baseline_acc is not None:
        ax.axhline(y=baseline_acc, color='black', linestyle='--', label='Baseline Accuracy')
    ax.set_title('Accuracy vs Token Retention Ratio')
    ax.set_xlabel('Retention Ratio (Lower is more pruning)')
    ax.set_ylabel('Similarity with Ground Truth')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_vs_ratio.png'))
    plt.close()

    # --- 2. Pareto Frontier (Speedup vs Accuracy) ---
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sns.scatterplot(data=df_methods, x='speedup', y='sim_vs_gt', hue='method', style='ratio', s=100, ax=ax, palette=palette)
    if baseline_acc is not None:
        ax.scatter([1.0], [baseline_acc], color='black', s=150, label='Baseline', marker='X')

    ax.set_title('Trade-off: Speedup vs Accuracy')
    ax.set_xlabel('Speedup Factor (x)')
    ax.set_ylabel('Similarity with Ground Truth')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tradeoff_pareto_labeled.png'))
    plt.close()

    # --- 3. NUOVO: Pareto Frontier (Similarità vs. Speedup) ---
    
    # Calcola le medie per ogni metodo e retention ratio
    df_mean = df_methods.groupby(['method', 'ratio']).agg(
        mean_speedup=('speedup', 'mean'),
        mean_similarity=('sim_vs_baseline', 'mean') # Usiamo 'sim_vs_baseline' direttamente!
    ).reset_index()

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Scatter plot dei punti medi
    sns.scatterplot(
        data=df_mean, 
        x='mean_speedup', 
        y='mean_similarity', 
        hue='method', 
        style='method', 
        s=150, 
        ax=ax, 
        palette=palette,
        legend='full'
    )
    
    # Aggiungi le label Ratio (R) a ciascun punto
    for line in range(0, df_mean.shape[0]):
         ratio = df_mean['ratio'].iloc[line]
         plt.text(
             df_mean['mean_speedup'].iloc[line] + 0.05, 
             df_mean['mean_similarity'].iloc[line], 
             f'R={ratio:.2f}', 
             horizontalalignment='left', 
             size='small', 
             color='black', 
             weight='semibold'
         )

    # Punto Baseline (1.0 Similarità, 1.0 Speedup)
    ax.scatter([1.0], [1.0], color='black', s=150, label='Baseline', marker='X')
    
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylim(bottom=df_mean['mean_similarity'].min() - 0.05, top=1.05) # Regola l'asse Y per chiarezza

    ax.set_title('Trade-off: Similarità alla Baseline vs. Speedup (Media per Ratio)')
    ax.set_xlabel('Speedup Factor (x)')
    ax.set_ylabel(r'Similarità alla Baseline ($\text{Sim}_{\text{Pruned vs Base}}$)')
    ax.legend(title='Pruning Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tradeoff_similarity_vs_speedup_mean.png'))
    plt.close()
    
    print("📊 Grafici generati in output.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava-fastvithd_0.5b_stage3")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--coco-json", type=str, default="data/coco/annotations/captions_val2017.json")
    parser.add_argument("--image-dir", type=str, default="data/coco/val2017")
    parser.add_argument("--output-dir", type=str, default="results_coco")
    parser.add_argument("--prompt", type=str, default="Describe this image concisely.")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=50, help="Numero immagini")
    parser.add_argument("--retention-ratios", type=float, nargs='+', default=[0.3, 0.5, 0.7])
    
    args = parser.parse_args()
    evaluate_coco(args)
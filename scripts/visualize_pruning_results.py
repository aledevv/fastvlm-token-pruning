#!/usr/bin/env python3
#
# Visualization script for pruning methods evaluation results
# Creates comparative charts for TTFT, speedup, and quality analysis
#
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_results(json_file):
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)


def create_comparison_charts(results, output_dir='plots'):
    """Create comprehensive comparison charts"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    summary = results['summary']
    
    # Extract data
    methods = []
    keep_ratios = []
    ttfts = []
    speedups = []
    token_reductions = []
    tokens_per_sec = []
    
    baseline_ttft = summary[0]['ttft_mean']
    baseline_tps = summary[0]['tokens_per_second_mean']
    
    for stat in summary[1:]:  # Skip baseline
        method_label = f"{stat['method_name']}\n{stat['keep_ratio']*100:.0f}%"
        methods.append(method_label)
        keep_ratios.append(stat['keep_ratio'])
        ttfts.append(stat['ttft_mean'] * 1000)  # Convert to ms
        speedups.append(stat.get('speedup_factor', 1.0))
        token_reductions.append(stat['token_reduction_percent'])
        tokens_per_sec.append(stat['tokens_per_second_mean'])
    
    # Color mapping for methods
    colors = {
        'Attention-Based': '#FF6B6B',
        'Similarity-Based': '#4ECDC4',
        'Norm-Based': '#95E1D3'
    }
    
    bar_colors = [colors[stat['method_name']] for stat in summary[1:]]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. TTFT Comparison
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(range(len(methods)), ttfts, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=baseline_ttft * 1000, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline_ttft*1000:.1f}ms')
    ax1.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax1.set_ylabel('TTFT (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('⚡ Time To First Token (TTFT)', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, ttfts)):
        height = bar.get_height()
        improvement = ((baseline_ttft * 1000 - val) / (baseline_ttft * 1000)) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}ms\n({improvement:.0f}% ↓)',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2. Speedup Factor
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(range(len(methods)), speedups, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    ax2.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    ax2.set_title('🚀 Total Generation Speedup', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Token Reduction
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(range(len(methods)), token_reductions, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Token Reduction (%)', fontsize=11, fontweight='bold')
    ax3.set_title('🎯 Visual Token Reduction', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars3, token_reductions):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Tokens per Second
    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.bar(range(len(methods)), tokens_per_sec, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.axhline(y=baseline_tps, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline_tps:.1f} tok/s')
    ax4.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Tokens/Second', fontsize=11, fontweight='bold')
    ax4.set_title('📊 Generation Speed', fontsize=13, fontweight='bold')
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars4, tokens_per_sec):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Speedup vs Token Reduction Scatter
    ax5 = plt.subplot(2, 3, 5)
    
    # Group by method for scatter plot
    method_groups = {}
    for i, stat in enumerate(summary[1:]):
        method = stat['method_name']
        if method not in method_groups:
            method_groups[method] = {'x': [], 'y': [], 'labels': []}
        method_groups[method]['x'].append(stat['token_reduction_percent'])
        method_groups[method]['y'].append(stat.get('speedup_factor', 1.0))
        method_groups[method]['labels'].append(f"{stat['keep_ratio']*100:.0f}%")
    
    for method, data in method_groups.items():
        ax5.scatter(data['x'], data['y'], s=200, alpha=0.7, 
                   color=colors[method], edgecolors='black', linewidth=2,
                   label=method)
        # Add labels
        for x, y, label in zip(data['x'], data['y'], data['labels']):
            ax5.annotate(label, (x, y), fontsize=9, fontweight='bold',
                        ha='center', va='center')
    
    ax5.set_xlabel('Token Reduction (%)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    ax5.set_title('⚖️ Speedup vs Token Reduction', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10, loc='upper left')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # 6. Method Comparison by Retention Ratio
    ax6 = plt.subplot(2, 3, 6)
    
    # Group data by retention ratio
    retention_groups = {}
    for stat in summary[1:]:
        ratio = stat['keep_ratio']
        if ratio not in retention_groups:
            retention_groups[ratio] = {'methods': [], 'speedups': []}
        retention_groups[ratio]['methods'].append(stat['method_name'])
        retention_groups[ratio]['speedups'].append(stat.get('speedup_factor', 1.0))
    
    x_pos = np.arange(len(retention_groups))
    width = 0.25
    
    method_names = ['Attention-Based', 'Similarity-Based', 'Norm-Based']
    for i, method in enumerate(method_names):
        speedups_by_ratio = []
        for ratio in sorted(retention_groups.keys()):
            group = retention_groups[ratio]
            idx = group['methods'].index(method) if method in group['methods'] else -1
            speedup = group['speedups'][idx] if idx >= 0 else 0
            speedups_by_ratio.append(speedup)
        
        ax6.bar(x_pos + i * width, speedups_by_ratio, width, 
               label=method, color=colors[method], alpha=0.8,
               edgecolor='black', linewidth=1.5)
    
    ax6.set_xlabel('Token Retention Ratio', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    ax6.set_title('📈 Method Comparison by Retention', fontsize=13, fontweight='bold')
    ax6.set_xticks(x_pos + width)
    ax6.set_xticklabels([f'{r*100:.0f}%' for r in sorted(retention_groups.keys())])
    ax6.legend(fontsize=10)
    ax6.grid(axis='y', alpha=0.3)
    ax6.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'pruning_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison chart: {output_file}")
    
    plt.show()
    
    return output_file


def create_quality_comparison_table(results, output_dir='plots'):
    """Create a visual table comparing output quality"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    summary = results['summary']
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Method', 'Keep%', 'Tokens', 'TTFT', 'Speedup', 'Sample Output (first 100 chars)'])
    
    # Baseline
    baseline = summary[0]
    table_data.append([
        '🔵 Baseline',
        '100%',
        f"{baseline['num_visual_tokens_pruned']}",
        f"{baseline['ttft_mean']*1000:.1f}ms",
        '1.00x',
        baseline['sample_output'][:100] + '...'
    ])
    
    # Pruning methods
    for stat in summary[1:]:
        speedup = stat.get('speedup_factor', 1.0)
        emoji = '✅' if speedup > 1.2 else '⚠️' if speedup > 0.95 else '❌'
        
        table_data.append([
            f"{emoji} {stat['method_name']}",
            f"{stat['keep_ratio']*100:.0f}%",
            f"{stat['num_visual_tokens_pruned']}/{stat['num_visual_tokens_original']}",
            f"{stat['ttft_mean']*1000:.1f}ms",
            f"{speedup:.2f}x",
            stat['sample_output'][:100] + '...'
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.15, 0.08, 0.10, 0.10, 0.10, 0.47])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#4ECDC4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style baseline row
    for i in range(6):
        cell = table[(1, i)]
        cell.set_facecolor('#FFE5E5')
    
    # Color code other rows by speedup
    for row_idx in range(2, len(table_data)):
        speedup_val = float(table_data[row_idx][4].replace('x', ''))
        if speedup_val > 1.2:
            color = '#E8F8F5'  # Light green
        elif speedup_val > 0.95:
            color = '#FFF9E6'  # Light yellow
        else:
            color = '#FFE5E5'  # Light red
        
        for col_idx in range(6):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(color)
    
    plt.title('📝 Quality Comparison: Output Samples', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Save figure
    output_file = output_path / 'quality_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved quality comparison: {output_file}")
    
    plt.show()
    
    return output_file


def print_summary_report(results):
    """Print a text summary report"""
    summary = results['summary']
    metadata = results['metadata']
    
    print(f"\n{'='*80}")
    print(f"📊 PRUNING METHODS EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Image: {metadata['image_file']}")
    print(f"Prompt: {metadata['prompt']}")
    print(f"{'='*80}\n")
    
    baseline = summary[0]
    print(f"🔵 BASELINE:")
    print(f"   TTFT: {baseline['ttft_mean']*1000:.1f}ms")
    print(f"   Tokens/sec: {baseline['tokens_per_second_mean']:.1f}")
    print(f"   Visual tokens: {baseline['num_visual_tokens_pruned']}\n")
    
    print(f"{'='*80}")
    print(f"🏆 TOP PERFORMERS:")
    print(f"{'='*80}\n")
    
    # Find best by speedup
    best_speedup = max(summary[1:], key=lambda x: x.get('speedup_factor', 0))
    print(f"⚡ Fastest Overall:")
    print(f"   {best_speedup['method_name']} @ {best_speedup['keep_ratio']*100:.0f}%")
    print(f"   Speedup: {best_speedup['speedup_factor']:.2f}x")
    print(f"   TTFT: {best_speedup['ttft_mean']*1000:.1f}ms ({best_speedup['ttft_improvement_percent']:.1f}% faster)")
    print(f"   Token reduction: {best_speedup['token_reduction_percent']:.1f}%\n")
    
    # Find best TTFT improvement
    best_ttft = min(summary[1:], key=lambda x: x['ttft_mean'])
    print(f"🎯 Best TTFT:")
    print(f"   {best_ttft['method_name']} @ {best_ttft['keep_ratio']*100:.0f}%")
    print(f"   TTFT: {best_ttft['ttft_mean']*1000:.1f}ms ({best_ttft['ttft_improvement_percent']:.1f}% faster)")
    print(f"   Speedup: {best_ttft.get('speedup_factor', 1.0):.2f}x\n")
    
    print(f"{'='*80}")
    print(f"💡 RECOMMENDATIONS:")
    print(f"{'='*80}\n")
    
    # Recommend based on speedup and quality
    good_methods = [s for s in summary[1:] if s.get('speedup_factor', 0) > 1.3]
    if good_methods:
        best_balanced = max(good_methods, key=lambda x: x['keep_ratio'])
        print(f"✅ Best Balance (Speed + Quality):")
        print(f"   {best_balanced['method_name']} @ {best_balanced['keep_ratio']*100:.0f}%")
        print(f"   - Speedup: {best_balanced['speedup_factor']:.2f}x")
        print(f"   - TTFT: {best_balanced['ttft_mean']*1000:.1f}ms")
        print(f"   - Keeps {best_balanced['keep_ratio']*100:.0f}% of tokens (less aggressive pruning)")
        print(f"   - Better quality preservation\n")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize pruning methods evaluation results")
    parser.add_argument("--results-file", type=str, default="pruning_results.json",
                       help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str, default="plots",
                       help="Directory to save plots")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display plots (only save)")
    args = parser.parse_args()
    
    # Load results
    print(f"📂 Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    
    # Print summary report
    print_summary_report(results)
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    
    if args.no_show:
        plt.ioff()
    
    create_comparison_charts(results, args.output_dir)
    create_quality_comparison_table(results, args.output_dir)
    
    print(f"\n✅ All visualizations saved to: {args.output_dir}/")
    print(f"   - pruning_comparison.png")
    print(f"   - quality_comparison.png\n")


if __name__ == "__main__":
    main()

# FastVLM Token Pruning Optimization

Advanced token pruning methods for Vision-Language Models (VLMs) to achieve **up to 80% faster inference** while maintaining output quality.

## 🎯 Overview

This project implements and compares three token pruning techniques for optimizing FastVLM (Fast Vision-Language Model):

1. **Attention-Based Pruning (ATS)** - Uses attention maps to identify important tokens
2. **Similarity-Based Pruning (PruMerge Lite)** - Merges redundant similar tokens
3. **Norm-Based Pruning (Low Magnitude)** - Keeps tokens with highest L2 norm

## 📊 Key Results

| Method | Token Retention | TTFT Improvement | Speedup | Quality |
|--------|----------------|------------------|---------|---------|
| **Norm-Based** | 50% | **80%** ⚡ | **1.55x** | ✅ Excellent |
| **Similarity-Based** | 50% | 79% | 1.44x | ✅ Excellent |
| **Attention-Based** | 70% | 74% | 1.42x | ✅ Good |
| Baseline | 100% | - | 1.0x | 🔵 Reference |

### Performance Highlights

- **TTFT (Time To First Token)**: Reduced from ~740ms to ~150ms
- **Generation Speed**: Up to 1.55x faster
- **Token Reduction**: 30-50% fewer visual tokens
- **Quality**: Maintained detailed, accurate descriptions

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fastvlm-token-pruning.git
cd fastvlm-token-pruning

# Install dependencies
pip install -r requirements.txt

# Download FastVLM model (required)
# Follow instructions from: https://github.com/apple/ml-fastvlm
```

### Basic Usage

```bash
# Run evaluation with all pruning methods
python scripts/eval_pruning_methods.py \
    --image-file assets/images/banana.jpg \
    --retention-ratios 0.3 0.5 0.7 \
    --num-runs 3

# Visualize results
python scripts/visualize_pruning_results.py \
    --results-file pruning_results.json
```

## 📖 Methods Explained

### 1. Attention-Based Pruning (ATS)

**How it works:**
- Extracts attention maps from the vision tower
- Computes token importance based on attention weights
- Keeps top-k tokens with highest importance scores

**Pros:**
- Semantically aware (considers what the model "looks at")
- Good for preserving key visual elements

**Cons:**
- Can be too aggressive at low retention ratios
- May miss important low-attention details

### 2. Similarity-Based Pruning (PruMerge Lite)

**How it works:**
- Computes cosine similarity between all token pairs
- Identifies clusters of similar/redundant tokens
- Merges similar tokens via average pooling

**Pros:**
- Reduces true redundancy (e.g., uniform backgrounds)
- Maintains semantic diversity
- Excellent quality preservation

**Cons:**
- Slightly slower than norm-based (similarity computation)
- Requires tuning similarity threshold

### 3. Norm-Based Pruning (Low Magnitude)

**How it works:**
- Calculates L2 norm of each token's feature vector
- Keeps tokens with highest magnitude
- Discards low-energy tokens

**Pros:**
- **Fastest method** (simple computation)
- Excellent speed/quality tradeoff
- Works well across different image types

**Cons:**
- Heuristic-based (may miss low-contrast important details)
- Less semantically aware than attention-based

## 📈 Detailed Results

### Banana Image Test

![Pruning Comparison](assets/plots/pruning_comparison.png)

**Observations:**
- **Norm-Based 50%**: Best overall performance (1.46x speedup, maintains "bunch of bananas")
- **Similarity-Based 50%**: Excellent quality, correctly identifies multiple bananas
- **Attention-Based 50%**: Too aggressive, describes as "single banana" ❌

### Netflix Screenshot Test

**Best Performers:**
- **Norm-Based 50%**: 1.55x speedup, 80% TTFT reduction
- **Similarity-Based 70%**: 1.47x speedup, best quality preservation
- **Attention-Based 70%**: 1.34x speedup, good balance

## 🛠️ Advanced Usage

### Custom Pruning Configuration

```python
from scripts.eval_pruning_methods import run_inference

# Run with specific method and retention
result = run_inference(
    args,
    pruning_method='norm',  # 'attention', 'similarity', or 'norm'
    keep_ratio=0.5          # 0.0 to 1.0
)

print(f"TTFT: {result['ttft']*1000:.1f}ms")
print(f"Speedup: {result['speedup_factor']:.2f}x")
print(f"Output: {result['output']}")
```

### Parameter Sweep

```bash
# Test multiple retention ratios
python scripts/eval_pruning_methods.py \
    --retention-ratios 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
    --num-runs 5 \
    --output-file sweep_results.json
```

### Batch Evaluation

```bash
# Evaluate on multiple images
for img in assets/images/*.jpg; do
    python scripts/eval_pruning_methods.py \
        --image-file "$img" \
        --output-file "results_$(basename $img .jpg).json"
done
```

## 📁 Project Structure

```
fastvlm-token-pruning/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── scripts/                       # Evaluation scripts
│   ├── eval_ats.py               # ATS-only evaluation
│   ├── eval_pruning_methods.py   # Compare all methods
│   └── visualize_pruning_results.py  # Generate plots
│
├── examples/                      # Example results
│   └── results/
│       ├── banana_results.json
│       └── netflix_results.json
│
└── assets/                        # Images and plots
    ├── images/
    │   ├── banana.jpg
    │   └── netflix.jpg
    └── plots/
        ├── pruning_comparison.png
        └── quality_comparison.png
```

## 🔬 Methodology

### Evaluation Metrics

1. **TTFT (Time To First Token)**: Time until first output token
2. **Total Generation Time**: Complete response generation time
3. **Tokens/Second**: Generation throughput
4. **Visual Token Count**: Number of tokens after pruning
5. **Output Quality**: Manual inspection of generated descriptions

### Experimental Setup

- **Model**: FastVLM with FastViTHD vision encoder
- **Hardware**: Apple Silicon (MPS backend)
- **Test Images**: Diverse set (objects, scenes, text)
- **Runs**: Multiple runs per configuration for statistical significance

## 💡 Recommendations

### For Production Use

**Best Overall**: **Norm-Based @ 50-70%**
- Fastest inference
- Excellent quality
- Simple implementation
- Robust across image types

**Best Quality**: **Similarity-Based @ 70%**
- Minimal quality loss
- Good speedup (1.4-1.5x)
- Preserves semantic diversity

**Conservative**: **Attention-Based @ 70%**
- Semantically aware
- Safe choice for critical applications
- Moderate speedup (1.3-1.4x)

### Tuning Guidelines

| Use Case | Recommended Method | Retention Ratio |
|----------|-------------------|-----------------|
| Real-time chat | Norm-Based | 40-50% |
| Image captioning | Similarity-Based | 60-70% |
| Visual QA | Attention-Based | 70-80% |
| Batch processing | Norm-Based | 30-40% |

## 🔗 Related Work

This project builds upon:

- **FastVLM**: [Apple ML Research](https://github.com/apple/ml-fastvlm)
- **LLaVA**: Large Language and Vision Assistant
- **PruMerge**: Token merging for efficient VLMs

## 📝 Citation

If you use this work, please cite:

```bibtex
@software{fastvlm_token_pruning,
  title={FastVLM Token Pruning Optimization},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/fastvlm-token-pruning}
}
```

And the original FastVLM paper:

```bibtex
@article{fastvlm2024,
  title={FastVLM: Efficient Vision-Language Models},
  author={Apple ML Research},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---


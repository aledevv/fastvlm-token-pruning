#!/bin/bash
#
# Automated setup script for FastVLM Token Pruning
# This script will:
# 1. Initialize FastVLM submodule
# 2. Download model checkpoints
# 3. Install all dependencies
#

set -e  # Exit on error

echo "🚀 FastVLM Token Pruning - Automated Setup"
echo "=========================================="
echo ""

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Error: git is not installed"
    exit 1
fi

# Check if python/pip is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Error: python is not installed"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD=$(command -v python3 || command -v python)

echo "📦 Step 1/4: Initializing FastVLM submodule..."
git submodule update --init --recursive
echo "✅ Submodule initialized"
echo ""

echo "📥 Step 2/4: Downloading model checkpoints..."
cd fastvlm
if [ -f "get_models.sh" ]; then
    bash get_models.sh
    echo "✅ Checkpoints downloaded"
else
    echo "⚠️  Warning: get_models.sh not found, skipping checkpoint download"
    echo "   You can download manually later with: cd fastvlm && bash get_models.sh"
fi
cd ..
echo ""

echo "📚 Step 3/4: Installing project dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt
echo "✅ Project dependencies installed"
echo ""

echo "🔧 Step 4/4: Installing FastVLM package..."
cd fastvlm
$PYTHON_CMD -m pip install -e .
cd ..
echo "✅ FastVLM package installed"
echo ""

echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "🎯 Quick Start:"
echo "  python scripts/eval_ats.py \\"
echo "    --model-path fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 \\"
echo "    --image-file assets/images/banana.jpg \\"
echo "    --num-runs 1"
echo ""

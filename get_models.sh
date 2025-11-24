#!/usr/bin/env bash
# For licensing see accompanying LICENSE_MODEL file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

set -euo pipefail

# macOS ships curl and unzip by default; verify they exist.
for tool in curl unzip; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: $tool not found. Install it (e.g., Xcode Command Line Tools or Homebrew) and retry." >&2
        exit 1
    fi
done

BASE_URL="https://ml-site.cdn-apple.com/datasets/fastvlm"
MODELS=(
    "llava-fastvithd_0.5b_stage2"
    "llava-fastvithd_0.5b_stage3"
    # "llava-fastvithd_1.5b_stage2"
    # "llava-fastvithd_1.5b_stage3"
    # "llava-fastvithd_7b_stage2"
    # "llava-fastvithd_7b_stage3"
)

mkdir -p checkpoints

# Download with curl (works on macOS). Resume if partially downloaded.
for m in "${MODELS[@]}"; do
    url="${BASE_URL}/${m}.zip"
    out="checkpoints/${m}.zip"
    echo "Downloading ${m}..."
    curl -L -f --retry 3 --retry-delay 5 -C - -o "${out}" "${url}"
done

# Extract models and clean up
(
    cd checkpoints
    for m in "${MODELS[@]}"; do
        unzip -q -o "${m}.zip"
        rm -f "${m}.zip"
    done
)

echo "All models downloaded and extracted to checkpoints/"

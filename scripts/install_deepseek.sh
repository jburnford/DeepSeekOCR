#!/bin/bash
#SBATCH --job-name=deepseek_install
#SBATCH --account=def-jic823
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=%x-%j.out

# DeepSeek-OCR Installation Script for DRAC/Alliance Clusters (Nibi)
# This script sets up DeepSeek-OCR following Alliance best practices

set -e  # Exit on error

echo "=========================================="
echo "DeepSeek-OCR Installation - Nibi Cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Module loading (DRAC/Alliance standard environment)
echo "[1/7] Loading modules..."
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.12

echo "Active modules:"
module list

# Create virtual environment using Alliance best practices
VENV_DIR="${HOME}/projects/def-jic823/deepseek_venv"
echo "[2/7] Creating virtual environment at $VENV_DIR..."

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Removing and recreating..."
    rm -rf "$VENV_DIR"
fi

# Use virtualenv with --no-download (Alliance best practice)
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip (CRITICAL: use --no-index for Alliance clusters)
echo "[3/7] Upgrading pip using Alliance wheelhouse..."
pip install --no-index --upgrade pip setuptools wheel

# Check what's available in Alliance wheelhouse
echo "[4/7] Checking Alliance wheelhouse for available packages..."
echo "Note: Some packages may need external PyPI (torch, flash-attn)"

# Install PyTorch with CUDA support (requires external PyPI)
echo "[5/7] Installing PyTorch 2.6.0 with CUDA 12.1..."
# Note: PyTorch not in Alliance wheelhouse, must use external source
# This requires internet access from login node
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attention (requires CUDA toolkit and compilation)
echo "[6/7] Installing flash-attention..."
# flash-attn requires compilation, may need --no-index removed
MAX_JOBS=4 pip install flash-attn==2.7.3 --no-build-isolation

# Install remaining dependencies
echo "[7/7] Installing remaining dependencies..."
# Try Alliance wheelhouse first, fallback to PyPI if needed
echo "Attempting to install from Alliance wheelhouse first..."

# DeepSeek-OCR requires exactly transformers 4.46.3
pip install --no-index transformers==4.46.3 || pip install transformers==4.46.3
pip install --no-index tokenizers || pip install tokenizers==0.20.3
pip install --no-index einops addict easydict || pip install einops addict easydict
pip install --no-index Pillow pdf2image numpy tqdm || pip install Pillow pdf2image numpy tqdm
pip install --no-index pyyaml requests || pip install pyyaml requests

# These might not be in wheelhouse
pip install jiwer editdistance

# DeepSeek-OCR requires matplotlib
pip install --no-index matplotlib || pip install matplotlib

# Verify installation
echo ""
echo "=========================================="
echo "Installation Complete - Verification"
echo "=========================================="

python3 << 'PYEOF'
import sys
import torch
import transformers
import flash_attn

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Transformers version: {transformers.__version__}")
print(f"Flash-attention version: {flash_attn.__version__}")
PYEOF

echo ""
echo "=========================================="
echo "To use this environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo "=========================================="
echo "Installation finished: $(date)"

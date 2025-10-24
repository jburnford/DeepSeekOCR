#!/bin/bash
#SBATCH --job-name=deepseek_fixed
#SBATCH --account=def-jic823
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=%x-%j.out

# DeepSeek-OCR Fixed Version Test
# Tests the fixed processing script that properly captures OCR output

set -euo pipefail

echo "=========================================="
echo "DeepSeek-OCR Fixed Version Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Directories
WORK_DIR="${HOME}/projects/def-jic823/deepseek-ocr-test"
IMAGES_DIR="${HOME}/ocr_bldata/25439023/BLN600/Images"
GT_DIR="${HOME}/ocr_bldata/25439023/BLN600/Ground Truth"
RESULTS_DIR="${WORK_DIR}/results/bl_newspapers_$(date +%Y%m%d_%H%M%S)_FIXED"

echo "Configuration:"
echo "  Work directory: $WORK_DIR"
echo "  Images: $IMAGES_DIR"
echo "  Ground Truth: $GT_DIR"
echo "  Results: $RESULTS_DIR"
echo ""

# Verify directories
if [[ ! -d "$IMAGES_DIR" ]]; then
    echo "Error: Images directory not found: $IMAGES_DIR"
    exit 1
fi

if [[ ! -d "$GT_DIR" ]]; then
    echo "Error: Ground truth directory not found: $GT_DIR"
    exit 1
fi

# Count images
IMAGE_COUNT=$(find "$IMAGES_DIR" -type f \( -name "*.tif" -o -name "*.jpg" -o -name "*.png" \) | wc -l)
echo "Found $IMAGE_COUNT images"
echo ""

# Load Python environment
module load python/3.11
source ~/projects/def-jic823/deepseek_venv/bin/activate

echo "Python environment:"
python --version
pip list | grep -E "torch|transformers|jiwer"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR/base_size_1024"

# Run fixed processing script
echo "Starting OCR processing..."
echo "=========================================="
echo ""

cd "$WORK_DIR"

# Process full dataset (all 600 images)
python3 scripts/process_bl_newspapers_fixed.py \
    --images "$IMAGES_DIR" \
    --ground-truth "$GT_DIR" \
    --output "$RESULTS_DIR/base_size_1024" \
    --base-size 1024 \
    --device cuda

echo ""
echo "=========================================="
echo "Processing complete!"
echo "End time: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="

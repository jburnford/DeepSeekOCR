#!/bin/bash
#SBATCH --job-name=deepseek_bl_news
#SBATCH --account=def-jic823
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=%x-%j.out

# DeepSeek-OCR British Library Newspapers Test
# Processes newspaper images and evaluates against ground truth

set -e

echo "=========================================="
echo "DeepSeek-OCR British Library Newspapers"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=========================================="

# Load modules
echo "Loading modules..."
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.12

# Activate virtual environment
VENV_DIR="${HOME}/projects/def-jic823/deepseek_venv"
echo "Activating virtual environment: $VENV_DIR"
source "$VENV_DIR/bin/activate"

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Configuration
REPO_DIR="${HOME}/projects/def-jic823/deepseek-ocr-test"
IMAGES_DIR="${HOME}/ocr_bldata/25439023/BLN600/Images"
GT_DIR="${HOME}/ocr_bldata/25439023/BLN600/Ground Truth"
RESULTS_DIR="${HOME}/projects/def-jic823/deepseek-ocr-test/results/bl_newspapers_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

echo ""
echo "Configuration:"
echo "  Repository: $REPO_DIR"
echo "  Images: $IMAGES_DIR"
echo "  Ground Truth: $GT_DIR"
echo "  Results: $RESULTS_DIR"
echo ""

# Check image count
IMAGE_COUNT=$(find "$IMAGES_DIR" -type f \( -name "*.tif" -o -name "*.jpg" -o -name "*.png" \) | wc -l)
echo "Found $IMAGE_COUNT images"
echo ""

# Process newspapers
cd "$REPO_DIR"

echo "Starting DeepSeek-OCR processing..."
echo "=========================================="

# Test with different model sizes for comparison
for BASE_SIZE in 1024; do
    echo ""
    echo "Testing with base_size=$BASE_SIZE"
    OUTPUT_DIR="$RESULTS_DIR/base_size_$BASE_SIZE"

    python3 scripts/process_bl_newspapers.py \
        --images "$IMAGES_DIR" \
        --ground-truth "$GT_DIR" \
        --output "$OUTPUT_DIR" \
        --base-size $BASE_SIZE \
        --device cuda

    echo "Completed base_size=$BASE_SIZE"
done

# Generate aggregate report
echo ""
echo "=========================================="
echo "Generating aggregate report..."
echo "=========================================="

SUMMARY_FILE="$RESULTS_DIR/aggregate_summary.txt"

cat > "$SUMMARY_FILE" << EOF
========================================
DeepSeek-OCR British Library Newspapers
Aggregate Results
========================================

Job Information:
  Job ID: $SLURM_JOB_ID
  Node: $SLURM_NODELIST
  Completed: $(date)

Dataset:
  British Library Newspapers BLN600
  Images processed: $IMAGE_COUNT
  Ground truth available: Yes

Model Configurations Tested:
EOF

for BASE_SIZE in 1024; do
    OUTPUT_DIR="$RESULTS_DIR/base_size_$BASE_SIZE"
    if [ -f "$OUTPUT_DIR/bl_newspapers_results.json" ]; then
        echo "" >> "$SUMMARY_FILE"
        echo "Base Size: $BASE_SIZE" >> "$SUMMARY_FILE"
        echo "---" >> "$SUMMARY_FILE"

        # Extract metrics using python
        python3 << PYEOF >> "$SUMMARY_FILE"
import json
with open('$OUTPUT_DIR/bl_newspapers_results.json') as f:
    data = json.load(f)
    meta = data['metadata']
    print(f"  Images processed: {meta.get('images_processed', 0)}")
    print(f"  Processing time: {meta.get('total_processing_time', 0):.2f}s")
    print(f"  Speed: {meta.get('images_per_second', 0):.4f} images/sec")
    if 'average_cer' in meta:
        print(f"  Average CER: {meta['average_cer']:.4f}")
        print(f"  Average WER: {meta['average_wer']:.4f}")
PYEOF
    fi
done

cat >> "$SUMMARY_FILE" << EOF

Output Location:
  $RESULTS_DIR

Files Generated:
  - Individual results: base_size_*/[image_id].json
  - Aggregate results: base_size_*/bl_newspapers_results.json
  - Summary reports: base_size_*/summary.txt

========================================
EOF

cat "$SUMMARY_FILE"

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="

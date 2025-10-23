#!/bin/bash
#SBATCH --job-name=deepseek_ocr_test
#SBATCH --account=def-jic823
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=%x-%j.out

# DeepSeek-OCR Testing Script for Nibi Cluster
# Processes historical documents with DeepSeek-OCR

set -e

echo "=========================================="
echo "DeepSeek-OCR Testing - Nibi Cluster"
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

# Verify GPU availability
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Configuration
REPO_DIR="${HOME}/projects/def-jic823/deepseek-ocr-test"
PDF_DIR="${HOME}/projects/def-jic823/olmocr/canadiana_pdfs"
RESULTS_DIR="${HOME}/projects/def-jic823/deepseek-ocr-test/results/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR"

echo ""
echo "Configuration:"
echo "  Repository: $REPO_DIR"
echo "  PDF Directory: $PDF_DIR"
echo "  Results Directory: $RESULTS_DIR"
echo ""

# Process PDFs
cd "$REPO_DIR"

# Count available PDFs
PDF_COUNT=$(find "$PDF_DIR" -name "*.pdf" -type f | wc -l)
echo "Found $PDF_COUNT PDF files in $PDF_DIR"
echo ""

# Process each PDF
PROCESSED=0
FAILED=0

for pdf_file in "$PDF_DIR"/*.pdf; do
    if [ -f "$pdf_file" ]; then
        basename=$(basename "$pdf_file")
        echo "=========================================="
        echo "Processing: $basename"
        echo "Progress: $((PROCESSED + 1))/$PDF_COUNT"
        echo "=========================================="

        # Process with both prompts to compare quality
        for prompt in "Free OCR" "Convert document to markdown"; do
            prompt_dir=$(echo "$prompt" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
            output_dir="$RESULTS_DIR/$prompt_dir"
            mkdir -p "$output_dir"

            echo ""
            echo "Running with prompt: '$prompt'"

            if python3 scripts/process_document.py \
                --input "$pdf_file" \
                --output "$output_dir" \
                --prompt "$prompt" \
                --base-size 1024 \
                --dpi 300 \
                --device cuda; then
                echo "✓ Successfully processed with '$prompt'"
            else
                echo "✗ Failed to process with '$prompt'"
                FAILED=$((FAILED + 1))
            fi
        done

        PROCESSED=$((PROCESSED + 1))
        echo ""
    fi
done

# Generate summary report
SUMMARY_FILE="$RESULTS_DIR/processing_summary.txt"

cat > "$SUMMARY_FILE" << EOF
========================================
DeepSeek-OCR Processing Summary
========================================

Job Information:
  Job ID: $SLURM_JOB_ID
  Node: $SLURM_NODELIST
  Start: $(date)

Configuration:
  Model: deepseek-ai/DeepSeek-OCR
  Base Size: 1024
  DPI: 300
  Device: CUDA (H100)

Processing Results:
  Total PDFs: $PDF_COUNT
  Successfully Processed: $PROCESSED
  Failed: $FAILED

Output Location:
  $RESULTS_DIR

Prompts Tested:
  1. Free OCR
  2. Convert document to markdown

========================================
EOF

cat "$SUMMARY_FILE"

# Calculate aggregate statistics
echo ""
echo "=========================================="
echo "Aggregate Statistics"
echo "=========================================="

if [ -f "$RESULTS_DIR/free_ocr/"*.summary.yaml ]; then
    python3 << 'PYEOF'
import yaml
import glob
from pathlib import Path
import sys

results_dir = Path(sys.argv[1])

for prompt_type in ['free_ocr', 'convert_document_to_markdown']:
    prompt_dir = results_dir / prompt_type
    if not prompt_dir.exists():
        continue

    summaries = list(prompt_dir.glob('*.summary.yaml'))
    if not summaries:
        continue

    print(f"\n{prompt_type.replace('_', ' ').title()}:")
    print("-" * 40)

    total_pages = 0
    total_time = 0
    speeds = []

    for summary_file in summaries:
        with open(summary_file) as f:
            data = yaml.safe_load(f)

        total_pages += data['total_pages']
        total_time += data['total_processing_time']
        speeds.append(data['pages_per_second'])

    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    overall_speed = total_pages / total_time if total_time > 0 else 0

    print(f"  Total pages processed: {total_pages}")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Overall speed: {overall_speed:.2f} pages/sec")
    print(f"  Average speed: {avg_speed:.2f} pages/sec")
    print(f"  Documents processed: {len(summaries)}")
PYEOF
fi

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="

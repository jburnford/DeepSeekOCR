# Usage Examples

Practical examples for common DeepSeek-OCR testing scenarios.

## Basic Processing

### Process Single Document

```bash
# Activate environment
source ~/projects/def-jic823/deepseek_venv/bin/activate

# Basic OCR
python scripts/process_document.py \
  --input ~/projects/def-jic823/olmocr/canadiana_pdfs/ColonialOfficeList1896.pdf \
  --output ./results/test1 \
  --prompt "Free OCR"
```

### Markdown Conversion

```bash
python scripts/process_document.py \
  --input document.pdf \
  --output ./results/markdown_test \
  --prompt "Convert document to markdown" \
  --dpi 300
```

### Different Model Sizes

```bash
# Tiny (fastest, lower quality)
python scripts/process_document.py \
  --input document.pdf \
  --output ./results/tiny \
  --base-size 448

# Small
python scripts/process_document.py \
  --input document.pdf \
  --output ./results/small \
  --base-size 672

# Base (balanced)
python scripts/process_document.py \
  --input document.pdf \
  --output ./results/base \
  --base-size 1024

# Large (highest quality, slowest)
python scripts/process_document.py \
  --input document.pdf \
  --output ./results/large \
  --base-size 1536
```

## Batch Processing

### Process All PDFs in Directory

Create a simple batch script:

```bash
#!/bin/bash
#SBATCH --job-name=deepseek_batch
#SBATCH --account=def-jic823
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=%x-%j.out

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.12
source ~/projects/def-jic823/deepseek_venv/bin/activate

PDF_DIR="~/projects/def-jic823/olmocr/canadiana_pdfs"
OUTPUT_DIR="~/projects/def-jic823/deepseek-ocr-test/results/batch_$(date +%Y%m%d)"

mkdir -p "$OUTPUT_DIR"

for pdf in "$PDF_DIR"/*.pdf; do
    echo "Processing: $(basename $pdf)"
    python scripts/process_document.py \
        --input "$pdf" \
        --output "$OUTPUT_DIR" \
        --prompt "Free OCR" \
        --base-size 1024
done
```

Save as `scripts/batch_process.sh` and run:
```bash
sbatch scripts/batch_process.sh
```

### Parallel Processing with Job Arrays

For large collections, use SLURM job arrays:

```bash
#!/bin/bash
#SBATCH --job-name=deepseek_array
#SBATCH --account=def-jic823
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:1
#SBATCH --array=0-9%5  # 10 jobs, max 5 concurrent
#SBATCH --output=array_%A_%a.out

module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.12
source ~/projects/def-jic823/deepseek_venv/bin/activate

# Get list of PDFs
PDF_DIR="~/projects/def-jic823/olmocr/canadiana_pdfs"
PDF_LIST=($(ls $PDF_DIR/*.pdf))

# Process one PDF per array task
PDF_FILE="${PDF_LIST[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="~/projects/def-jic823/deepseek-ocr-test/results/array_job"

python scripts/process_document.py \
    --input "$PDF_FILE" \
    --output "$OUTPUT_DIR" \
    --prompt "Free OCR"
```

## Quality Evaluation

### Compare Two Models

```bash
python scripts/evaluate_ocr_quality.py \
  --compare \
    deepseek=results/doc1.deepseek.json \
    olmocr=results/doc1.olmocr.json \
  --output doc1_comparison.yaml \
  --report doc1_comparison.txt
```

### Evaluate Against Ground Truth

```bash
python scripts/evaluate_ocr_quality.py \
  --input results/doc.deepseek.json \
  --reference ground_truth/doc.txt \
  --output doc_evaluation.yaml
```

### Batch Evaluation

Compare all documents in a directory:

```bash
#!/bin/bash

DEEPSEEK_DIR="results/deepseek_batch"
OLMOCR_DIR="results/olmocr_batch"
EVAL_DIR="evaluation_results"

mkdir -p "$EVAL_DIR"

for deepseek_file in "$DEEPSEEK_DIR"/*.deepseek.json; do
    basename=$(basename "$deepseek_file" .deepseek.json)
    olmocr_file="$OLMOCR_DIR/$basename.olmocr.json"

    if [ -f "$olmocr_file" ]; then
        echo "Comparing: $basename"
        python scripts/evaluate_ocr_quality.py \
            --compare \
                deepseek="$deepseek_file" \
                olmocr="$olmocr_file" \
            --output "$EVAL_DIR/$basename.comparison.yaml" \
            --report "$EVAL_DIR/$basename.comparison.txt"
    fi
done
```

## Interactive Testing

### Interactive Session for Quick Tests

```bash
# Start interactive session
salloc --account=def-jic823 --time=1:00:00 --mem=32G --gpus-per-node=h100:1

# Load environment
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.12
source ~/projects/def-jic823/deepseek_venv/bin/activate

# Quick test
cd ~/projects/def-jic823/deepseek-ocr-test
python scripts/process_document.py \
    --input test.pdf \
    --output ./quick_test \
    --prompt "Free OCR"

# Exit when done
exit
```

### Python Interactive Testing

```python
# In interactive Python session
from pathlib import Path
from scripts.process_document import DeepSeekOCRProcessor

# Initialize
processor = DeepSeekOCRProcessor(
    model_name="deepseek-ai/DeepSeek-OCR",
    device="cuda",
    base_size=1024
)

# Process
results = processor.process_pdf(
    pdf_path=Path("test.pdf"),
    output_dir=Path("./test_output"),
    prompt="Free OCR"
)

# Inspect
print(f"Processed {results['total_pages']} pages")
print(f"Speed: {results['pages_per_second']:.2f} pages/sec")
print(f"First page text: {results['pages'][0]['text'][:200]}")
```

## Performance Testing

### Speed Benchmark

Test different configurations:

```bash
#!/bin/bash

PDF="~/projects/def-jic823/olmocr/canadiana_pdfs/ColonialOfficeList1896.pdf"
BASE_OUTPUT="./benchmark_results"

# Test different sizes
for size in 448 672 1024 1536; do
    echo "Testing base_size=$size"
    python scripts/process_document.py \
        --input "$PDF" \
        --output "$BASE_OUTPUT/size_$size" \
        --base-size $size \
        --prompt "Free OCR"
done

# Compare speeds
echo -e "\nSpeed Comparison:"
for size in 448 672 1024 1536; do
    summary="$BASE_OUTPUT/size_$size/*.summary.yaml"
    if [ -f $summary ]; then
        speed=$(grep "pages_per_second" $summary | awk '{print $2}')
        echo "Size $size: $speed pages/sec"
    fi
done
```

### DPI Impact Test

```bash
#!/bin/bash

PDF="test_document.pdf"

for dpi in 150 200 300 400; do
    echo "Testing DPI=$dpi"
    python scripts/process_document.py \
        --input "$PDF" \
        --output "./dpi_test/dpi_$dpi" \
        --dpi $dpi
done
```

## Historical Document Specific

### Colonial Documents (1600-1900)

```bash
# Optimize for historical text
python scripts/process_document.py \
  --input colonial_record.pdf \
  --output ./colonial_results \
  --prompt "Free OCR" \
  --base-size 1024 \
  --dpi 300  # Higher DPI for degraded text
```

### Census Records

```bash
# Tables and structured data - use markdown
python scripts/process_document.py \
  --input census_1921.pdf \
  --output ./census_results \
  --prompt "Convert document to markdown" \
  --base-size 1024
```

### Handwritten Documents

```bash
# Larger model for complex handwriting
python scripts/process_document.py \
  --input handwritten_letter.pdf \
  --output ./handwriting_results \
  --prompt "Free OCR" \
  --base-size 1536 \
  --dpi 400
```

## Result Analysis

### Extract Specific Information

```python
import json
from pathlib import Path

# Load results
result_file = Path("results/document.deepseek.json")
with open(result_file) as f:
    data = json.load(f)

# Extract page-specific text
page_5_text = data['pages'][4]['text']  # 0-indexed

# Find pages with specific keywords
keyword = "railway"
matching_pages = [
    p for p in data['pages']
    if keyword.lower() in p['text'].lower()
]

print(f"Found '{keyword}' on {len(matching_pages)} pages:")
for page in matching_pages:
    print(f"  Page {page['page_number']}")
```

### Generate Statistics

```bash
# Count total characters across all results
find results/ -name "*.deepseek.json" -exec python -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
    total_chars = sum(len(p['text']) for p in data['pages'])
    print(f\"{sys.argv[1]}: {total_chars:,} characters\")
" {} \;
```

## Troubleshooting Examples

### Test GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Test Model Loading

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True
)
print("Tokenizer loaded successfully!")
```

### Memory Usage Check

```bash
# Monitor GPU during processing
nvidia-smi dmon -s u -d 1 -c 60 > gpu_usage.log &
MONITOR_PID=$!

python scripts/process_document.py --input test.pdf --output ./test

kill $MONITOR_PID
cat gpu_usage.log
```

## Integration Examples

### Integrate with Neo4j Knowledge Graph

```python
from neo4j import GraphDatabase
import json

# Load OCR results
with open('results/census_doc.deepseek.json') as f:
    ocr_data = json.load(f)

# Extract entities and add to graph
driver = GraphDatabase.driver("bolt://localhost:7687",
                             auth=("neo4j", "password"))

with driver.session() as session:
    for page in ocr_data['pages']:
        text = page['text']
        # Extract settlement names, dates, etc.
        # Add to knowledge graph
        pass
```

### Combine with GPT-OSS for RAG

```python
# Use DeepSeek-OCR output for Graph-RAG
ocr_text = extract_text_from_results('document.deepseek.json')

# Query knowledge graph
graph_results = query_neo4j_graph(ocr_text)

# Combine with LLM
llm_response = query_gpt_oss_120b(
    context=graph_results,
    document_text=ocr_text,
    question="What settlements were founded after railway arrival?"
)
```

## Next Steps

- See [SETUP_GUIDE.md](SETUP_GUIDE.md) for installation details
- See [QUICKSTART.md](QUICKSTART.md) for getting started
- Check [README.md](../README.md) for project overview

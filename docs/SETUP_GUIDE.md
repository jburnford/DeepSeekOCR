# DeepSeek-OCR Setup Guide for DRAC/Alliance Clusters

This guide walks through setting up and testing DeepSeek-OCR on Digital Research Alliance of Canada (DRAC) clusters, specifically Nibi with H100 GPUs.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Installation on Nibi](#installation-on-nibi)
4. [Running Tests](#running-tests)
5. [Evaluating Results](#evaluating-results)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Local Machine
- Git access to the repository
- SSH access to Nibi cluster
- GitHub account (for repository access)

### Nibi Cluster Account
- Active account with `def-jic823` allocation
- Access to H100 GPUs
- Sufficient storage in `~/projects/def-jic823/`

## Initial Setup

### 1. Clone Repository Locally

```bash
cd /home/jic823
git clone git@github.com:jburnford/DeepSeekOCR.git deepseek-ocr-test
cd deepseek-ocr-test
```

### 2. Push to GitHub (First Time)

```bash
git add .
git commit -m "Initial DeepSeek-OCR testing setup"
git push -u origin main
```

### 3. Clone on Nibi Cluster

```bash
ssh nibi
cd ~/projects/def-jic823/
git clone git@github.com:jburnford/DeepSeekOCR.git deepseek-ocr-test
```

## Installation on Nibi

### Understanding DRAC/Alliance Environment

DRAC clusters use **environment modules** to manage software. Key principles:

1. **Module System**: Use `module load` to access software
2. **Virtual Environments**: Create isolated Python environments with `virtualenv`
3. **Alliance Wheelhouse**: Pre-built Python packages optimized for clusters
4. **No Direct PyPI Access**: Worker nodes cannot reach PyPI directly

### Installation Steps

#### 1. Submit Installation Job

```bash
cd ~/projects/def-jic823/deepseek-ocr-test
sbatch scripts/install_deepseek.sh
```

This script will:
- Load required modules (Python 3.12, CUDA 12.2, GCC 12.3)
- Create a virtual environment at `~/projects/def-jic823/deepseek_venv`
- Install PyTorch 2.6.0 with CUDA support
- Compile and install flash-attention 2.7.3
- Install transformers, tokenizers, and other dependencies
- Use Alliance wheelhouse when possible (`--no-index` flag)

#### 2. Monitor Installation

```bash
# Check job status
squeue -u $USER

# View installation log (replace JOBID)
tail -f ~/projects/def-jic823/deepseek-ocr-test/install_deepseek-JOBID.out
```

Expected installation time: 30-60 minutes (flash-attention compilation is slow)

#### 3. Verify Installation

Once the job completes, check the output file for:

```
Installation Complete - Verification
========================================
Python version: 3.12.x
PyTorch version: 2.6.0
CUDA available: True
CUDA version: 12.2
GPU: NVIDIA H100 80GB HBM3
Transformers version: 4.46.3
Flash-attention version: 2.7.3
```

### Common Installation Issues

**Problem**: `flash-attn` compilation fails
- **Solution**: Ensure CUDA module is loaded correctly
- Check that job has GPU allocated

**Problem**: PyTorch not finding CUDA
- **Solution**: Verify CUDA module version matches PyTorch requirements
- Try `module spider cuda` to see available versions

**Problem**: `--no-index` fails for some packages
- **Solution**: Script has fallback to PyPI if wheelhouse missing
- Normal for newer packages like `flash-attn`

## Running Tests

### 1. Prepare Test Documents

Place PDFs in the configured directory:

```bash
# Default location (already has test PDFs)
cd ~/projects/def-jic823/olmocr/canadiana_pdfs
ls *.pdf
```

Or copy your own:

```bash
cp /path/to/your/pdfs/*.pdf ~/projects/def-jic823/olmocr/canadiana_pdfs/
```

### 2. Submit Test Job

```bash
cd ~/projects/def-jic823/deepseek-ocr-test
sbatch scripts/nibi_deepseek_test.sh
```

This will:
- Process all PDFs in the input directory
- Test both "Free OCR" and "Convert document to markdown" prompts
- Save results to timestamped directory
- Generate performance statistics

### 3. Monitor Processing

```bash
# Check job status
squeue -u $USER

# Watch progress (replace JOBID)
tail -f deepseek_ocr_test-JOBID.out
```

### 4. Manual Processing (Single Document)

For testing individual documents:

```bash
# Interactive session
salloc --account=def-jic823 --time=1:00:00 --mem=32G --gpus-per-node=h100:1

# Load environment
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.12
source ~/projects/def-jic823/deepseek_venv/bin/activate

# Process document
cd ~/projects/def-jic823/deepseek-ocr-test
python scripts/process_document.py \
  --input ~/projects/def-jic823/olmocr/canadiana_pdfs/ColonialOfficeList1896.pdf \
  --output ./test_results \
  --prompt "Free OCR" \
  --base-size 1024 \
  --dpi 300
```

## Evaluating Results

### Result Files

Each processed document generates:

```
results/YYYYMMDD_HHMMSS/
├── free_ocr/
│   ├── document_name.deepseek.json    # Full results with metadata
│   ├── document_name.deepseek.txt     # Plain text output
│   ├── document_name.summary.yaml     # Performance summary
└── convert_document_to_markdown/
    ├── document_name.deepseek.json
    ├── document_name.deepseek.md      # Markdown formatted
    └── document_name.summary.yaml
```

### Quality Evaluation

#### Compare DeepSeek-OCR with OLMoCR

```bash
python scripts/evaluate_ocr_quality.py \
  --compare \
    deepseek=results/free_ocr/document.deepseek.json \
    olmocr=~/projects/def-jic823/cluster/results/document.olmocr.json \
  --output comparison.yaml \
  --report comparison_report.txt
```

#### View Comparison Report

```bash
cat comparison_report.txt
```

The report includes:
- Processing speed comparison (pages/sec)
- Text quality metrics (character/word statistics)
- Cross-model similarity (CER, WER, F1 scores)
- Error indicators (suspicious words, repeated characters)

### Performance Metrics

Key metrics to evaluate:

1. **Speed**: Pages per second
   - OLMoCR baseline: 1.39 pages/sec (H100)
   - Marker reference: 20-120 pages/sec

2. **Quality**: Text accuracy
   - Character Error Rate (CER): Lower is better
   - Word Error Rate (WER): Lower is better
   - Alphabetic ratio: Should be ~0.80-0.90 for English text

3. **Cost Efficiency**: GPU hours per million pages
   - Calculate: `(1M pages / pages_per_sec) / 3600 hours`

## Troubleshooting

### Job Fails Immediately

**Check**: SLURM output file for error messages

```bash
cat deepseek_ocr_test-JOBID.out
```

**Common causes**:
- Virtual environment not found → Re-run installation
- GPU not available → Check partition and allocation
- Out of memory → Increase `--mem` in SLURM script

### CUDA Out of Memory

**Solution 1**: Reduce model size
```python
# In process_document.py or via CLI
--base-size 672  # Use Small instead of Base
```

**Solution 2**: Lower DPI
```python
--dpi 200  # Instead of 300
```

**Solution 3**: Process fewer pages at once
- Modify script to process documents in batches

### Slow Processing

**Check**: GPU utilization
```bash
# In interactive session or modify SLURM script
nvidia-smi dmon -s u
```

**Potential issues**:
- Not using GPU (check `device=cuda` in config)
- I/O bottleneck (PDF conversion slow)
- Model loading taking too long

### Package Import Errors

**Check**: Virtual environment activated
```bash
which python
# Should show: /home/USER/projects/def-jic823/deepseek_venv/bin/python
```

**Solution**: Reinstall packages
```bash
source ~/projects/def-jic823/deepseek_venv/bin/activate
pip install --no-index --upgrade transformers tokenizers
```

## Best Practices

### 1. Development Workflow

```
Local Machine → Git → GitHub → Nibi Cluster
     ↓                              ↓
   Edit code              Pull & test on cluster
     ↓                              ↓
   Commit                 Report results back
     ↓
   Push
```

### 2. Resource Management

- **Start small**: Test on 1-2 documents first
- **Monitor usage**: Check GPU and memory utilization
- **Clean up**: Remove large result files when done
- **Batch wisely**: Group similar documents together

### 3. Git Workflow

```bash
# On local machine (edit code)
git add scripts/process_document.py
git commit -m "Updated processing parameters"
git push

# On Nibi (update and run)
cd ~/projects/def-jic823/deepseek-ocr-test
git pull
sbatch scripts/nibi_deepseek_test.sh
```

## Next Steps

1. **Optimize parameters**: Test different `base_size` values (448, 672, 1024, 1536)
2. **Compare prompts**: Evaluate "Free OCR" vs "Convert document to markdown"
3. **Scale testing**: Process full document collections
4. **Quality analysis**: Compare with OLMoCR and Marker outputs
5. **Cost analysis**: Calculate GPU hours and cost per million pages

## References

- [DeepSeek-OCR Model Card](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [DRAC Documentation](https://docs.alliancecan.ca/)
- [Alliance Python Packages](https://docs.alliancecan.ca/wiki/Python)
- [SLURM Job Submission](https://docs.alliancecan.ca/wiki/Running_jobs)

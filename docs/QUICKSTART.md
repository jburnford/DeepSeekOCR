# QuickStart Guide

Get started testing DeepSeek-OCR on historical documents in 5 minutes.

## Prerequisites

- SSH access to Nibi cluster
- GitHub repository cloned locally and on Nibi

## Steps

### 1. Clone Repository (Local & Nibi)

**Local:**
```bash
cd /home/jic823
git clone git@github.com:jburnford/DeepSeekOCR.git deepseek-ocr-test
```

**Nibi:**
```bash
ssh nibi
cd ~/projects/def-jic823/
git clone git@github.com:jburnford/DeepSeekOCR.git deepseek-ocr-test
```

### 2. Install DeepSeek-OCR (Nibi)

```bash
cd ~/projects/def-jic823/deepseek-ocr-test
sbatch scripts/install_deepseek.sh
```

⏱️ **Wait 30-60 minutes** for installation (watch with `squeue -u $USER`)

### 3. Run Test Processing

```bash
cd ~/projects/def-jic823/deepseek-ocr-test
sbatch scripts/nibi_deepseek_test.sh
```

### 4. Check Results

```bash
# Find your results directory
ls -lt ~/projects/def-jic823/deepseek-ocr-test/results/

# View summary
cat ~/projects/def-jic823/deepseek-ocr-test/results/YYYYMMDD_HHMMSS/processing_summary.txt

# Check OCR output
less ~/projects/def-jic823/deepseek-ocr-test/results/YYYYMMDD_HHMMSS/free_ocr/*.deepseek.txt
```

### 5. Evaluate Quality

```bash
cd ~/projects/def-jic823/deepseek-ocr-test

python scripts/evaluate_ocr_quality.py \
  --compare \
    deepseek=results/YYYYMMDD_HHMMSS/free_ocr/document.deepseek.json \
    olmocr=~/projects/def-jic823/cluster/results/document.olmocr.json \
  --output comparison.yaml \
  --report comparison_report.txt

cat comparison_report.txt
```

## What's Next?

- **Optimize**: Try different `base_size` values (448, 672, 1024, 1536)
- **Compare**: Test both OCR prompts ("Free OCR" vs markdown)
- **Scale**: Process your full document collection
- **Analyze**: Compare speed and quality with OLMoCR/Marker

## Common Commands

```bash
# Check running jobs
squeue -u $USER

# View job output
tail -f jobname-JOBID.out

# Cancel job
scancel JOBID

# Interactive session
salloc --account=def-jic823 --time=1:00:00 --mem=32G --gpus-per-node=h100:1
```

## Need Help?

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions and troubleshooting.

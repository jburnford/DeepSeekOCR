# Deployment Checklist for Nibi Cluster

Complete guide to deploying DeepSeek-OCR on Nibi cluster from scratch.

## Pre-Deployment Checklist

### Local Machine

- [ ] Git repository cloned locally
- [ ] SSH access to Nibi configured
- [ ] GitHub SSH keys set up
- [ ] Code reviewed and tested

### Nibi Cluster Access

- [ ] Can SSH to Nibi: `ssh nibi`
- [ ] Account has `def-jic823` allocation
- [ ] Adequate storage space in `~/projects/def-jic823/`
- [ ] Can submit SLURM jobs: `squeue -u $USER`

## Deployment Steps

### Step 1: Clone Repository on Nibi

```bash
ssh nibi
cd ~/projects/def-jic823/
git clone git@github.com:jburnford/DeepSeekOCR.git deepseek-ocr-test
cd deepseek-ocr-test
```

**Verification:**
- [ ] Repository cloned successfully
- [ ] All scripts present: `ls scripts/`
- [ ] Configuration files present: `ls config/`

### Step 2: Verify Test Documents Available

```bash
cd ~/projects/def-jic823/olmocr/canadiana_pdfs
ls -lh *.pdf
```

**Verification:**
- [ ] PDF directory exists
- [ ] Test PDFs present (ColonialOfficeList1896.pdf, etc.)
- [ ] Files readable

**If missing PDFs:**
```bash
mkdir -p ~/projects/def-jic823/olmocr/canadiana_pdfs
# Copy PDFs from source location
```

### Step 3: Submit Installation Job

```bash
cd ~/projects/def-jic823/deepseek-ocr-test
sbatch scripts/install_deepseek.sh
```

**Monitor installation:**
```bash
# Check job status
squeue -u $USER

# Watch log (replace JOBID with actual job ID)
tail -f install_deepseek-JOBID.out
```

**Expected duration:** 30-60 minutes

**Verification:**
- [ ] Job completes successfully (check `squeue`)
- [ ] Virtual environment created: `ls ~/projects/def-jic823/deepseek_venv/`
- [ ] Installation log shows successful verification
- [ ] PyTorch with CUDA installed
- [ ] Flash-attention compiled
- [ ] All dependencies installed

**Check installation log for:**
```
Installation Complete - Verification
========================================
PyTorch version: 2.6.0
CUDA available: True
GPU: NVIDIA H100 80GB HBM3
Transformers version: 4.46.3
Flash-attention version: 2.7.3
```

### Step 4: Test Installation (Optional but Recommended)

Run a quick test on a single document:

```bash
# Start interactive session
salloc --account=def-jic823 --time=30:00 --mem=32G --gpus-per-node=h100:1

# Load environment
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.12
source ~/projects/def-jic823/deepseek_venv/bin/activate

# Quick test
cd ~/projects/def-jic823/deepseek-ocr-test
python scripts/process_document.py \
  --input ~/projects/def-jic823/olmocr/canadiana_pdfs/ColonialOfficeList1896.pdf \
  --output ./installation_test \
  --prompt "Free OCR" \
  --base-size 1024 \
  --dpi 300

# Exit
exit
```

**Verification:**
- [ ] Model loads without errors
- [ ] GPU detected and used
- [ ] Document processes successfully
- [ ] Output files generated
- [ ] Processing speed reasonable (>0.5 pages/sec)

### Step 5: Submit Full Test Job

```bash
cd ~/projects/def-jic823/deepseek-ocr-test
sbatch scripts/nibi_deepseek_test.sh
```

**Monitor processing:**
```bash
squeue -u $USER
tail -f deepseek_ocr_test-JOBID.out
```

**Verification:**
- [ ] Job starts successfully
- [ ] PDFs being processed
- [ ] Results directory created
- [ ] Both prompts tested
- [ ] Summary generated

### Step 6: Verify Results

```bash
cd ~/projects/def-jic823/deepseek-ocr-test

# List results
ls -lh results/

# Check latest results
LATEST=$(ls -t results/ | head -1)
echo "Latest results: results/$LATEST"

# View summary
cat "results/$LATEST/processing_summary.txt"

# Check output files
ls "results/$LATEST/free_ocr/"
ls "results/$LATEST/convert_document_to_markdown/"
```

**Verification:**
- [ ] Results directory created with timestamp
- [ ] Processing summary generated
- [ ] Both prompt directories present
- [ ] JSON, TXT, YAML files created
- [ ] Markdown files created (for markdown prompt)

### Step 7: Evaluate Quality

```bash
cd ~/projects/def-jic823/deepseek-ocr-test

# If OLMoCR results available
python scripts/evaluate_ocr_quality.py \
  --compare \
    deepseek=results/YYYYMMDD_HHMMSS/free_ocr/ColonialOfficeList1896.deepseek.json \
    olmocr=~/projects/def-jic823/cluster/results/ColonialOfficeList1896.olmocr.json \
  --output comparison.yaml \
  --report comparison_report.txt

# View report
cat comparison_report.txt
```

**Verification:**
- [ ] Comparison completes successfully
- [ ] YAML results generated
- [ ] Report is readable
- [ ] Metrics look reasonable (CER, WER, speed)

## Post-Deployment Verification

### Performance Check

Expected performance baselines:

| Metric | Target | Status |
|--------|--------|--------|
| Processing Speed | >0.5 pages/sec | [ ] |
| Model Loading | <2 minutes | [ ] |
| GPU Utilization | >50% during processing | [ ] |
| Memory Usage | <60GB for base model | [ ] |

### Quality Check

Review OCR output manually:

```bash
# View first page of OCR output
less results/YYYYMMDD_HHMMSS/free_ocr/*.deepseek.txt
```

**Check for:**
- [ ] Text is readable
- [ ] Layout preserved reasonably
- [ ] No excessive gibberish or errors
- [ ] Historical text handled appropriately
- [ ] Special characters rendered correctly

### Comparison with OLMoCR

If both results available:

| Aspect | DeepSeek-OCR | OLMoCR | Winner |
|--------|--------------|--------|--------|
| Speed (pages/sec) | ___ | 1.39 | [ ] |
| Text accuracy | ___ | ___ | [ ] |
| Layout preservation | ___ | ___ | [ ] |
| Special char handling | ___ | ___ | [ ] |

## Troubleshooting Common Issues

### Installation Fails

**Problem:** flash-attn compilation fails
```bash
# Check CUDA version
module list
# Should show cuda/12.2 or similar

# Check GPU allocation in job
squeue -u $USER --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"
```

**Problem:** PyTorch not finding CUDA
```bash
# Verify CUDA module loaded
module list | grep cuda

# Check torch installation
source ~/projects/def-jic823/deepseek_venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

### Processing Fails

**Problem:** Out of GPU memory
```bash
# Solution: Use smaller model size
python scripts/process_document.py ... --base-size 672
```

**Problem:** Model download fails
```bash
# Check internet connectivity from login node
# Model downloads happen during processing, not installation
```

**Problem:** PDF conversion fails
```bash
# Check pdf2image dependencies
pip list | grep pdf2image
pip list | grep Pillow

# May need system package: poppler-utils
# Contact admin if missing
```

### Slow Performance

**Problem:** Processing much slower than expected

**Check GPU usage:**
```bash
# In separate terminal while job running
ssh nibi
ssh <compute-node>  # Get from squeue
nvidia-smi dmon -s u -d 1
```

**Expected:** GPU utilization >50% during processing

**If low:** Model might not be using GPU properly

## Deployment Success Criteria

### Minimum Requirements (Must Pass)

- [x] Installation completes without errors
- [ ] Virtual environment activated successfully
- [ ] Can import DeepSeek-OCR model
- [ ] GPU detected and used
- [ ] Can process at least one document
- [ ] Output files generated correctly

### Optimal Performance (Should Pass)

- [ ] Processing speed >0.5 pages/sec
- [ ] Model loads in <2 minutes
- [ ] GPU utilization >50%
- [ ] Memory usage reasonable (<60GB)
- [ ] Both prompts work correctly
- [ ] Evaluation scripts run successfully

### Production Ready (Nice to Have)

- [ ] Speed competitive with OLMoCR (>1.0 pages/sec)
- [ ] Quality better than or comparable to OLMoCR
- [ ] Batch processing works reliably
- [ ] Can process full collection without errors
- [ ] Comprehensive comparison reports available

## Next Steps After Deployment

1. **Optimize Parameters**
   - Test different `base_size` values
   - Compare prompt types
   - Adjust DPI for quality/speed tradeoff

2. **Scale Testing**
   - Process full Caribbean collection
   - Process Canadiana.org documents
   - Test on 1600-1700 historical documents

3. **Quality Analysis**
   - Detailed comparison with OLMoCR
   - Comparison with Marker (if available)
   - Manual review of sample outputs

4. **Cost Analysis**
   - Calculate GPU hours per 1M pages
   - Compare with commercial OCR APIs
   - Estimate costs for full 60M page corpus

5. **Integration**
   - Connect to Saskatchewan Knowledge Graph
   - Feed results to GPT-OSS-120B
   - Build Graph-RAG pipeline

## Deployment Log

Document your deployment:

```
Deployment Date: _________________
Deployed By: _____________________
Nibi Account: def-jic823
Installation Job ID: _____________
Test Job ID: _____________________
Results Directory: _______________

Installation Time: _____ minutes
First Test Time: _______ minutes
Processing Speed: ______ pages/sec

Issues Encountered:
-
-

Resolutions:
-
-

Notes:
-
-
```

## Support Resources

- **DRAC Documentation:** https://docs.alliancecan.ca/
- **DeepSeek-OCR:** https://huggingface.co/deepseek-ai/DeepSeek-OCR
- **GitHub Issues:** https://github.com/jburnford/DeepSeekOCR/issues
- **Local Docs:** See `docs/` directory

## Sign-off

- [ ] All checklist items completed
- [ ] Performance meets requirements
- [ ] Quality acceptable for historical documents
- [ ] Documentation reviewed
- [ ] Ready for production testing

**Deployed By:** _________________ **Date:** _________

**Verified By:** _________________ **Date:** _________

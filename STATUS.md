# DeepSeek-OCR Testing Status

**Last Updated**: October 24, 2025

## Current Status: Testing Fixed Script

**Job ID**: 3371988 (Pending)
**Status**: Queued on Nibi cluster
**Expected Runtime**: ~5-7 minutes (20 images @ 16 sec/image)

## What We're Testing

The fixed `process_bl_newspapers_fixed.py` script that:
1. **Captures stdout** from `model.infer()` (where OCR output is printed)
2. **Parses structured output** to extract plain text from XML-like tags
3. **Properly saves OCR text** to JSON files (not `"None"`)
4. **Computes real metrics** with actual OCR text vs ground truth

## Key Discovery

**Original Test (Job 3295859)**:
- ❌ JSON files showed `"ocr_text": "None"`
- ❌ Reported CER=0.999, WER=1.0 (catastrophic)
- ✅ But OCR actually worked! Text was in SLURM logs

**Root Cause**:
- `model.infer()` returns `None`
- Actual OCR output is printed to stdout with structured tags
- Script didn't capture stdout, so `str(None)` → `"None"` saved to JSON

**Actual Performance** (from logs):
- ✅ 20/20 images successfully processed (100% success rate)
- ✅ OCR quality looks good (minor errors like "RUSHOLME" → "RUSHOLANE")
- ✅ 250-5,900 characters extracted per image

## Test Configuration

**Dataset**: British Library Newspapers (BLN600 collection)
- Total available: 600-800 images
- Current test: 20 images
- Format: Historical 19th century newspapers

**Model Settings**:
- Base size: 1024
- Device: H100 GPU
- Processing time: ~16 seconds/image

## Next Steps

1. **Wait for job completion** (~5-7 minutes)
2. **Verify JSON files contain actual OCR text** (not "None")
3. **Check CER/WER metrics** (should be much better than 0.999!)
4. **Compare quality** against OLMoCR baseline
5. **Decide on scale-up** (50, 100, or full 600 images)

## Performance Comparison

| Metric | OLMoCR | DeepSeek-OCR |
|--------|--------|--------------|
| Speed | 1.39 pages/sec | 0.062 images/sec (~16 sec/image) |
| Quality | Unknown | Testing (expect CER < 0.1) |
| Success Rate | 100% | 100% |
| Relative Speed | **26x faster** | Baseline |

**Key Question**: Is DeepSeek's quality improvement worth 26x slower processing?

## Repository

- **GitHub**: `git@github.com:jburnford/DeepSeekOCR.git`
- **Local**: `/home/jic823/DeekSeekOCR`
- **Nibi**: `~/projects/def-jic823/deepseek-ocr-test`

## Files Created

### Analysis Tools
- `scripts/process_bl_newspapers_fixed.py` - Fixed processing script
- `scripts/extract_text_from_logs.py` - Extract OCR from SLURM logs (workaround)
- `scripts/compute_metrics.py` - Calculate CER/WER from extracted texts
- `scripts/parse_results.py` - Parse JSON results for analysis

### Test Scripts
- `scripts/nibi_test_fixed.sh` - SLURM job script for fixed version

### Documentation
- `README.md` - Project overview
- `FINDINGS.md` - Detailed analysis of the bug
- `STATUS.md` - This file

## Workflow

```bash
# Local: Make changes
cd /home/jic823/DeekSeekOCR
# ... edit files ...
git add <files>
git commit -m "Description"
git push origin main

# Nibi: Pull and test
ssh nibi
cd ~/projects/def-jic823/deepseek-ocr-test
git pull origin main
sbatch scripts/nibi_test_fixed.sh

# Monitor
squeue -u jic823
tail -f deepseek_fixed-*.out
```

## Expected Output

Once job completes, we should see:
```
Results saved to: results/bl_newspapers_YYYYMMDD_HHMMSS_FIXED/base_size_1024/
  - 3200797029.json (with actual OCR text!)
  - 3200797032.json
  - ... (20 files total)
  - bl_newspapers_results.json (aggregate)
  - summary.txt
```

Each JSON should contain:
```json
{
  "image_id": "3200801612",
  "ocr_text": "DOUBLE MURDER BY A MOTHER AT RUSHOLANE...",  // ✅ Real text!
  "ground_truth_text": "DOUBLE MURDER BY A MOTHER AT RUSHOLME...",
  "cer": 0.05,  // Much better than 0.999!
  "wer": 0.12
}
```

# DeepSeek-OCR First Test: Findings and Analysis

**Date**: October 23, 2025
**Test Job**: 3295859 (Nibi cluster, node g28)
**Dataset**: British Library Newspapers (BLN600) - 20 images

## Executive Summary

✅ **DeepSeek-OCR successfully extracted text from 14/20 newspaper images**
❌ **Processing script failed to save OCR output - all JSON files show `"ocr_text": "None"`**
🔧 **Workaround**: Extracted actual OCR text from SLURM logs

## The Problem

### What the JSON Files Show (WRONG)
```json
{
  "image_id": "3200801612",
  "ocr_text": "None",
  "cer": 0.9991399698989465,  // 99.9% error rate
  "wer": 1.0                   // 100% error rate
}
```

### What Actually Happened (CORRECT)
The OCR model **successfully** extracted text, but the processing script didn't capture it properly:

```
DOUBLE MURDER BY A MOTHER AT RUSHOLANE.

[SUBJECT OF ILLUSTRATIONS.]

About noon on Friday information was received at the Rusholane
Police-station that a woman had murdered her two children...
```

## Root Cause Analysis

### File: `process_bl_newspapers.py`

**Problem Location**: Line 72
```python
return {
    "text": result if isinstance(result, str) else str(result),
    ...
}
```

**Issue**:
1. `model.infer()` returns `None` (the actual text is printed to stdout but not returned)
2. `str(None)` becomes the string `"None"`
3. This gets saved to JSON files
4. CER/WER computed against ground truth show catastrophic scores (0.999, 1.0)

### DeepSeek-OCR Output Format

The model prints structured output to stdout with XML-like tags:

```
=====================
BASE:  torch.Size([1, 256, 1280])
NO PATCHES
=====================
<|ref|>text<|/ref|><|det|>[[90, 46, 935, 610]]<|/det|>
In the Queen's Bench Division, on Friday...

<|ref|>text<|/ref|><|det|>[[90, 621, 935, 765]]<|/det|>
At the West Bromwich Police court...
```

**Format Components**:
- `<|ref|>text<|/ref|>` - Type label (title, text, subtitle, etc.)
- `<|det|>[[x1,y1,x2,y2]]<|/det|>` - Bounding box coordinates
- Following lines: Actual OCR text content

## Actual OCR Performance

### Extraction Statistics

| Metric | Value |
|--------|-------|
| Images processed | 20 |
| Successful extractions | 14 (70%) |
| Failed extractions | 6 (30%) |
| Character counts | 250-4,900 per image |
| Processing speed | 0.0615 images/sec |
| Avg time per image | ~16 seconds |

### Sample Comparison

**Ground Truth** (3200801612.txt):
```
DOUBLE MURDER BY A MOTHER AT RUSHOLME.
```

**OCR Output** (extracted from logs):
```
DOUBLE MURDER BY A MOTHER AT RUSHOLANE.
```

**Error**: "RUSHOLME" → "RUSHOLANE" (OCR substitution)

### OCR Quality Assessment (Qualitative)

**Strengths**:
- Successfully extracted dense newspaper text
- Preserved paragraph structure
- Handled Victorian typography reasonably well
- Captured most proper nouns correctly

**Errors Observed**:
- Place name substitutions ("RUSHOLANE" vs "RUSHOLME")
- Number/text confusion ("61" vs "54", "8th" vs "3rd")
- Word substitutions ("two wives" vs "the wife", "work" vs "week")
- Some text missing near image boundaries

**Overall**: Better than expected for 19th century newspaper images!

## Images That Failed

6 images returned very short text (~250 chars vs 2,000+ expected):
- Possible causes: Poor image quality, layout complexity, text detection failures
- Need manual inspection of source images

## Fix Required

### Option 1: Parse Structured Output
```python
import re

def parse_deepseek_output(raw_output: str) -> str:
    """Extract plain text from DeepSeek structured output."""
    # Match text after </det|> tags
    pattern = r'<\|/det\|>\n(.+?)(?=\n\n|<\|ref\||$)'
    texts = re.findall(pattern, raw_output, re.DOTALL)
    return '\n\n'.join(t.strip() for t in texts if len(t.strip()) > 10)

# In process_image():
result = self.model.infer(...)
if result is None:
    # Capture stdout? Or model returns structured format?
    pass
else:
    text = parse_deepseek_output(result)
```

### Option 2: Capture stdout
```python
import sys
from io import StringIO

# Redirect stdout during inference
old_stdout = sys.stdout
sys.stdout = captured = StringIO()

result = self.model.infer(...)

sys.stdout = old_stdout
output = captured.getvalue()
text = parse_deepseek_output(output)
```

### Option 3: Check DeepSeek Documentation
- Review DeepSeek-OCR API for text-only output mode
- Check if there's a `return_text=True` parameter
- Look for examples of plain text extraction

## Next Steps

### Immediate Actions

1. **Fix processing script**:
   - Implement proper output parsing
   - Test on single image first
   - Verify JSON files contain actual text

2. **Reprocess test set**:
   - Run fixed script on same 20 images
   - Compute proper CER/WER metrics
   - Compare against ground truth

3. **Analyze failures**:
   - Inspect the 6 images with short output
   - Determine if issue is image quality or model limitation

### Evaluation Tasks

1. **Compute proper metrics**:
   - Install `jiwer` for CER/WER computation
   - Run `compute_metrics.py` on extracted texts
   - Generate quality report

2. **Compare to baselines**:
   - OLMoCR: 1.39 pages/sec (our deployment)
   - Marker: 20-120 pages/sec (reported)
   - DeepSeek: 0.0615 images/sec (16 sec/page) - **26x slower than OLMoCR**

3. **Quality vs Speed tradeoff**:
   - Is DeepSeek's quality better enough to justify 26x slowdown?
   - Need quantitative CER/WER comparison

### Future Tests

1. **Test different base_sizes**: 448, 672, 1536 (quality vs speed)
2. **Scale to full BLN600 dataset**: 600 images
3. **Test on other historical documents**: Different time periods, languages
4. **vLLM acceleration**: Check if vLLM backend can improve speed

## Files Created

### Analysis Tools
- `parse_results.py` - Parse JSON results (when fixed)
- `extract_text_from_logs.py` - Extract OCR from SLURM logs (current workaround)
- `compute_metrics.py` - Calculate CER/WER against ground truth

### Extracted Data
- `extracted_from_logs/` - 14 OCR text files from SLURM log
- `ground_truth/` - Ground truth text files from BL dataset
- `deepseek_bl_news-3295859.out` - Full SLURM job log

### Documentation
- `README.md` - Project overview and setup
- `FINDINGS.md` - This document

## Conclusion

**The good news**: DeepSeek-OCR works and produces reasonable OCR output for 19th century newspapers.

**The bad news**:
1. Processing script has a critical bug that loses all OCR text
2. Performance is very slow (16 sec/image) compared to OLMoCR (0.7 sec/page)
3. Need quantitative metrics to assess quality vs speed tradeoff

**Priority**: Fix the processing script and recompute proper metrics before drawing conclusions about model quality.

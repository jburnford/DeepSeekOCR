# Full Dataset Processing - 600 Images

**Job ID**: 3376583
**Status**: Pending (queued)
**Submitted**: October 24, 2025

## Configuration

- **Dataset**: British Library Newspapers (BLN600)
- **Images**: 600 total
- **Model**: DeepSeek-OCR (base_size=1024)
- **Hardware**: Nibi H100 GPU
- **Walltime**: 4 hours

## Time Estimates

Based on 20-image test performance:
- **Average processing time**: ~14.4 seconds/image
- **Estimated total time**: 600 × 14.4 sec = **8,640 seconds (~2.4 hours)**
- **With overhead**: ~2.5-3 hours total

Should complete well within the 4-hour limit.

## Baseline Comparison

### OLMoCR (Baseline)
- **Word Error Rate**: ~5% (across full corpus)
- **Processing speed**: 1.39 pages/sec

### DeepSeek-OCR (20-image test)
- **Excellent images (70%)**: CER 2.28%, WER 5.59%
- **All successful (85%)**: CER 12.22%, WER 15.21%
- **Processing speed**: 0.0693 images/sec (~14.4 sec/image)
- **Speed ratio**: **20x slower** than OLMoCR

## Research Questions

1. **Quality at Scale**: Does the 14/20 (70%) excellent quality rate hold across 600 images?
2. **WER Comparison**: Can DeepSeek achieve ≤5% WER like OLMoCR?
3. **Failure Rate**: How many images fail (CER ≥ 0.9) in full corpus?
4. **Error Patterns**: What types of documents cause poor performance?

## Expected Results

**If DeepSeek matches our 20-image sample**:
- ~420 images with excellent quality (CER < 0.1)
- ~90 images with poor quality (CER 0.1-0.9)
- ~90 images failed (CER ≥ 0.9)
- Overall WER: 12-15%

**Success criteria**:
- ✅ WER ≤ 5%: Matches OLMoCR, worth considering despite 20x slowdown
- ⚠️ WER 5-10%: Competitive but slower, need quality analysis
- ❌ WER > 10%: OLMoCR is clearly better choice

## Monitoring

```bash
# Check job status
ssh nibi "squeue -u jic823"

# Monitor progress (once running)
ssh nibi "tail -f ~/projects/def-jic823/deepseek-ocr-test/deepseek_fixed-3376583.out"

# Check completion
ssh nibi "grep 'Processing Complete' ~/projects/def-jic823/deepseek-ocr-test/deepseek_fixed-3376583.out"
```

## Output Location

Results will be saved to:
```
~/projects/def-jic823/deepseek-ocr-test/results/bl_newspapers_YYYYMMDD_HHMMSS_FIXED/base_size_1024/
```

Files:
- 600 individual JSON files (one per image)
- `bl_newspapers_results.json` (aggregate results)
- `summary.txt` (statistics)

## Post-Processing Analysis

Once complete, we'll:
1. **Copy results locally** for analysis
2. **Generate quality distribution** (CER/WER histograms)
3. **Identify failure cases** and patterns
4. **Compare to OLMoCR baseline**
5. **Make recommendation** on model selection

## Notes

- 20-image test showed bimodal distribution: 70% excellent, 15% poor, 15% failed
- Average metrics are misleading due to failed images pulling down the mean
- Need to analyze by image category/difficulty rather than simple averages
- OLMoCR's 5% WER is the target to beat

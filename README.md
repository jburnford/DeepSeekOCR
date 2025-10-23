# DeepSeek-OCR Historical Document Testing

Testing DeepSeek-OCR's performance on historical documents (1600-1900) using DRAC clusters.

## Project Overview

This project evaluates DeepSeek-OCR against existing OCR solutions (OLMoCR, Marker) for historical document processing. DeepSeek-OCR is a 3B parameter model designed to expand LLM context windows, with strong OCR capabilities.

## Repository Structure

```
deepseek-ocr-test/
├── scripts/              # SLURM and processing scripts
├── config/               # Configuration files
├── test_documents/       # Sample historical PDFs
├── results/              # OCR output and quality metrics
└── docs/                 # Documentation and reports
```

## Model Specifications

- **Parameters**: 3B (BF16 precision)
- **Framework**: HuggingFace Transformers
- **GPU Required**: NVIDIA (CUDA 11.8+)
- **Inference Options**: Standard + vLLM accelerated
- **Size Variants**: Tiny, Small, Base, Large, Gundam

## Target Hardware

- **Cluster**: Nibi (H100 GPUs)
- **Fallback**: Cedar (if compatible)

## Comparison Baseline

| Model | Status | Performance (pages/sec) |
|-------|--------|-------------------------|
| OLMoCR | Deployed | 1.39 (single doc) |
| Marker | Reference | 20-120 |
| DeepSeek-OCR | Testing | TBD |

## Installation

See `scripts/install_deepseek.sh` for Nibi cluster setup.

## Usage

```bash
# Submit test job
sbatch scripts/nibi_deepseek_test.sh

# Process single document
python scripts/process_document.py --input test.pdf --output results/
```

## Quality Evaluation

We evaluate on:
1. **Text Accuracy**: Character error rate (CER), word error rate (WER)
2. **Layout Preservation**: Table/column detection
3. **Historical Text Handling**: Archaic spelling, ligatures, damaged text
4. **Processing Speed**: Pages per second
5. **Cost Efficiency**: GPU hours per 1M pages

## Test Documents

Historical documents from 1600-1900:
- Colonial administrative records (1896-1898)
- Caribbean historical documents
- Census records
- Archival materials from Canadiana.org

## License

MIT (following DeepSeek-OCR license)

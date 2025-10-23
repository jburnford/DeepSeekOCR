#!/usr/bin/env python3
"""
Process British Library Newspaper Images with DeepSeek-OCR
Evaluate against ground truth data from Gale's collection.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import tempfile
import os

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import yaml


class BLNewspaperProcessor:
    """Process British Library newspaper images with DeepSeek-OCR."""

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR",
                 device: str = "cuda",
                 base_size: int = 1024):
        print(f"Loading DeepSeek-OCR model...")
        self.device = device
        self.base_size = base_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True
        )
        self.model = self.model.eval().to(device).to(torch.bfloat16)

        print(f"Model loaded on {device}!")

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image using DeepSeek-OCR's infer() method."""
        start_time = time.time()

        # Use DeepSeek-OCR's built-in infer() method with proper prompt format
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "

        # Create temporary output directory for this image
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Call model.infer() as documented
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=tmp_dir,
                base_size=self.base_size,
                image_size=self.base_size,
                crop_mode=False,
                save_results=False,
                test_compress=False
            )

        elapsed = time.time() - start_time

        return {
            "text": result if isinstance(result, str) else str(result),
            "processing_time": elapsed,
            "image_path": str(image_path)
        }


def load_ground_truth(gt_path: Path) -> str:
    """Load ground truth text."""
    with open(gt_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def compute_cer_wer(predicted: str, reference: str) -> Dict[str, float]:
    """Compute Character Error Rate and Word Error Rate."""
    try:
        import jiwer

        # Normalize whitespace
        pred_norm = ' '.join(predicted.split())
        ref_norm = ' '.join(reference.split())

        cer = jiwer.cer(ref_norm, pred_norm)
        wer = jiwer.wer(ref_norm, pred_norm)

        return {"cer": cer, "wer": wer}
    except ImportError:
        return {"cer": -1, "wer": -1, "error": "jiwer not installed"}


def main():
    parser = argparse.ArgumentParser(
        description="Process British Library newspapers with ground truth evaluation"
    )

    parser.add_argument("--images", type=Path, required=True,
                       help="Directory with newspaper images")
    parser.add_argument("--ground-truth", type=Path, required=True,
                       help="Directory with ground truth text files")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory for results")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR",
                       help="Model name")
    parser.add_argument("--base-size", type=int, default=1024,
                       help="Base size (448/672/1024/1536)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of images to process")
    parser.add_argument("--device", default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Validate directories
    if not args.images.exists():
        print(f"Error: Images directory not found: {args.images}")
        return 1

    if not args.ground_truth.exists():
        print(f"Error: Ground truth directory not found: {args.ground_truth}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    # Get list of images
    image_extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}
    images = sorted([
        f for f in args.images.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if args.limit:
        images = images[:args.limit]

    print(f"\n{'='*70}")
    print(f"British Library Newspaper OCR - DeepSeek-OCR")
    print(f"{'='*70}")
    print(f"Images directory: {args.images}")
    print(f"Ground truth directory: {args.ground_truth}")
    print(f"Output directory: {args.output}")
    print(f"Total images: {len(images)}")
    print(f"Base size: {args.base_size}")
    print(f"{'='*70}\n")

    # Initialize processor
    processor = BLNewspaperProcessor(
        model_name=args.model,
        device=args.device,
        base_size=args.base_size
    )

    # Process images
    results = {
        "metadata": {
            "dataset": "British Library Newspapers - BLN600",
            "model": args.model,
            "base_size": args.base_size,
            "total_images": len(images),
            "processing_start": datetime.now().isoformat()
        },
        "images": []
    }

    total_cer = 0
    total_wer = 0
    valid_comparisons = 0
    total_time = 0

    for idx, image_path in enumerate(images, 1):
        print(f"\n[{idx}/{len(images)}] Processing: {image_path.name}")

        # Find corresponding ground truth
        gt_path = args.ground_truth / f"{image_path.stem}.txt"

        if not gt_path.exists():
            print(f"  ⚠️  Ground truth not found: {gt_path.name}")
            continue

        # Process image
        try:
            ocr_result = processor.process_image(image_path)
            total_time += ocr_result["processing_time"]

            # Load ground truth
            ground_truth = load_ground_truth(gt_path)

            # Compute metrics
            metrics = compute_cer_wer(ocr_result["text"], ground_truth)

            # Combine results
            image_result = {
                "image_id": image_path.stem,
                "image_file": image_path.name,
                "ocr_text": ocr_result["text"],
                "ground_truth_text": ground_truth,
                "processing_time": ocr_result["processing_time"],
                "cer": metrics.get("cer", -1),
                "wer": metrics.get("wer", -1)
            }

            results["images"].append(image_result)

            # Update totals
            if metrics.get("cer", -1) >= 0:
                total_cer += metrics["cer"]
                total_wer += metrics["wer"]
                valid_comparisons += 1

            # Print metrics
            print(f"  Time: {ocr_result['processing_time']:.2f}s")
            print(f"  CER: {metrics.get('cer', -1):.4f}")
            print(f"  WER: {metrics.get('wer', -1):.4f}")
            print(f"  OCR length: {len(ocr_result['text'])} chars")
            print(f"  GT length: {len(ground_truth)} chars")

            # Save individual result
            result_file = args.output / f"{image_path.stem}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(image_result, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  ❌ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute aggregate statistics
    results["metadata"]["processing_end"] = datetime.now().isoformat()
    results["metadata"]["total_processing_time"] = total_time
    results["metadata"]["images_processed"] = len(results["images"])
    results["metadata"]["valid_comparisons"] = valid_comparisons

    if valid_comparisons > 0:
        results["metadata"]["average_cer"] = total_cer / valid_comparisons
        results["metadata"]["average_wer"] = total_wer / valid_comparisons
        results["metadata"]["images_per_second"] = len(results["images"]) / total_time if total_time > 0 else 0

    # Save full results
    results_file = args.output / "bl_newspapers_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate summary report
    summary_file = args.output / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("British Library Newspapers - DeepSeek-OCR Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: BLN600\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Base Size: {args.base_size}\n\n")
        f.write(f"Images Processed: {len(results['images'])}\n")
        f.write(f"Valid Comparisons: {valid_comparisons}\n")
        f.write(f"Total Processing Time: {total_time:.2f}s\n")
        if valid_comparisons > 0:
            f.write(f"Average Speed: {len(results['images']) / total_time:.4f} images/sec\n\n")
            f.write(f"Quality Metrics:\n")
            f.write(f"  Average CER: {results['metadata']['average_cer']:.4f}\n")
            f.write(f"  Average WER: {results['metadata']['average_wer']:.4f}\n")

    # Print summary
    print(f"\n{'='*70}")
    print("Processing Complete!")
    print(f"{'='*70}")
    print(f"Images processed: {len(results['images'])}")
    print(f"Valid comparisons: {valid_comparisons}")
    print(f"Total time: {total_time:.2f}s")
    if valid_comparisons > 0:
        print(f"Average speed: {len(results['images']) / total_time:.4f} images/sec")
        print(f"Average CER: {results['metadata']['average_cer']:.4f}")
        print(f"Average WER: {results['metadata']['average_wer']:.4f}")
    print(f"\nResults saved to: {args.output}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    exit(main())

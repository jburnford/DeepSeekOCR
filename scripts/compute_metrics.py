#!/usr/bin/env python3
"""
Compute CER/WER metrics from extracted OCR texts against ground truth.

This script properly evaluates the OCR quality after extracting text from logs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


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
        print("\nError: jiwer not installed. Install with:")
        print("  python3 -m pip install jiwer")
        return {"cer": -1, "wer": -1, "error": "jiwer not installed"}


def load_ground_truth(gt_dir: Path, image_id: str) -> str:
    """Load ground truth text for an image."""
    gt_path = gt_dir / f"{image_id}.txt"
    if not gt_path.exists():
        return None

    with open(gt_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(
        description="Compute CER/WER from extracted OCR texts"
    )
    parser.add_argument(
        "ocr_dir",
        type=Path,
        help="Directory with extracted OCR texts (*_ocr_from_log.txt)"
    )
    parser.add_argument(
        "ground_truth_dir",
        type=Path,
        help="Directory with ground truth text files (*.txt)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results (default: metrics.json in ocr_dir)"
    )

    args = parser.parse_args()

    if not args.ocr_dir.exists():
        print(f"Error: OCR directory not found: {args.ocr_dir}")
        return 1

    if not args.ground_truth_dir.exists():
        print(f"Error: Ground truth directory not found: {args.ground_truth_dir}")
        return 1

    output_file = args.output or (args.ocr_dir / "metrics.json")

    # Get all OCR files
    ocr_files = sorted(args.ocr_dir.glob("*_ocr_from_log.txt"))

    if not ocr_files:
        print(f"No OCR files found in {args.ocr_dir}")
        return 1

    print(f"Found {len(ocr_files)} OCR files")
    print(f"Ground truth directory: {args.ground_truth_dir}\n")

    results = []
    total_cer = 0
    total_wer = 0
    valid_comparisons = 0

    for ocr_file in ocr_files:
        # Extract image_id from filename
        image_id = ocr_file.stem.replace("_ocr_from_log", "")

        # Load OCR text
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_text = f.read()

        # Load ground truth
        gt_text = load_ground_truth(args.ground_truth_dir, image_id)

        if gt_text is None:
            print(f"⚠️  {image_id}: No ground truth found")
            continue

        # Compute metrics
        metrics = compute_cer_wer(ocr_text, gt_text)

        if metrics.get("error"):
            print(f"❌ Error computing metrics: {metrics['error']}")
            return 1

        # Store result
        result = {
            "image_id": image_id,
            "ocr_length": len(ocr_text),
            "gt_length": len(gt_text),
            "cer": metrics["cer"],
            "wer": metrics["wer"]
        }
        results.append(result)

        # Update totals
        total_cer += metrics["cer"]
        total_wer += metrics["wer"]
        valid_comparisons += 1

        # Print progress
        print(f"{image_id}:")
        print(f"  OCR: {len(ocr_text):,} chars")
        print(f"  GT:  {len(gt_text):,} chars")
        print(f"  CER: {metrics['cer']:.4f}")
        print(f"  WER: {metrics['wer']:.4f}")

    # Compute averages
    summary = {
        "total_images": len(results),
        "valid_comparisons": valid_comparisons,
        "average_cer": total_cer / valid_comparisons if valid_comparisons > 0 else 0,
        "average_wer": total_wer / valid_comparisons if valid_comparisons > 0 else 0,
        "results": results
    }

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images: {len(results)}")
    print(f"Valid comparisons: {valid_comparisons}")
    if valid_comparisons > 0:
        print(f"\nAverage CER: {summary['average_cer']:.4f} ({summary['average_cer']*100:.2f}%)")
        print(f"Average WER: {summary['average_wer']:.4f} ({summary['average_wer']*100:.2f}%)")
    print(f"\nResults saved to: {output_file}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())

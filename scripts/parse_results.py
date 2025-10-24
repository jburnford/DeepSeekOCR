#!/usr/bin/env python3
"""
Parse DeepSeek OCR results and extract plain text for evaluation.

Usage:
    python3 parse_results.py <results_dir> [--output-dir <dir>]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def parse_individual_result(json_path: Path) -> Dict:
    """Parse a single result JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {
        'image_id': data.get('image_id'),
        'ocr_text': data.get('ocr_text'),
        'ground_truth': data.get('ground_truth_text'),
        'cer': data.get('cer'),
        'wer': data.get('wer'),
        'processing_time': data.get('processing_time')
    }


def parse_aggregate_result(json_path: Path) -> Dict:
    """Parse the aggregate results JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def extract_ocr_texts(results_dir: Path, base_size: str = None) -> List[Dict]:
    """
    Extract OCR texts from all individual result files.

    Args:
        results_dir: Path to results directory (e.g., bl_newspapers_20251023_180318)
        base_size: Specific base_size directory to process (e.g., "1024")
                   If None, processes all base_size_* directories

    Returns:
        List of result dictionaries
    """
    results = []

    if base_size:
        base_dirs = [results_dir / f"base_size_{base_size}"]
    else:
        base_dirs = sorted(results_dir.glob("base_size_*"))

    for base_dir in base_dirs:
        if not base_dir.is_dir():
            print(f"Warning: {base_dir} not found, skipping")
            continue

        print(f"\nProcessing {base_dir.name}...")

        # Get all individual JSON files (exclude aggregate)
        json_files = sorted([
            f for f in base_dir.glob("*.json")
            if not f.name.endswith("_results.json")
        ])

        print(f"  Found {len(json_files)} result files")

        for json_file in json_files:
            try:
                result = parse_individual_result(json_file)
                result['base_size'] = base_dir.name.replace("base_size_", "")
                results.append(result)
            except Exception as e:
                print(f"  Error parsing {json_file.name}: {e}")

    return results


def save_extracted_texts(results: List[Dict], output_dir: Path):
    """
    Save extracted OCR texts and ground truth to separate files.

    Creates:
        - <image_id>_ocr.txt: OCR output
        - <image_id>_gt.txt: Ground truth
        - <image_id>_metrics.json: Metrics only
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        image_id = result['image_id']
        base_size = result['base_size']

        # Create subdirectory for each base_size
        size_dir = output_dir / f"base_size_{base_size}"
        size_dir.mkdir(exist_ok=True)

        # Save OCR text
        ocr_file = size_dir / f"{image_id}_ocr.txt"
        with open(ocr_file, 'w', encoding='utf-8') as f:
            f.write(result['ocr_text'] if result['ocr_text'] else "")

        # Save ground truth
        gt_file = size_dir / f"{image_id}_gt.txt"
        with open(gt_file, 'w', encoding='utf-8') as f:
            f.write(result['ground_truth'] if result['ground_truth'] else "")

        # Save metrics
        metrics_file = size_dir / f"{image_id}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'image_id': image_id,
                'cer': result['cer'],
                'wer': result['wer'],
                'processing_time': result['processing_time']
            }, f, indent=2)


def generate_summary(results: List[Dict]) -> Dict:
    """Generate summary statistics across all results."""
    by_base_size = {}

    for result in results:
        base_size = result['base_size']
        if base_size not in by_base_size:
            by_base_size[base_size] = {
                'count': 0,
                'cer_scores': [],
                'wer_scores': [],
                'processing_times': [],
                'failed_ocr': 0  # ocr_text is "None" or empty
            }

        stats = by_base_size[base_size]
        stats['count'] += 1

        if result['cer'] is not None:
            stats['cer_scores'].append(result['cer'])
        if result['wer'] is not None:
            stats['wer_scores'].append(result['wer'])
        if result['processing_time'] is not None:
            stats['processing_times'].append(result['processing_time'])

        # Check for OCR failures
        if result['ocr_text'] in [None, "None", ""]:
            stats['failed_ocr'] += 1

    # Calculate averages
    summary = {}
    for base_size, stats in by_base_size.items():
        summary[base_size] = {
            'total_images': stats['count'],
            'failed_ocr': stats['failed_ocr'],
            'success_rate': (stats['count'] - stats['failed_ocr']) / stats['count'] * 100,
            'avg_cer': sum(stats['cer_scores']) / len(stats['cer_scores']) if stats['cer_scores'] else None,
            'avg_wer': sum(stats['wer_scores']) / len(stats['wer_scores']) if stats['wer_scores'] else None,
            'avg_processing_time': sum(stats['processing_times']) / len(stats['processing_times']) if stats['processing_times'] else None
        }

    return summary


def print_summary(summary: Dict):
    """Print summary statistics to console."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for base_size, stats in sorted(summary.items()):
        print(f"\nBase Size: {base_size}")
        print(f"  Total Images: {stats['total_images']}")
        print(f"  Failed OCR: {stats['failed_ocr']}")
        print(f"  Success Rate: {stats['success_rate']:.2f}%")
        print(f"  Average CER: {stats['avg_cer']:.4f}" if stats['avg_cer'] else "  Average CER: N/A")
        print(f"  Average WER: {stats['avg_wer']:.4f}" if stats['avg_wer'] else "  Average WER: N/A")
        print(f"  Avg Processing Time: {stats['avg_processing_time']:.2f}s" if stats['avg_processing_time'] else "  Avg Processing Time: N/A")


def main():
    parser = argparse.ArgumentParser(
        description="Parse DeepSeek OCR results and extract plain text"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to results directory (e.g., bl_newspapers_20251023_180318)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for extracted texts (default: <results_dir>/extracted)"
    )
    parser.add_argument(
        "--base-size",
        type=str,
        default=None,
        help="Specific base_size to process (e.g., '1024'). If not specified, processes all."
    )

    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    output_dir = args.output_dir or (results_dir / "extracted")

    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Extract all results
    results = extract_ocr_texts(results_dir, args.base_size)

    if not results:
        print("\nNo results found!")
        return 1

    print(f"\nTotal results extracted: {len(results)}")

    # Save extracted texts
    print(f"\nSaving extracted texts to {output_dir}...")
    save_extracted_texts(results, output_dir)

    # Generate and print summary
    summary = generate_summary(results)
    print_summary(summary)

    # Save summary to JSON
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Analyze full 600-image DeepSeek-OCR results and compare to OLMoCR baseline.

Usage:
    python3 analyze_full_results.py <results_dir>
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def load_results(results_dir: Path) -> List[Dict]:
    """Load all individual result JSON files."""
    results = []

    json_files = sorted([
        f for f in results_dir.glob("*.json")
        if not f.name.endswith("_results.json")
    ])

    print(f"Loading {len(json_files)} result files...")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")

    return results


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze results and categorize by quality."""

    excellent = []  # CER < 0.1
    good = []       # 0.1 <= CER < 0.2
    fair = []       # 0.2 <= CER < 0.5
    poor = []       # 0.5 <= CER < 0.9
    failed = []     # CER >= 0.9

    total_time = 0

    for result in results:
        cer = result.get('cer', 0)
        wer = result.get('wer', 0)
        time = result.get('processing_time', 0)
        total_time += time

        if cer < 0.1:
            excellent.append(result)
        elif cer < 0.2:
            good.append(result)
        elif cer < 0.5:
            fair.append(result)
        elif cer < 0.9:
            poor.append(result)
        else:
            failed.append(result)

    return {
        'total': len(results),
        'excellent': excellent,
        'good': good,
        'fair': fair,
        'poor': poor,
        'failed': failed,
        'total_time': total_time
    }


def compute_stats(results: List[Dict]) -> Dict:
    """Compute average CER/WER for a list of results."""
    if not results:
        return {'avg_cer': 0, 'avg_wer': 0, 'count': 0}

    total_cer = sum(r.get('cer', 0) for r in results)
    total_wer = sum(r.get('wer', 0) for r in results)

    return {
        'avg_cer': total_cer / len(results),
        'avg_wer': total_wer / len(results),
        'count': len(results)
    }


def print_report(analysis: Dict, olmocr_wer: float = 0.05):
    """Print analysis report."""

    print("\n" + "="*80)
    print("DeepSeek-OCR Full Dataset Analysis")
    print("="*80)

    total = analysis['total']

    print(f"\nTotal Images: {total}")
    print(f"Total Processing Time: {analysis['total_time']:.1f}s ({analysis['total_time']/3600:.2f} hours)")
    print(f"Average Speed: {total/analysis['total_time']:.4f} images/sec")
    print(f"Average Time per Image: {analysis['total_time']/total:.2f}s")

    print("\n" + "-"*80)
    print("Quality Distribution")
    print("-"*80)

    categories = [
        ('Excellent (CER < 0.1)', analysis['excellent']),
        ('Good (0.1 ≤ CER < 0.2)', analysis['good']),
        ('Fair (0.2 ≤ CER < 0.5)', analysis['fair']),
        ('Poor (0.5 ≤ CER < 0.9)', analysis['poor']),
        ('Failed (CER ≥ 0.9)', analysis['failed'])
    ]

    for label, results in categories:
        count = len(results)
        pct = (count / total * 100) if total > 0 else 0
        stats = compute_stats(results)

        print(f"\n{label}: {count} images ({pct:.1f}%)")
        if count > 0:
            print(f"  Average CER: {stats['avg_cer']:.4f} ({stats['avg_cer']*100:.2f}%)")
            print(f"  Average WER: {stats['avg_wer']:.4f} ({stats['avg_wer']*100:.2f}%)")

    print("\n" + "-"*80)
    print("Overall Statistics")
    print("-"*80)

    # All images
    all_stats = compute_stats(analysis['excellent'] + analysis['good'] +
                               analysis['fair'] + analysis['poor'] + analysis['failed'])
    print(f"\nAll Images:")
    print(f"  Average CER: {all_stats['avg_cer']:.4f} ({all_stats['avg_cer']*100:.2f}%)")
    print(f"  Average WER: {all_stats['avg_wer']:.4f} ({all_stats['avg_wer']*100:.2f}%)")

    # Excluding failures
    successful = analysis['excellent'] + analysis['good'] + analysis['fair'] + analysis['poor']
    success_stats = compute_stats(successful)
    print(f"\nExcluding Failed (CER < 0.9): {len(successful)} images")
    print(f"  Average CER: {success_stats['avg_cer']:.4f} ({success_stats['avg_cer']*100:.2f}%)")
    print(f"  Average WER: {success_stats['avg_wer']:.4f} ({success_stats['avg_wer']*100:.2f}%)")

    # High quality only
    high_quality = analysis['excellent'] + analysis['good']
    hq_stats = compute_stats(high_quality)
    print(f"\nHigh Quality Only (CER < 0.2): {len(high_quality)} images ({len(high_quality)/total*100:.1f}%)")
    print(f"  Average CER: {hq_stats['avg_cer']:.4f} ({hq_stats['avg_cer']*100:.2f}%)")
    print(f"  Average WER: {hq_stats['avg_wer']:.4f} ({hq_stats['avg_wer']*100:.2f}%)")

    print("\n" + "-"*80)
    print("Comparison to OLMoCR Baseline")
    print("-"*80)

    print(f"\nOLMoCR Baseline WER: {olmocr_wer*100:.1f}%")
    print(f"DeepSeek Overall WER: {all_stats['avg_wer']*100:.2f}%")

    wer_diff = all_stats['avg_wer'] - olmocr_wer
    if wer_diff < 0:
        print(f"✅ DeepSeek is BETTER by {abs(wer_diff)*100:.2f} percentage points")
    elif wer_diff < 0.05:
        print(f"⚠️  DeepSeek is comparable (within 5 percentage points)")
    else:
        print(f"❌ DeepSeek is WORSE by {wer_diff*100:.2f} percentage points")

    print(f"\nSuccess Rate (CER < 0.9): {len(successful)}/{total} = {len(successful)/total*100:.1f}%")

    # List failed images
    if analysis['failed']:
        print("\n" + "-"*80)
        print(f"Failed Images ({len(analysis['failed'])} total)")
        print("-"*80)
        for result in sorted(analysis['failed'], key=lambda x: x.get('cer', 0), reverse=True)[:10]:
            print(f"  {result['image_id']}: CER={result.get('cer', 0):.4f}, "
                  f"WER={result.get('wer', 0):.4f}, "
                  f"OCR_len={len(result.get('ocr_text', ''))}")
        if len(analysis['failed']) > 10:
            print(f"  ... and {len(analysis['failed']) - 10} more")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze full DeepSeek-OCR results"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Results directory (e.g., bl_newspapers_YYYYMMDD_HHMMSS_FIXED/base_size_1024)"
    )
    parser.add_argument(
        "--olmocr-wer",
        type=float,
        default=0.05,
        help="OLMoCR baseline WER (default: 0.05 = 5%%)"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1

    # Load results
    results = load_results(args.results_dir)

    if not results:
        print("Error: No results found!")
        return 1

    # Analyze
    analysis = analyze_results(results)

    # Print report
    print_report(analysis, args.olmocr_wer)

    return 0


if __name__ == "__main__":
    exit(main())

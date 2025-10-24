#!/usr/bin/env python3
"""
Extract actual OCR text from SLURM job logs where it was printed but not saved.

The DeepSeek model outputs structured text with XML-like tags:
<|ref|>text<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
Actual OCR text here...

This script parses those logs to extract the real OCR output.
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List


def parse_ocr_output(log_content: str) -> Dict[str, str]:
    """
    Parse OCR output from DeepSeek log content.

    Returns dict mapping image_id to extracted plain text.
    """
    results = {}

    # Pattern to find processing blocks
    # Looking for: [N/M] Processing: image_id.{tif,jpg,png}
    # Followed by structured output
    # Ending at Time: X.XXs

    # Updated to handle .tif, .jpg, .png extensions
    pattern = r'\[(\d+)/\d+\] Processing: (\d+)\.(tif|jpg|png)\n(.*?)\n  Time:'

    matches = re.finditer(pattern, log_content, re.DOTALL)

    for match in matches:
        idx = match.group(1)
        image_id = match.group(2)
        extension = match.group(3)
        ocr_block = match.group(4)

        # Extract text portions (ignoring XML-like tags and coordinates)
        text_parts = []

        # Pattern for text after </det|> tags
        # The text can include markdown headers (## ), so we need to capture everything
        text_pattern = r'<\|/det\|>\n(.+?)(?=\n<\|ref\||$)'

        for text_match in re.finditer(text_pattern, ocr_block, re.DOTALL):
            text = text_match.group(1).strip()
            # Remove markdown header markers if present at start
            text = re.sub(r'^## ', '', text)
            # Skip empty or very short strings (likely artifacts)
            if len(text) > 10:
                text_parts.append(text)

        if text_parts:
            results[image_id] = '\n\n'.join(text_parts)
        else:
            # If no substantial text found, mark as empty
            results[image_id] = ""

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract OCR text from DeepSeek SLURM logs"
    )
    parser.add_argument(
        "log_file",
        type=Path,
        help="SLURM log file (e.g., deepseek_bl_news-3295859.out)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for extracted texts (default: same as log file)"
    )

    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return 1

    output_dir = args.output_dir or (args.log_file.parent / "extracted_from_logs")
    output_dir.mkdir(exist_ok=True)

    print(f"Reading log: {args.log_file}")
    with open(args.log_file, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()

    print("Extracting OCR text...")
    results = parse_ocr_output(log_content)

    print(f"\nFound {len(results)} images with OCR output")

    # Save extracted texts
    for image_id, text in results.items():
        output_file = output_dir / f"{image_id}_ocr_from_log.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)

        char_count = len(text)
        print(f"  {image_id}: {char_count} chars")

    print(f"\nExtracted texts saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())

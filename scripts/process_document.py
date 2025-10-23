#!/usr/bin/env python3
"""
DeepSeek-OCR Document Processing Script
Processes PDFs or images with DeepSeek-OCR and saves results in multiple formats.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from pdf2image import convert_from_path
import yaml


class DeepSeekOCRProcessor:
    """Handles DeepSeek-OCR model loading and document processing."""

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-OCR",
                 device: str = "cuda",
                 base_size: int = 1024,
                 image_size: int = 1024):
        """
        Initialize DeepSeek-OCR processor.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu)
            base_size: Base size parameter (Tiny=448, Small=672, Base=1024, Large=1536)
            image_size: Image size parameter
        """
        self.device = device
        self.base_size = base_size
        self.image_size = image_size

        print(f"Loading DeepSeek-OCR model: {model_name}")
        print(f"Device: {device}, Base size: {base_size}, Image size: {image_size}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()

        print("Model loaded successfully!")

    def process_image(self, image: Image.Image, prompt: str = "Free OCR") -> Dict[str, Any]:
        """
        Process a single image with DeepSeek-OCR.

        Args:
            image: PIL Image object
            prompt: OCR prompt ("Free OCR" or "Convert document to markdown")

        Returns:
            Dictionary with OCR results and metadata
        """
        start_time = time.time()

        # Prepare inputs
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}]

        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        elapsed = time.time() - start_time

        return {
            "text": text.strip(),
            "processing_time": elapsed,
            "prompt": prompt,
            "model_config": {
                "base_size": self.base_size,
                "image_size": self.image_size
            }
        }

    def process_pdf(self, pdf_path: Path,
                   output_dir: Path,
                   prompt: str = "Free OCR",
                   dpi: int = 300) -> Dict[str, Any]:
        """
        Process a PDF document page by page.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory for output files
            prompt: OCR prompt
            dpi: DPI for PDF to image conversion

        Returns:
            Processing results and metadata
        """
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*60}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert PDF to images
        print(f"Converting PDF to images (DPI={dpi})...")
        conversion_start = time.time()
        images = convert_from_path(str(pdf_path), dpi=dpi)
        conversion_time = time.time() - conversion_start
        print(f"Converted {len(images)} pages in {conversion_time:.2f}s")

        # Process each page
        results = {
            "document": pdf_path.name,
            "total_pages": len(images),
            "pages": [],
            "dpi": dpi,
            "prompt": prompt,
            "conversion_time": conversion_time,
            "processing_start": datetime.now().isoformat()
        }

        total_processing_time = 0

        for page_num, image in enumerate(images, 1):
            print(f"\nProcessing page {page_num}/{len(images)}...", end=" ")

            page_result = self.process_image(image, prompt)
            page_result["page_number"] = page_num
            results["pages"].append(page_result)

            total_processing_time += page_result["processing_time"]

            print(f"✓ ({page_result['processing_time']:.2f}s, "
                  f"{len(page_result['text'])} chars)")

        results["total_processing_time"] = total_processing_time
        results["pages_per_second"] = len(images) / total_processing_time if total_processing_time > 0 else 0
        results["processing_end"] = datetime.now().isoformat()

        # Save results
        self._save_results(pdf_path, output_dir, results)

        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Total time: {total_processing_time:.2f}s")
        print(f"Speed: {results['pages_per_second']:.2f} pages/sec")
        print(f"{'='*60}\n")

        return results

    def _save_results(self, pdf_path: Path, output_dir: Path, results: Dict[str, Any]):
        """Save processing results in multiple formats."""
        base_name = pdf_path.stem

        # Save JSON
        json_path = output_dir / f"{base_name}.deepseek.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {json_path}")

        # Save plain text
        txt_path = output_dir / f"{base_name}.deepseek.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for page in results["pages"]:
                f.write(f"{'='*60}\n")
                f.write(f"Page {page['page_number']}\n")
                f.write(f"{'='*60}\n\n")
                f.write(page["text"])
                f.write("\n\n")
        print(f"Saved text: {txt_path}")

        # Save markdown (if markdown prompt was used)
        if "markdown" in results["prompt"].lower():
            md_path = output_dir / f"{base_name}.deepseek.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {pdf_path.name}\n\n")
                for page in results["pages"]:
                    f.write(f"## Page {page['page_number']}\n\n")
                    f.write(page["text"])
                    f.write("\n\n---\n\n")
            print(f"Saved markdown: {md_path}")

        # Save metadata summary
        summary_path = output_dir / f"{base_name}.summary.yaml"
        summary = {
            "document": results["document"],
            "total_pages": results["total_pages"],
            "total_processing_time": results["total_processing_time"],
            "pages_per_second": results["pages_per_second"],
            "dpi": results["dpi"],
            "prompt": results["prompt"],
            "model_config": results["pages"][0]["model_config"] if results["pages"] else {},
            "processing_start": results["processing_start"],
            "processing_end": results["processing_end"]
        }
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        print(f"Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process documents with DeepSeek-OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic OCR
  python process_document.py --input document.pdf --output results/

  # Markdown output
  python process_document.py --input doc.pdf --output results/ --prompt "Convert document to markdown"

  # Custom model size
  python process_document.py --input doc.pdf --output results/ --base-size 1536

Model Sizes:
  Tiny:  base_size=448
  Small: base_size=672
  Base:  base_size=1024 (default)
  Large: base_size=1536
        """
    )

    parser.add_argument("--input", "-i", required=True, type=Path,
                       help="Input PDF file")
    parser.add_argument("--output", "-o", required=True, type=Path,
                       help="Output directory")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR",
                       help="HuggingFace model identifier")
    parser.add_argument("--prompt", default="Free OCR",
                       choices=["Free OCR", "Convert document to markdown"],
                       help="OCR prompt")
    parser.add_argument("--base-size", type=int, default=1024,
                       help="Base size parameter (448/672/1024/1536)")
    parser.add_argument("--image-size", type=int, default=1024,
                       help="Image size parameter")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for PDF conversion")
    parser.add_argument("--device", default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Initialize processor
    processor = DeepSeekOCRProcessor(
        model_name=args.model,
        device=args.device,
        base_size=args.base_size,
        image_size=args.image_size
    )

    # Process document
    results = processor.process_pdf(
        pdf_path=args.input,
        output_dir=args.output,
        prompt=args.prompt,
        dpi=args.dpi
    )

    return 0


if __name__ == "__main__":
    exit(main())

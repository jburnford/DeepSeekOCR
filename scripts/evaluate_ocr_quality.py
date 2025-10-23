#!/usr/bin/env python3
"""
OCR Quality Evaluation Framework
Compare OCR outputs from different models (DeepSeek-OCR, OLMoCR, Marker, etc.)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import yaml

try:
    import jiwer
    import editdistance
except ImportError:
    print("Warning: jiwer or editdistance not installed. Install with:")
    print("  pip install jiwer editdistance")
    exit(1)


class OCRQualityEvaluator:
    """Evaluate and compare OCR quality across different models."""

    def __init__(self):
        self.metrics = {}

    def load_text_file(self, file_path: Path) -> str:
        """Load text from a file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def load_json_result(self, file_path: Path) -> Dict[str, Any]:
        """Load OCR results from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def compute_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Compute basic text statistics for quality assessment.

        Args:
            text: OCR output text

        Returns:
            Dictionary of statistics
        """
        # Remove excessive whitespace for analysis
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Character counts
        total_chars = len(cleaned_text)
        alpha_chars = sum(c.isalpha() for c in cleaned_text)
        digit_chars = sum(c.isdigit() for c in cleaned_text)
        punct_chars = sum(c in '.,;:!?"\'()-' for c in cleaned_text)
        space_chars = sum(c.isspace() for c in cleaned_text)

        # Word analysis
        words = cleaned_text.split()
        word_lengths = [len(w) for w in words]

        # Line analysis
        line_lengths = [len(line) for line in lines]

        # Detect potential OCR errors
        # Very long words often indicate OCR errors
        suspicious_words = [w for w in words if len(w) > 20]

        # Repeated characters (e.g., "aaaa") often indicate errors
        repeated_pattern = re.findall(r'(.)\1{4,}', text)

        # Mixed case within words (excluding proper nouns pattern)
        mixed_case_words = [w for w in words if
                           any(c.islower() for c in w) and
                           any(c.isupper() for c in w[1:])]

        return {
            "total_characters": total_chars,
            "alphabetic_chars": alpha_chars,
            "digit_chars": digit_chars,
            "punctuation_chars": punct_chars,
            "whitespace_chars": space_chars,
            "total_words": len(words),
            "unique_words": len(set(words)),
            "avg_word_length": sum(word_lengths) / len(word_lengths) if word_lengths else 0,
            "max_word_length": max(word_lengths) if word_lengths else 0,
            "total_lines": len(lines),
            "avg_line_length": sum(line_lengths) / len(line_lengths) if line_lengths else 0,
            "suspicious_long_words": len(suspicious_words),
            "repeated_char_sequences": len(repeated_pattern),
            "mixed_case_words": len(mixed_case_words),
            "alpha_ratio": alpha_chars / total_chars if total_chars > 0 else 0,
            "digit_ratio": digit_chars / total_chars if total_chars > 0 else 0,
        }

    def compute_similarity_metrics(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Compare two OCR outputs using multiple similarity metrics.

        Args:
            text1: First OCR output
            text2: Second OCR output (reference or alternative)

        Returns:
            Dictionary of similarity metrics
        """
        # Normalize whitespace
        text1_norm = re.sub(r'\s+', ' ', text1).strip()
        text2_norm = re.sub(r'\s+', ' ', text2).strip()

        # Character Error Rate (CER)
        cer = jiwer.cer(text2_norm, text1_norm)

        # Word Error Rate (WER)
        wer = jiwer.wer(text2_norm, text1_norm)

        # Edit distance
        edit_dist = editdistance.eval(text1_norm, text2_norm)
        max_len = max(len(text1_norm), len(text2_norm))
        normalized_edit_dist = edit_dist / max_len if max_len > 0 else 0

        # Word overlap
        words1 = set(text1_norm.lower().split())
        words2 = set(text2_norm.lower().split())

        if words1 or words2:
            word_precision = len(words1 & words2) / len(words1) if words1 else 0
            word_recall = len(words1 & words2) / len(words2) if words2 else 0
            word_f1 = (2 * word_precision * word_recall / (word_precision + word_recall)
                      if (word_precision + word_recall) > 0 else 0)
        else:
            word_precision = word_recall = word_f1 = 0

        return {
            "character_error_rate": cer,
            "word_error_rate": wer,
            "edit_distance": edit_dist,
            "normalized_edit_distance": normalized_edit_dist,
            "word_precision": word_precision,
            "word_recall": word_recall,
            "word_f1": word_f1,
        }

    def evaluate_model_output(self, result_file: Path,
                            reference_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single model's OCR output.

        Args:
            result_file: Path to OCR result JSON
            reference_text: Optional ground truth text for comparison

        Returns:
            Evaluation metrics
        """
        result_data = self.load_json_result(result_file)

        # Extract full text
        if "pages" in result_data:
            # DeepSeek-OCR format
            full_text = "\n\n".join(page["text"] for page in result_data["pages"])
            model_name = "deepseek-ocr"
            pages = result_data["total_pages"]
            processing_time = result_data["total_processing_time"]
            pages_per_sec = result_data["pages_per_second"]
        else:
            # Generic format
            full_text = result_data.get("text", "")
            model_name = result_data.get("model", "unknown")
            pages = result_data.get("pages", 1)
            processing_time = result_data.get("processing_time", 0)
            pages_per_sec = pages / processing_time if processing_time > 0 else 0

        # Compute statistics
        stats = self.compute_text_statistics(full_text)

        evaluation = {
            "model": model_name,
            "document": result_data.get("document", result_file.stem),
            "pages": pages,
            "processing_time": processing_time,
            "pages_per_second": pages_per_sec,
            "text_statistics": stats,
        }

        # Compare to reference if available
        if reference_text:
            similarity = self.compute_similarity_metrics(full_text, reference_text)
            evaluation["similarity_to_reference"] = similarity

        return evaluation

    def compare_models(self, model_results: Dict[str, Path]) -> Dict[str, Any]:
        """
        Compare OCR outputs from multiple models.

        Args:
            model_results: Dictionary mapping model names to result file paths

        Returns:
            Comparison metrics
        """
        evaluations = {}

        # Evaluate each model
        for model_name, result_file in model_results.items():
            print(f"Evaluating {model_name}...")
            evaluations[model_name] = self.evaluate_model_output(result_file)

        # Cross-compare models
        model_names = list(evaluations.keys())
        if len(model_names) >= 2:
            comparisons = {}

            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    # Extract texts
                    data1 = self.load_json_result(model_results[model1])
                    data2 = self.load_json_result(model_results[model2])

                    if "pages" in data1:
                        text1 = "\n\n".join(p["text"] for p in data1["pages"])
                    else:
                        text1 = data1.get("text", "")

                    if "pages" in data2:
                        text2 = "\n\n".join(p["text"] for p in data2["pages"])
                    else:
                        text2 = data2.get("text", "")

                    # Compare
                    similarity = self.compute_similarity_metrics(text1, text2)
                    comparisons[f"{model1}_vs_{model2}"] = similarity

            return {
                "individual_evaluations": evaluations,
                "cross_model_comparisons": comparisons
            }

        return {"individual_evaluations": evaluations}

    def generate_report(self, comparison_results: Dict[str, Any],
                       output_file: Path):
        """Generate a human-readable comparison report."""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("OCR Quality Evaluation Report\n")
            f.write("=" * 80 + "\n\n")

            # Individual evaluations
            for model, eval_data in comparison_results["individual_evaluations"].items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Model: {model}\n")
                f.write(f"{'=' * 80}\n")

                f.write(f"\nDocument: {eval_data['document']}\n")
                f.write(f"Pages: {eval_data['pages']}\n")
                f.write(f"Processing Time: {eval_data['processing_time']:.2f}s\n")
                f.write(f"Speed: {eval_data['pages_per_second']:.2f} pages/sec\n")

                f.write(f"\nText Statistics:\n")
                f.write("-" * 40 + "\n")
                stats = eval_data['text_statistics']
                f.write(f"  Total characters: {stats['total_characters']:,}\n")
                f.write(f"  Total words: {stats['total_words']:,}\n")
                f.write(f"  Unique words: {stats['unique_words']:,}\n")
                f.write(f"  Average word length: {stats['avg_word_length']:.2f}\n")
                f.write(f"  Total lines: {stats['total_lines']:,}\n")
                f.write(f"  Average line length: {stats['avg_line_length']:.2f}\n")

                f.write(f"\nQuality Indicators:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Alphabetic ratio: {stats['alpha_ratio']:.2%}\n")
                f.write(f"  Digit ratio: {stats['digit_ratio']:.2%}\n")
                f.write(f"  Suspicious long words: {stats['suspicious_long_words']}\n")
                f.write(f"  Repeated char sequences: {stats['repeated_char_sequences']}\n")
                f.write(f"  Mixed case words: {stats['mixed_case_words']}\n")

            # Cross-model comparisons
            if "cross_model_comparisons" in comparison_results:
                f.write(f"\n\n{'=' * 80}\n")
                f.write("Cross-Model Comparisons\n")
                f.write(f"{'=' * 80}\n")

                for comparison_name, metrics in comparison_results["cross_model_comparisons"].items():
                    f.write(f"\n{comparison_name}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  Character Error Rate: {metrics['character_error_rate']:.4f}\n")
                    f.write(f"  Word Error Rate: {metrics['word_error_rate']:.4f}\n")
                    f.write(f"  Word F1 Score: {metrics['word_f1']:.4f}\n")
                    f.write(f"  Normalized Edit Distance: {metrics['normalized_edit_distance']:.4f}\n")

        print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare OCR quality across models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model output
  python evaluate_ocr_quality.py --input results/doc.deepseek.json --output evaluation.yaml

  # Compare two models
  python evaluate_ocr_quality.py \\
    --compare deepseek=results/doc.deepseek.json olmocr=results/doc.olmocr.json \\
    --output comparison.yaml

  # Compare with reference text
  python evaluate_ocr_quality.py --input results/doc.json --reference ground_truth.txt
        """
    )

    parser.add_argument("--input", "-i", type=Path,
                       help="Single OCR result file to evaluate")
    parser.add_argument("--compare", "-c", nargs='+',
                       help="Compare multiple models (format: name=path)")
    parser.add_argument("--reference", "-r", type=Path,
                       help="Ground truth text for comparison")
    parser.add_argument("--output", "-o", type=Path, required=True,
                       help="Output file for results (YAML)")
    parser.add_argument("--report", type=Path,
                       help="Generate human-readable report (TXT)")

    args = parser.parse_args()

    evaluator = OCRQualityEvaluator()

    # Single file evaluation
    if args.input:
        reference_text = None
        if args.reference:
            reference_text = evaluator.load_text_file(args.reference)

        results = evaluator.evaluate_model_output(args.input, reference_text)

        # Save results
        with open(args.output, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)

        print(f"Evaluation saved to: {args.output}")

    # Multi-model comparison
    elif args.compare:
        model_results = {}
        for item in args.compare:
            try:
                name, path = item.split('=', 1)
                model_results[name] = Path(path)
            except ValueError:
                print(f"Error: Invalid format '{item}'. Expected 'name=path'")
                return 1

        results = evaluator.compare_models(model_results)

        # Save results
        with open(args.output, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)

        print(f"Comparison saved to: {args.output}")

        # Generate human-readable report
        if args.report:
            evaluator.generate_report(results, args.report)

    else:
        print("Error: Must specify either --input or --compare")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

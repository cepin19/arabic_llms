#!/usr/bin/env python3
"""
Evaluation script for AraBench dataset (reverse direction) using realtime2.py
Translates all Arabic files to English.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, List
from openai import AsyncOpenAI
import sys
from sacrebleu import BLEU, CHRF

# Import translation function from realtime2.py
sys.path.insert(0, str(Path(__file__).parent))
from realtime2 import translate_file, MODEL, MAX_CONCURRENT_REQUESTS


def get_reference_file(input_file: Path, dataset_dir: Path) -> Optional[Path]:
    """
    Get the reference English file corresponding to an Arabic file.
    Example: madar.test.glf.0.qa.ar -> madar.test.glf.0.qa.en
    """
    # Replace .ar with .en
    ref_name = input_file.name.replace('.ar', '.en')
    ref_file = dataset_dir / ref_name
    
    if ref_file.exists():
        return ref_file
    return None


def compute_metrics(hypothesis: List[str], reference: List[str]) -> Dict[str, float]:
    """
    Compute BLEU and CHRF scores.
    """
    # Prepare for sacrebleu (expects list of strings)
    bleu = BLEU()
    chrf = CHRF()
    
    # Compute scores
    bleu_score = bleu.corpus_score(hypothesis, [reference]).score
    chrf_score = chrf.corpus_score(hypothesis, [reference]).score
    
    return {
        'BLEU': round(bleu_score, 4),
        'CHRF': round(chrf_score, 4)
    }


async def translate_arabench_file(
    client: AsyncOpenAI,
    model: str,
    input_file: Path,
    output_dir: Path,
    dataset_dir: Path,
    base_url: str = "https://api.openai.com/v1"
) -> Optional[Dict]:
    """
    Translate a single AraBench Arabic file to English.
    
    Returns metrics dictionary if reference file exists, None otherwise.
    """
    print(f"\n{'='*80}")
    print(f"📄 Processing: {input_file.name}")
    print(f"{'='*80}\n")
    
    # Get output path
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_file.stem.replace('.ar', '')
    english_output = output_dir / f"{base_name}.en.translated"
    
    # Get reference file
    ref_file = get_reference_file(input_file, dataset_dir)
    
    # Translate to English
    print(f"🔄 Translating to English...")
    print(f"   Input:  {input_file}")
    print(f"   Output: {english_output}")
    try:
        await translate_file(client, model, input_file, english_output, "English")
        print(f"✅ Completed English translation\n")
    except Exception as e:
        print(f"❌ Error translating to English: {e}\n")
        return None
    
    # Compute metrics if reference file exists
    metrics = {}
    if ref_file and ref_file.exists():
        print(f"📊 Computing metrics against reference: {ref_file.name}")
        
        # Load reference and hypothesis
        ref_lines = ref_file.read_text(encoding="utf-8").strip().split('\n')
        hyp_lines = english_output.read_text(encoding="utf-8").strip().split('\n')
        
        # Ensure same length
        min_len = min(len(ref_lines), len(hyp_lines))
        ref_lines = ref_lines[:min_len]
        hyp_lines = hyp_lines[:min_len]
        
        # Compute metrics
        file_metrics = compute_metrics(hyp_lines, ref_lines)
        metrics.update(file_metrics)
        metrics['filename'] = input_file.name
        metrics['num_sentences'] = min_len
        
        print(f"   BLEU: {file_metrics['BLEU']:.4f}, CHRF: {file_metrics['CHRF']:.4f}")
    else:
        print(f"⚠️  Reference file not found: {ref_file}, skipping metrics computation")
    
    return metrics


async def process_all_files(
    dataset_dir: Path,
    output_dir: Path,
    model: str = MODEL,
    base_url: str = "https://api.openai.com/v1"
):
    """
    Process all Arabic files in the AraBench dataset.
    """
    # Find all .ar files (excluding .ids files and .general/.dialect files)
    ar_files = [
        f for f in dataset_dir.glob("*.ar")
        if not f.name.endswith(".ids")
        and '.general' not in f.name
        and not any(f.name.endswith(f'.ar.{code}') for code in ['ma', 'tn', 'ms', 'sy', 'eg', 'jo', 'ps', 'lv', 'qa', 'lb', 'iq', 'om', 'sa', 'ye', 'pa', 'dz', 'ly', 'sd'])
    ]
    
    if not ar_files:
        print(f"❌ No .ar files found in {dataset_dir}")
        return
    
    print(f"📊 Found {len(ar_files)} Arabic files to translate\n")
    
    # Initialize client
    client = AsyncOpenAI(base_url=base_url)
    
    # Store all metrics
    all_metrics = []
    
    # Process files sequentially (to avoid overwhelming the API)
    for i, ar_file in enumerate(ar_files, 1):
        print(f"\n[{i}/{len(ar_files)}] Processing {ar_file.name}...")
        metrics = await translate_arabench_file(client, model, ar_file, output_dir, dataset_dir, base_url)
        if metrics:
            all_metrics.append(metrics)
    
    # Save scores to file
    scores_file = output_dir / "scores.json"
    scores_summary = {
        'model': model,
        'total_files': len(ar_files),
        'files_with_metrics': len(all_metrics),
        'results': all_metrics
    }
    
    # Compute averages
    if all_metrics:
        bleu_scores = [m['BLEU'] for m in all_metrics if 'BLEU' in m]
        chrf_scores = [m['CHRF'] for m in all_metrics if 'CHRF' in m]
        
        scores_summary['averages'] = {
            'BLEU': round(sum(bleu_scores) / len(bleu_scores), 4) if bleu_scores else 0,
            'CHRF': round(sum(chrf_scores) / len(chrf_scores), 4) if chrf_scores else 0,
        }
    
    # Save JSON
    scores_file.write_text(json.dumps(scores_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Also save a human-readable summary
    summary_file = output_dir / "scores_summary.txt"
    with summary_file.open('w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("AraBench Evaluation Results (Arabic -> English)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model}\n")
        f.write(f"Total files processed: {len(ar_files)}\n")
        f.write(f"Files with metrics: {len(all_metrics)}\n\n")
        
        if all_metrics and 'averages' in scores_summary:
            f.write("Average Scores:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  BLEU: {scores_summary['averages']['BLEU']:.4f}\n")
            f.write(f"  CHRF: {scores_summary['averages']['CHRF']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Per-file Results:\n")
        f.write("=" * 80 + "\n\n")
        
        for metrics in all_metrics:
            f.write(f"File: {metrics.get('filename', 'unknown')}\n")
            f.write(f"Sentences: {metrics.get('num_sentences', 0)}\n")
            if 'BLEU' in metrics:
                f.write(f"  BLEU: {metrics['BLEU']:.4f}, CHRF: {metrics['CHRF']:.4f}\n")
            f.write("\n")
    
    print(f"\n{'='*80}")
    print(f"✅ All translations completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Scores saved to: {scores_file}")
    print(f"📄 Summary saved to: {summary_file}")
    if all_metrics and 'averages' in scores_summary:
        print(f"\n📈 Average Scores:")
        print(f"   BLEU: {scores_summary['averages']['BLEU']:.4f}, CHRF: {scores_summary['averages']['CHRF']:.4f}")
    print(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate AraBench dataset by translating Arabic files to English"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("AraBench_dataset"),
        help="Path to AraBench dataset directory (default: AraBench_dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("arabench_translations_reverse"),
        help="Path to output directory for translations (default: arabench_translations_reverse)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help=f"Model to use for translation (default: {MODEL})"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the API (default: https://api.openai.com/v1)"
    )
    
    args = parser.parse_args()
    
    # Validate dataset directory
    if not args.dataset_dir.exists():
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)
    
    # Run async processing
    asyncio.run(process_all_files(
        args.dataset_dir,
        args.output_dir,
        args.model,
        args.base_url
    ))


if __name__ == "__main__":
    main()


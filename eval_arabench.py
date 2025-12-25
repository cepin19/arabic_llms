#!/usr/bin/env python3
"""
Evaluation script for AraBench dataset using realtime2.py
Translates all English files to Arabic first, then to specific dialects.
"""

import asyncio
import re
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from collections import defaultdict
from openai import AsyncOpenAI
import sys
from sacrebleu import BLEU, CHRF

# Import translation function from realtime2.py
sys.path.insert(0, str(Path(__file__).parent))
from realtime2 import translate_file, MODEL, MAX_CONCURRENT_REQUESTS

# Dialect code to full name mapping
DIALECT_NAMES = {
    'ma': 'Moroccan Arabic',
    'tn': 'Tunisian Arabic',
    'ms': 'Modern Standard Arabic',
    'sy': 'Syrian Arabic',
    'eg': 'Egyptian Arabic',
    'jo': 'Jordanian Arabic',
    'ps': 'Palestinian Arabic',
    'lv': 'Levantine Arabic',
    'qa': 'Qatari Arabic',
    'lb': 'Lebanese Arabic',
    'iq': 'Iraqi Arabic',
    'om': 'Omani Arabic',
    'sa': 'Saudi Arabic',
    'ye': 'Yemeni Arabic',
    'pa': 'Palestinian Arabic',
    'dz': 'Algerian Arabic',
    'ly': 'Libyan Arabic',
    'sd': 'Sudanese Arabic',
}


def extract_dialect_from_filename(filename: str) -> Optional[str]:
    """
    Extract dialect code from AraBench filename.
    Pattern: dataset.split.source.0.dialect.en
    Example: madar.test.glf.0.qa.en -> qa
    """
    # Match pattern: anything.dialect.en (not .ids)
    match = re.search(r'\.([a-z]{2})\.en$', filename)
    if match:
        return match.group(1)
    return None


def get_output_paths(input_file: Path, output_dir: Path, dialect: str) -> Tuple[Path, Path]:
    """
    Generate output paths for Arabic and dialect translations.
    Returns: (arabic_output_path, dialect_output_path)
    """
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base filename without extension
    base_name = input_file.stem.replace('.en', '')
    
    # Output paths
    arabic_output = output_dir / f"{base_name}.ar.general"
    dialect_output = output_dir / f"{base_name}.ar.{dialect}"
    
    return arabic_output, dialect_output


def get_reference_file(input_file: Path, dataset_dir: Path) -> Optional[Path]:
    """
    Get the reference Arabic file corresponding to an English file.
    Example: madar.test.glf.0.qa.en -> madar.test.glf.0.qa.ar
    """
    # Replace .en with .ar
    ref_name = input_file.name.replace('.en', '.ar')
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
    Translate a single AraBench English file:
    1. First to Arabic (general)
    2. Then to specific dialect
    
    Returns metrics dictionary if reference file exists, None otherwise.
    """
    dialect_code = extract_dialect_from_filename(input_file.name)
    
    if not dialect_code:
        print(f"⚠️  Could not extract dialect from {input_file.name}, skipping...")
        return None
    
    dialect_name = DIALECT_NAMES.get(dialect_code, f"{dialect_code} Arabic")
    
    print(f"\n{'='*80}")
    print(f"📄 Processing: {input_file.name}")
    print(f"🌍 Dialect: {dialect_name} ({dialect_code})")
    print(f"{'='*80}\n")
    
    # Get output paths
    arabic_output, dialect_output = get_output_paths(input_file, output_dir, dialect_code)
    
    # Get reference file
    ref_file = get_reference_file(input_file, dataset_dir)
    
    # Step 1: Translate to Arabic (general)
    if arabic_output.exists():
        print(f"⏭️  Step 1: Arabic (general) translation already exists, skipping...")
        print(f"   Output: {arabic_output}\n")
    else:
    print(f"🔄 Step 1: Translating to Arabic (general)...")
    print(f"   Input:  {input_file}")
    print(f"   Output: {arabic_output}")
    try:
        await translate_file(client, model, input_file, arabic_output, "Arabic")
        print(f"✅ Completed Arabic translation\n")
    except Exception as e:
        print(f"❌ Error translating to Arabic: {e}\n")
        return None
    
    # Step 2: Translate to specific dialect
    if dialect_output.exists():
        print(f"⏭️  Step 2: {dialect_name} translation already exists, skipping...")
        print(f"   Output: {dialect_output}\n")
    else:
    print(f"🔄 Step 2: Translating to {dialect_name}...")
    print(f"   Input:  {input_file}")
    print(f"   Output: {dialect_output}")
    try:
        await translate_file(client, model, input_file, dialect_output, dialect_name)
        print(f"✅ Completed {dialect_name} translation\n")
    except Exception as e:
        print(f"❌ Error translating to {dialect_name}: {e}\n")
        return None
    
    # Compute metrics if reference file exists
    metrics = {}
    if ref_file and ref_file.exists():
        print(f"📊 Computing metrics against reference: {ref_file.name}")
        
        # Load reference and hypotheses
        ref_lines = ref_file.read_text(encoding="utf-8").strip().split('\n')
        arabic_hyp = arabic_output.read_text(encoding="utf-8").strip().split('\n')
        dialect_hyp = dialect_output.read_text(encoding="utf-8").strip().split('\n')
        
        # Ensure same length
        min_len = min(len(ref_lines), len(arabic_hyp), len(dialect_hyp))
        ref_lines = ref_lines[:min_len]
        arabic_hyp = arabic_hyp[:min_len]
        dialect_hyp = dialect_hyp[:min_len]
        
        # Compute metrics for Arabic (general)
        arabic_metrics = compute_metrics(arabic_hyp, ref_lines)
        metrics['arabic_general'] = arabic_metrics
        print(f"   Arabic (general) - BLEU: {arabic_metrics['BLEU']:.4f}, CHRF: {arabic_metrics['CHRF']:.4f}")
        
        # Compute metrics for dialect
        dialect_metrics = compute_metrics(dialect_hyp, ref_lines)
        metrics['dialect'] = dialect_metrics
        print(f"   {dialect_name} - BLEU: {dialect_metrics['BLEU']:.4f}, CHRF: {dialect_metrics['CHRF']:.4f}")
        
        metrics['filename'] = input_file.name
        metrics['dialect_code'] = dialect_code
        metrics['dialect_name'] = dialect_name
        metrics['num_sentences'] = min_len
    else:
        print(f"⚠️  Reference file not found: {ref_file}, skipping metrics computation")
    
    return metrics


async def translate_arabench_file_roundtrip(
    client: AsyncOpenAI,
    model: str,
    input_file: Path,
    forward_output_dir: Path,
    roundtrip_output_dir: Path,
    dataset_dir: Path,
    base_url: str = "https://api.openai.com/v1"
) -> Optional[Dict]:
    """
    Round-trip translation: English -> Arabic (dialect) -> English
    
    1. Translate English to Arabic (general)
    2. Translate English to specific dialect
    3. Translate dialect Arabic back to English
    4. Compute metrics against original English source
    
    Returns metrics dictionary.
    """
    dialect_code = extract_dialect_from_filename(input_file.name)
    
    if not dialect_code:
        print(f"⚠️  Could not extract dialect from {input_file.name}, skipping...")
        return None
    
    dialect_name = DIALECT_NAMES.get(dialect_code, f"{dialect_code} Arabic")
    
    print(f"\n{'='*80}")
    print(f"📄 Processing (round-trip): {input_file.name}")
    print(f"🌍 Dialect: {dialect_name} ({dialect_code})")
    print(f"{'='*80}\n")
    
    # Get forward output paths (ensure forward dir exists for new translations)
    forward_output_dir.mkdir(parents=True, exist_ok=True)
    arabic_output, dialect_output = get_output_paths(input_file, forward_output_dir, dialect_code)
    
    # Get round-trip output path
    roundtrip_output_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_file.stem.replace('.en', '')
    roundtrip_output = roundtrip_output_dir / f"{base_name}.en.roundtrip"
    
    # Debug: show paths being checked
    print(f"   Checking forward outputs: {arabic_output.exists()=}, {dialect_output.exists()=}")
    
    # Step 1: Translate to Arabic (general) - reuse from forward if exists
    if arabic_output.exists():
        print(f"⏭️  Step 1: Arabic (general) translation already exists, skipping...")
        print(f"   Output: {arabic_output}\n")
    else:
        print(f"🔄 Step 1: Translating to Arabic (general)...")
        print(f"   Input:  {input_file}")
        print(f"   Output: {arabic_output}")
        try:
            await translate_file(client, model, input_file, arabic_output, "Arabic")
            print(f"✅ Completed Arabic translation\n")
        except Exception as e:
            print(f"❌ Error translating to Arabic: {e}\n")
            return None
    
    # Step 2: Translate to specific dialect - reuse from forward if exists
    if dialect_output.exists():
        print(f"⏭️  Step 2: {dialect_name} translation already exists, skipping...")
        print(f"   Output: {dialect_output}\n")
    else:
        print(f"🔄 Step 2: Translating to {dialect_name}...")
        print(f"   Input:  {input_file}")
        print(f"   Output: {dialect_output}")
        try:
            await translate_file(client, model, input_file, dialect_output, dialect_name)
            print(f"✅ Completed {dialect_name} translation\n")
        except Exception as e:
            print(f"❌ Error translating to {dialect_name}: {e}\n")
            return None
    
    # Step 3: Translate dialect Arabic back to English
    if roundtrip_output.exists():
        print(f"⏭️  Step 3: Round-trip English translation already exists, skipping...")
        print(f"   Output: {roundtrip_output}\n")
    else:
        print(f"🔄 Step 3: Translating {dialect_name} back to English (round-trip)...")
        print(f"   Input:  {dialect_output}")
        print(f"   Output: {roundtrip_output}")
        try:
            await translate_file(client, model, dialect_output, roundtrip_output, "English")
            print(f"✅ Completed round-trip English translation\n")
        except Exception as e:
            print(f"❌ Error translating back to English: {e}\n")
            return None
    
    # Step 4: Compute metrics against original English source
    metrics = {}
    print(f"📊 Computing round-trip metrics against original: {input_file.name}")
    
    # Load original English source and round-trip hypothesis
    source_lines = input_file.read_text(encoding="utf-8").strip().split('\n')
    roundtrip_lines = roundtrip_output.read_text(encoding="utf-8").strip().split('\n')
    
    # Ensure same length
    min_len = min(len(source_lines), len(roundtrip_lines))
    source_lines = source_lines[:min_len]
    roundtrip_lines = roundtrip_lines[:min_len]
    
    # Compute metrics for round-trip
    roundtrip_metrics = compute_metrics(roundtrip_lines, source_lines)
    metrics['roundtrip'] = roundtrip_metrics
    print(f"   Round-trip - BLEU: {roundtrip_metrics['BLEU']:.4f}, CHRF: {roundtrip_metrics['CHRF']:.4f}")
    
    metrics['filename'] = input_file.name
    metrics['dialect_code'] = dialect_code
    metrics['dialect_name'] = dialect_name
    metrics['num_sentences'] = min_len
    
    return metrics


async def translate_arabench_file_reverse(
    client: AsyncOpenAI,
    model: str,
    input_file: Path,
    output_dir: Path,
    dataset_dir: Path,
    base_url: str = "https://api.openai.com/v1"
) -> Optional[Dict]:
    """
    Translate a single AraBench Arabic file to English (reverse translation).
    
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
    ref_name = input_file.name.replace('.ar', '.en')
    ref_file = dataset_dir / ref_name
    
    # Translate to English
    if english_output.exists():
        print(f"⏭️  English translation already exists, skipping...")
        print(f"   Output: {english_output}\n")
    else:
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


def compute_merged_dialect_scores(
    all_metrics: List[Dict],
    dataset_dir: Path,
    output_dir: Path
) -> List[Dict]:
    """
    Merge test sets by dialect, concatenate files, and recompute scores.
    
    Returns list of merged dialect metrics with type='dialect_merged'.
    """
    # Group metrics by dialect
    dialect_groups = defaultdict(list)
    
    for metrics in all_metrics:
        if 'dialect_code' in metrics and 'dialect_name' in metrics:
            dialect_code = metrics['dialect_code']
            dialect_name = metrics['dialect_name']
            dialect_groups[(dialect_code, dialect_name)].append(metrics)
    
    merged_results = []
    
    for (dialect_code, dialect_name), metrics_list in dialect_groups.items():
        if len(metrics_list) < 1:
            continue
        
        print(f"\n{'='*80}")
        print(f"🔄 Merging {len(metrics_list)} test set(s) for {dialect_name} ({dialect_code})...")
        print(f"{'='*80}\n")
        
        # Collect all file tuples (ref, arabic_general, dialect) - only include if ALL exist
        file_tuples = []
        source_files = []
        
        for m in metrics_list:
            filename = m.get('filename', '')
            if not filename:
                continue
            
            # Get reference file
            ref_name = filename.replace('.en', '.ar')
            ref_file = dataset_dir / ref_name
            
            # Get output files
            base_name = Path(filename).stem.replace('.en', '')
            arabic_output = output_dir / f"{base_name}.ar.general"
            dialect_output = output_dir / f"{base_name}.ar.{dialect_code}"
            
            # Only include if ALL three files exist (to ensure alignment)
            if ref_file.exists() and arabic_output.exists() and dialect_output.exists():
                file_tuples.append((ref_file, arabic_output, dialect_output))
                source_files.append(filename)
            else:
                missing = []
                if not ref_file.exists():
                    missing.append('ref')
                if not arabic_output.exists():
                    missing.append('arabic_general')
                if not dialect_output.exists():
                    missing.append('dialect')
                print(f"   Skipping {filename}: missing {', '.join(missing)}")
        
        # Check if we have files to merge
        if not file_tuples:
            print(f"⚠️  No complete file sets found for {dialect_name}, skipping merge")
            continue
        
        # Warn if some files are missing
        expected_count = len(metrics_list)
        if len(file_tuples) < expected_count:
            print(f"⚠️  Warning: Only {len(file_tuples)}/{expected_count} complete file sets for {dialect_name}")
        
        # Concatenate all files (maintaining alignment)
        print(f"📄 Concatenating {len(file_tuples)} file set(s)...")
        all_ref_lines = []
        all_arabic_general_lines = []
        all_dialect_lines = []
        
        for ref_file, arabic_file, dialect_file in file_tuples:
            ref_lines = ref_file.read_text(encoding="utf-8").strip().split('\n')
            arabic_lines = arabic_file.read_text(encoding="utf-8").strip().split('\n')
            dialect_lines = dialect_file.read_text(encoding="utf-8").strip().split('\n')
            
            # Ensure same length within each file set
            file_min_len = min(len(ref_lines), len(arabic_lines), len(dialect_lines))
            all_ref_lines.extend(ref_lines[:file_min_len])
            all_arabic_general_lines.extend(arabic_lines[:file_min_len])
            all_dialect_lines.extend(dialect_lines[:file_min_len])
        
        min_len = len(all_ref_lines)  # Should all be same length now
        
        print(f"📊 Total sentences after concatenation: {min_len}")
        
        # Compute metrics on concatenated files
        print(f"📊 Computing metrics for merged {dialect_name}...")
        arabic_metrics = compute_metrics(all_arabic_general_lines, all_ref_lines)
        dialect_metrics = compute_metrics(all_dialect_lines, all_ref_lines)
        
        print(f"   Arabic (general) - BLEU: {arabic_metrics['BLEU']:.4f}, CHRF: {arabic_metrics['CHRF']:.4f}")
        print(f"   {dialect_name} - BLEU: {dialect_metrics['BLEU']:.4f}, CHRF: {dialect_metrics['CHRF']:.4f}")
        
        # Create merged result entry
        merged_result = {
            'type': 'dialect_merged',
            'dialect_code': dialect_code,
            'dialect_name': dialect_name,
            'num_test_sets': len(metrics_list),
            'source_files': source_files,
            'num_sentences': min_len,
            'arabic_general': arabic_metrics,
            'dialect': dialect_metrics
        }
        
        merged_results.append(merged_result)
    
    return merged_results


def compute_overall_scores(
    all_metrics: List[Dict],
    dataset_dir: Path,
    output_dir: Path,
    roundtrip_output_dir: Optional[Path] = None,
    direction: str = 'forward'
) -> Optional[Dict]:
    """
    Compute overall BLEU and CHRF scores by concatenating all test sets.
    
    Args:
        all_metrics: List of all file metrics
        dataset_dir: Path to dataset directory
        output_dir: Path to output directory
        roundtrip_output_dir: Path to roundtrip output directory (for roundtrip direction)
        direction: 'forward', 'reverse', or 'roundtrip'
    
    Returns:
        Dictionary with overall scores, or None if no files found
    """
    if not all_metrics:
        return None
    
    print(f"\n{'='*80}")
    print(f"🔄 Computing overall scores (all test sets concatenated)...")
    print(f"{'='*80}\n")
    
    if direction == 'forward':
        # Forward: concatenate all reference, arabic_general, and dialect files
        all_ref_lines = []
        all_arabic_general_lines = []
        all_dialect_lines = []
        source_files = []
        
        for m in all_metrics:
            filename = m.get('filename', '')
            if not filename:
                continue
            
            # Get reference file
            ref_name = filename.replace('.en', '.ar')
            ref_file = dataset_dir / ref_name
            
            # Get dialect code
            dialect_code = m.get('dialect_code', '')
            if not dialect_code:
                continue
            
            # Get output files
            base_name = Path(filename).stem.replace('.en', '')
            arabic_output = output_dir / f"{base_name}.ar.general"
            dialect_output = output_dir / f"{base_name}.ar.{dialect_code}"
            
            # Only include if ALL three files exist
            if ref_file.exists() and arabic_output.exists() and dialect_output.exists():
                ref_lines = ref_file.read_text(encoding="utf-8").strip().split('\n')
                arabic_lines = arabic_output.read_text(encoding="utf-8").strip().split('\n')
                dialect_lines = dialect_output.read_text(encoding="utf-8").strip().split('\n')
                
                # Ensure same length within each file set
                file_min_len = min(len(ref_lines), len(arabic_lines), len(dialect_lines))
                all_ref_lines.extend(ref_lines[:file_min_len])
                all_arabic_general_lines.extend(arabic_lines[:file_min_len])
                all_dialect_lines.extend(dialect_lines[:file_min_len])
                source_files.append(filename)
        
        if not all_ref_lines:
            print(f"⚠️  No complete file sets found for overall scores")
            return None
        
        min_len = len(all_ref_lines)
        print(f"📊 Total sentences after concatenation: {min_len}")
        print(f"📄 Total test sets: {len(source_files)}")
        
        # Compute metrics on concatenated files
        print(f"📊 Computing overall metrics...")
        arabic_metrics = compute_metrics(all_arabic_general_lines, all_ref_lines)
        dialect_metrics = compute_metrics(all_dialect_lines, all_ref_lines)
        
        print(f"   Arabic (general) - BLEU: {arabic_metrics['BLEU']:.4f}, CHRF: {arabic_metrics['CHRF']:.4f}")
        print(f"   Dialect (all) - BLEU: {dialect_metrics['BLEU']:.4f}, CHRF: {dialect_metrics['CHRF']:.4f}")
        
        return {
            'type': 'overall',
            'num_test_sets': len(source_files),
            'num_sentences': min_len,
            'arabic_general': arabic_metrics,
            'dialect': dialect_metrics
        }
    
    elif direction == 'reverse':
        # Reverse: concatenate all reference and translated files
        all_ref_lines = []
        all_hyp_lines = []
        source_files = []
        
        for m in all_metrics:
            filename = m.get('filename', '')
            if not filename:
                continue
            
            # Get reference file (English)
            ref_name = filename.replace('.ar', '.en')
            ref_file = dataset_dir / ref_name
            
            # Get output file
            base_name = Path(filename).stem.replace('.ar', '')
            hyp_output = output_dir / f"{base_name}.en.translated"
            
            # Only include if both files exist
            if ref_file.exists() and hyp_output.exists():
                ref_lines = ref_file.read_text(encoding="utf-8").strip().split('\n')
                hyp_lines = hyp_output.read_text(encoding="utf-8").strip().split('\n')
                
                # Ensure same length
                file_min_len = min(len(ref_lines), len(hyp_lines))
                all_ref_lines.extend(ref_lines[:file_min_len])
                all_hyp_lines.extend(hyp_lines[:file_min_len])
                source_files.append(filename)
        
        if not all_ref_lines:
            print(f"⚠️  No complete file sets found for overall scores")
            return None
        
        min_len = len(all_ref_lines)
        print(f"📊 Total sentences after concatenation: {min_len}")
        print(f"📄 Total test sets: {len(source_files)}")
        
        # Compute metrics on concatenated files
        print(f"📊 Computing overall metrics...")
        overall_metrics = compute_metrics(all_hyp_lines, all_ref_lines)
        
        print(f"   Overall - BLEU: {overall_metrics['BLEU']:.4f}, CHRF: {overall_metrics['CHRF']:.4f}")
        
        return {
            'type': 'overall',
            'num_test_sets': len(source_files),
            'num_sentences': min_len,
            'BLEU': overall_metrics['BLEU'],
            'CHRF': overall_metrics['CHRF']
        }
    
    elif direction == 'roundtrip':
        # Roundtrip: concatenate all source and roundtrip files
        if not roundtrip_output_dir:
            print(f"⚠️  Roundtrip output directory not provided")
            return None
        
        all_source_lines = []
        all_roundtrip_lines = []
        source_files = []
        
        for m in all_metrics:
            filename = m.get('filename', '')
            if not filename:
                continue
            
            # Source file is the original English file
            source_file = dataset_dir / filename
            
            # Get roundtrip output file
            base_name = Path(filename).stem.replace('.en', '')
            roundtrip_output = roundtrip_output_dir / f"{base_name}.en.roundtrip"
            
            # Only include if both files exist
            if source_file.exists() and roundtrip_output.exists():
                source_lines = source_file.read_text(encoding="utf-8").strip().split('\n')
                roundtrip_lines = roundtrip_output.read_text(encoding="utf-8").strip().split('\n')
                
                # Ensure same length
                file_min_len = min(len(source_lines), len(roundtrip_lines))
                all_source_lines.extend(source_lines[:file_min_len])
                all_roundtrip_lines.extend(roundtrip_lines[:file_min_len])
                source_files.append(filename)
        
        if not all_source_lines:
            print(f"⚠️  No complete file sets found for overall scores")
            return None
        
        min_len = len(all_source_lines)
        print(f"📊 Total sentences after concatenation: {min_len}")
        print(f"📄 Total test sets: {len(source_files)}")
        
        # Compute metrics on concatenated files
        print(f"📊 Computing overall metrics...")
        overall_metrics = compute_metrics(all_roundtrip_lines, all_source_lines)
        
        print(f"   Overall - BLEU: {overall_metrics['BLEU']:.4f}, CHRF: {overall_metrics['CHRF']:.4f}")
        
        return {
            'type': 'overall',
            'num_test_sets': len(source_files),
            'num_sentences': min_len,
            'roundtrip': overall_metrics
        }
    
    return None


async def process_all_files(
    dataset_dir: Path,
    output_dir: Path,
    model: str = MODEL,
    base_url: str = "https://api.openai.com/v1",
    reverse: bool = False,
    roundtrip: bool = False,
    roundtrip_output_dir: Optional[Path] = None
):
    """
    Process all files in the AraBench dataset.
    
    Args:
        reverse: If True, translate Arabic to English. If False, translate English to Arabic.
        roundtrip: If True, do round-trip translation (English -> Arabic dialect -> English).
        roundtrip_output_dir: Output directory for round-trip translations (required if roundtrip=True).
    """
    # Initialize client
    client = AsyncOpenAI(base_url=base_url)
    
    # Store all metrics
    all_metrics = []
    total_files = 0
    
    if roundtrip:
        # Round-trip translation: English -> Arabic (dialect) -> English
        if not roundtrip_output_dir:
            print("❌ Round-trip output directory is required for round-trip translation")
            return
        
        roundtrip_output_dir.mkdir(parents=True, exist_ok=True)
        
    # Find all .en files (excluding .ids files)
    en_files = [f for f in dataset_dir.glob("*.en") if not f.name.endswith(".ids")]
    
    if not en_files:
        print(f"❌ No .en files found in {dataset_dir}")
        return
    
        total_files = len(en_files)
        print(f"📊 Found {total_files} English files for round-trip translation\n")
        print(f"   Forward output: {output_dir}")
        print(f"   Round-trip output: {roundtrip_output_dir}\n")
        
        # Process files sequentially
        for i, en_file in enumerate(en_files, 1):
            print(f"\n[{i}/{total_files}] Processing {en_file.name}...")
            metrics = await translate_arabench_file_roundtrip(
                client, model, en_file, output_dir, roundtrip_output_dir, dataset_dir, base_url
            )
            if metrics:
                metrics['type'] = 'file'
                all_metrics.append(metrics)
    
        # No merged dialect scores for round-trip
        merged_dialect_results = []
        direction_str = 'roundtrip'
    
    elif reverse:
        # Find all .ar files (excluding .ids files, .general files, and dialect-specific files)
        ar_files = [
            f for f in dataset_dir.glob("*.ar")
            if not f.name.endswith(".ids")
            and '.general' not in f.name
            and not any(f.name.endswith(f'.ar.{code}') for code in ['ma', 'tn', 'ms', 'sy', 'eg', 'jo', 'ps', 'lv', 'qa', 'lb', 'iq', 'om', 'sa', 'ye', 'pa', 'dz', 'ly', 'sd'])
        ]
        
        if not ar_files:
            print(f"❌ No .ar files found in {dataset_dir}")
            return
        
        total_files = len(ar_files)
        print(f"📊 Found {total_files} Arabic files to translate (reverse: Arabic -> English)\n")
        
        # Process files sequentially (to avoid overwhelming the API)
        for i, ar_file in enumerate(ar_files, 1):
            print(f"\n[{i}/{total_files}] Processing {ar_file.name}...")
            metrics = await translate_arabench_file_reverse(client, model, ar_file, output_dir, dataset_dir, base_url)
            if metrics:
                # Mark as individual file result
                metrics['type'] = 'file'
                all_metrics.append(metrics)
        
        # For reverse translation, we don't compute merged dialect scores
        # (since we're translating to English, not to dialects)
        merged_dialect_results = []
        direction_str = 'reverse'
    
    else:
        # Find all .en files (excluding .ids files)
        en_files = [f for f in dataset_dir.glob("*.en") if not f.name.endswith(".ids")]
        
        if not en_files:
            print(f"❌ No .en files found in {dataset_dir}")
            return
        
        total_files = len(en_files)
        print(f"📊 Found {total_files} English files to translate (forward: English -> Arabic)\n")
    
    # Process files sequentially (to avoid overwhelming the API)
    for i, en_file in enumerate(en_files, 1):
            print(f"\n[{i}/{total_files}] Processing {en_file.name}...")
        metrics = await translate_arabench_file(client, model, en_file, output_dir, dataset_dir, base_url)
        if metrics:
                # Mark as individual file result
                metrics['type'] = 'file'
            all_metrics.append(metrics)
    
        # Compute merged dialect scores (only for forward translation)
        merged_dialect_results = compute_merged_dialect_scores(all_metrics, dataset_dir, output_dir)
        direction_str = 'forward'
    
    # Compute overall scores by concatenating all test sets
    overall_result = compute_overall_scores(
        all_metrics, dataset_dir, output_dir, roundtrip_output_dir, direction_str
    )
    
    # Combine all results (individual files + merged dialects + overall)
    all_results = all_metrics + merged_dialect_results
    if overall_result:
        all_results.append(overall_result)
    
    # Determine output dir for scores (use roundtrip_output_dir for roundtrip)
    scores_output_dir = roundtrip_output_dir if roundtrip else output_dir
    scores_file = scores_output_dir / "scores.json"
    
    scores_summary = {
        'model': model,
        'direction': direction_str,
        'total_files': total_files,
        'files_with_metrics': len(all_metrics),
        'merged_dialects': len(merged_dialect_results),
        'has_overall_scores': overall_result is not None,
        'results': all_results
    }
    
    # Compute averages
    if all_metrics:
        if roundtrip:
            # For round-trip translation, we have roundtrip scores
            bleu_scores = [m['roundtrip']['BLEU'] for m in all_metrics if 'roundtrip' in m]
            chrf_scores = [m['roundtrip']['CHRF'] for m in all_metrics if 'roundtrip' in m]
            
            scores_summary['averages'] = {
                'BLEU': round(sum(bleu_scores) / len(bleu_scores), 4) if bleu_scores else 0,
                'CHRF': round(sum(chrf_scores) / len(chrf_scores), 4) if chrf_scores else 0,
            }
        elif reverse:
            # For reverse translation, we only have BLEU and CHRF directly
            bleu_scores = [m['BLEU'] for m in all_metrics if 'BLEU' in m]
            chrf_scores = [m['CHRF'] for m in all_metrics if 'CHRF' in m]
            
            scores_summary['averages'] = {
                'BLEU': round(sum(bleu_scores) / len(bleu_scores), 4) if bleu_scores else 0,
                'CHRF': round(sum(chrf_scores) / len(chrf_scores), 4) if chrf_scores else 0,
            }
        else:
            # For forward translation, we have arabic_general and dialect
        arabic_bleu_scores = [m['arabic_general']['BLEU'] for m in all_metrics if 'arabic_general' in m]
        arabic_chrf_scores = [m['arabic_general']['CHRF'] for m in all_metrics if 'arabic_general' in m]
        dialect_bleu_scores = [m['dialect']['BLEU'] for m in all_metrics if 'dialect' in m]
        dialect_chrf_scores = [m['dialect']['CHRF'] for m in all_metrics if 'dialect' in m]
        
        scores_summary['averages'] = {
            'arabic_general': {
                'BLEU': round(sum(arabic_bleu_scores) / len(arabic_bleu_scores), 4) if arabic_bleu_scores else 0,
                'CHRF': round(sum(arabic_chrf_scores) / len(arabic_chrf_scores), 4) if arabic_chrf_scores else 0,
            },
            'dialect': {
                'BLEU': round(sum(dialect_bleu_scores) / len(dialect_bleu_scores), 4) if dialect_bleu_scores else 0,
                'CHRF': round(sum(dialect_chrf_scores) / len(dialect_chrf_scores), 4) if dialect_chrf_scores else 0,
            }
        }
    
    # Save JSON
    scores_file.write_text(json.dumps(scores_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Also save a human-readable summary
    summary_file = scores_output_dir / "scores_summary.txt"
    if roundtrip:
        direction_label = "English -> Arabic (dialect) -> English (round-trip)"
    elif reverse:
        direction_label = "Arabic -> English"
    else:
        direction_label = "English -> Arabic"
    with summary_file.open('w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"AraBench Evaluation Results ({direction_label})\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model}\n")
        f.write(f"Direction: {direction_label}\n")
        f.write(f"Total files processed: {total_files}\n")
        f.write(f"Files with metrics: {len(all_metrics)}\n\n")
        
        if all_metrics and 'averages' in scores_summary:
            f.write("Average Scores (per-file averages):\n")
            f.write("-" * 80 + "\n")
            if roundtrip or reverse:
                f.write(f"  BLEU: {scores_summary['averages']['BLEU']:.4f}\n")
                f.write(f"  CHRF: {scores_summary['averages']['CHRF']:.4f}\n\n")
            else:
            f.write(f"Arabic (general):\n")
            f.write(f"  BLEU: {scores_summary['averages']['arabic_general']['BLEU']:.4f}\n")
            f.write(f"  CHRF: {scores_summary['averages']['arabic_general']['CHRF']:.4f}\n\n")
            f.write(f"Dialect-specific:\n")
            f.write(f"  BLEU: {scores_summary['averages']['dialect']['BLEU']:.4f}\n")
            f.write(f"  CHRF: {scores_summary['averages']['dialect']['CHRF']:.4f}\n\n")
        
        # Write overall scores if available
        if overall_result:
            f.write("=" * 80 + "\n")
            f.write("Overall Scores (all test sets concatenated):\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total test sets: {overall_result.get('num_test_sets', 0)}\n")
            f.write(f"Total sentences: {overall_result.get('num_sentences', 0)}\n\n")
            if direction_str == 'forward':
                f.write(f"Arabic (general):\n")
                f.write(f"  BLEU: {overall_result['arabic_general']['BLEU']:.4f}\n")
                f.write(f"  CHRF: {overall_result['arabic_general']['CHRF']:.4f}\n\n")
                f.write(f"Dialect (all):\n")
                f.write(f"  BLEU: {overall_result['dialect']['BLEU']:.4f}\n")
                f.write(f"  CHRF: {overall_result['dialect']['CHRF']:.4f}\n\n")
            elif direction_str == 'reverse':
                f.write(f"  BLEU: {overall_result['BLEU']:.4f}\n")
                f.write(f"  CHRF: {overall_result['CHRF']:.4f}\n\n")
            elif direction_str == 'roundtrip':
                f.write(f"  BLEU: {overall_result['roundtrip']['BLEU']:.4f}\n")
                f.write(f"  CHRF: {overall_result['roundtrip']['CHRF']:.4f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Per-file Results:\n")
        f.write("=" * 80 + "\n\n")
        
        for metrics in all_metrics:
            f.write(f"File: {metrics.get('filename', 'unknown')}\n")
            if not reverse:
            f.write(f"Dialect: {metrics.get('dialect_name', 'unknown')} ({metrics.get('dialect_code', 'unknown')})\n")
            f.write(f"Sentences: {metrics.get('num_sentences', 0)}\n")
            if reverse:
                if 'BLEU' in metrics:
                    f.write(f"  BLEU: {metrics['BLEU']:.4f}, CHRF: {metrics['CHRF']:.4f}\n")
            else:
            if 'arabic_general' in metrics:
                f.write(f"  Arabic (general) - BLEU: {metrics['arabic_general']['BLEU']:.4f}, CHRF: {metrics['arabic_general']['CHRF']:.4f}\n")
            if 'dialect' in metrics:
                f.write(f"  Dialect - BLEU: {metrics['dialect']['BLEU']:.4f}, CHRF: {metrics['dialect']['CHRF']:.4f}\n")
            f.write("\n")
    
    print(f"\n{'='*80}")
    print(f"✅ All translations completed!")
    print(f"📁 Output directory: {scores_output_dir}")
    print(f"📊 Scores saved to: {scores_file}")
    print(f"📄 Summary saved to: {summary_file}")
    if all_metrics and 'averages' in scores_summary:
        print(f"\n📈 Average Scores (per-file averages):")
        if roundtrip or reverse:
            print(f"   BLEU: {scores_summary['averages']['BLEU']:.4f}, CHRF: {scores_summary['averages']['CHRF']:.4f}")
        else:
        print(f"   Arabic (general) - BLEU: {scores_summary['averages']['arabic_general']['BLEU']:.4f}, CHRF: {scores_summary['averages']['arabic_general']['CHRF']:.4f}")
        print(f"   Dialect-specific - BLEU: {scores_summary['averages']['dialect']['BLEU']:.4f}, CHRF: {scores_summary['averages']['dialect']['CHRF']:.4f}")
    if overall_result:
        print(f"\n📈 Overall Scores (all test sets concatenated):")
        print(f"   Total test sets: {overall_result.get('num_test_sets', 0)}")
        print(f"   Total sentences: {overall_result.get('num_sentences', 0)}")
        if direction_str == 'forward':
            print(f"   Arabic (general) - BLEU: {overall_result['arabic_general']['BLEU']:.4f}, CHRF: {overall_result['arabic_general']['CHRF']:.4f}")
            print(f"   Dialect (all) - BLEU: {overall_result['dialect']['BLEU']:.4f}, CHRF: {overall_result['dialect']['CHRF']:.4f}")
        elif direction_str == 'reverse':
            print(f"   BLEU: {overall_result['BLEU']:.4f}, CHRF: {overall_result['CHRF']:.4f}")
        elif direction_str == 'roundtrip':
            print(f"   BLEU: {overall_result['roundtrip']['BLEU']:.4f}, CHRF: {overall_result['roundtrip']['CHRF']:.4f}")
    print(f"{'='*80}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate AraBench dataset by translating English files to Arabic and dialects"
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
        default=Path("arabench_translations"),
        help="Path to output directory for translations (default: arabench_translations)"
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
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Enable reverse translation (Arabic -> English) instead of forward (English -> Arabic)"
    )
    parser.add_argument(
        "--roundtrip",
        action="store_true",
        help="Enable round-trip translation (English -> Arabic dialect -> English)"
    )
    parser.add_argument(
        "--roundtrip-output-dir",
        type=Path,
        default=None,
        help="Output directory for round-trip translations (required with --roundtrip)"
    )
    
    args = parser.parse_args()
    
    # Validate roundtrip options
    if args.roundtrip and not args.roundtrip_output_dir:
        print("❌ --roundtrip-output-dir is required when using --roundtrip")
        sys.exit(1)
    
    if args.roundtrip and args.reverse:
        print("❌ Cannot use --roundtrip and --reverse together")
        sys.exit(1)
    
    # Validate dataset directory
    if not args.dataset_dir.exists():
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        sys.exit(1)
    
    # Run async processing
    asyncio.run(process_all_files(
        args.dataset_dir,
        args.output_dir,
        args.model,
        args.base_url,
        args.reverse,
        args.roundtrip,
        args.roundtrip_output_dir
    ))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Compute COMET scores for all translations.

This script:
1. Finds all translation directories (by looking for scores.json files)
2. For each translation file, finds the corresponding reference file
3. Computes COMET scores using the unbabel-comet library
4. Updates the scores.json file with COMET scores
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import numpy as np

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    print("❌ Error: unbabel-comet is not installed.")
    print("   Please install it with: pip install unbabel-comet")
    exit(1)


def find_translation_directories(root_dir: Path) -> List[Path]:
    """Find all directories containing translation files (by looking for scores.json)."""
    translation_dirs = []
    
    # Look for scores.json files to identify model output directories
    for scores_file in root_dir.rglob("scores.json"):
        translation_dir = scores_file.parent
        if translation_dir not in translation_dirs:
            translation_dirs.append(translation_dir)
    
    return sorted(translation_dirs)


def find_reference_file(translation_file: Path, dataset_dir: Path, file_type: str, direction: str) -> Optional[Path]:
    """
    Find the reference file corresponding to a translation file.
    
    Args:
        translation_file: Path to the translation file
        dataset_dir: Path to the dataset directory
        file_type: Type of translation ('arabic_general', 'dialect', 'reverse', 'roundtrip')
        direction: Translation direction ('forward', 'reverse', 'roundtrip')
    """
    filename = translation_file.name
    
    if direction == 'forward':
        if file_type == 'arabic_general':
            # Forward Arabic general: reference is .ar file
            # Example: madar.test.glf.0.qa.ar.general -> madar.test.glf.0.qa.ar
            ref_name = filename.replace('.ar.general', '.ar')
        elif file_type == 'dialect':
            # Forward Dialect: reference is .ar file
            # Example: madar.test.glf.0.qa.ar.qa -> madar.test.glf.0.qa.ar
            ref_name = re.sub(r'\.ar\.[a-z]{2}$', '.ar', filename)
        else:
            return None
        ref_file = dataset_dir / ref_name
        
    elif direction == 'reverse':
        # Reverse: reference is .en file
        # Example: madar.test.glf.0.qa.en.translated -> madar.test.glf.0.qa.en
        ref_name = filename.replace('.en.translated', '.en')
        ref_file = dataset_dir / ref_name
        
    elif direction == 'roundtrip':
        # Roundtrip: reference is .en file (original English)
        # Example: madar.test.glf.0.qa.en.roundtrip -> madar.test.glf.0.qa.en
        ref_name = filename.replace('.en.roundtrip', '.en')
        ref_file = dataset_dir / ref_name
    else:
        return None
    
    if ref_file.exists():
        return ref_file
    
    return None


def get_comet_key(model_name: str, reference_less: bool) -> str:
    """
    Generate a COMET key name based on model name and reference mode.
    
    Args:
        model_name: COMET model name (e.g., 'Unbabel/wmt22-comet-da')
        reference_less: Whether reference-less mode is used
    
    Returns:
        COMET key name (e.g., 'COMET_wmt22-comet-da' or 'COMET_wmt22-comet-da_ref-less')
    """
    # Clean model name (remove path prefixes, slashes, etc.)
    clean_name = model_name.replace('/', '_').replace('\\', '_')
    if '/' in clean_name:
        clean_name = clean_name.split('/')[-1]
    
    if reference_less:
        return f'COMET_{clean_name}_ref-less'
    else:
        return f'COMET_{clean_name}'


def compute_comet_averages(scores_data: Dict, comet_key: str, direction: str):
    """
    Compute average COMET scores across all files for the given COMET variant.
    
    Args:
        scores_data: The scores.json data structure
        comet_key: The COMET key name (e.g., 'COMET_wmt22-comet-da' or 'COMET_wmt22-comet-da_ref-less')
        direction: Translation direction ('forward', 'reverse', 'roundtrip')
    """
    if 'results' not in scores_data:
        return
    
    # Initialize averages structure if not present
    if 'averages' not in scores_data:
        scores_data['averages'] = {}
    
    if direction == 'forward':
        # For forward direction, compute averages for arabic_general and dialect separately
        arabic_scores = []
        dialect_scores = []
        
        for result in scores_data['results']:
            if result.get('type') == 'file':
                if 'arabic_general' in result and comet_key in result['arabic_general']:
                    arabic_scores.append(result['arabic_general'][comet_key])
                if 'dialect' in result and comet_key in result['dialect']:
                    dialect_scores.append(result['dialect'][comet_key])
        
        # Update averages structure
        if 'arabic_general' not in scores_data['averages']:
            scores_data['averages']['arabic_general'] = {}
        if 'dialect' not in scores_data['averages']:
            scores_data['averages']['dialect'] = {}
        
        if arabic_scores:
            scores_data['averages']['arabic_general'][comet_key] = round(sum(arabic_scores) / len(arabic_scores), 4)
            print(f"   📊 Average {comet_key} (arabic_general): {scores_data['averages']['arabic_general'][comet_key]:.4f} (based on {len(arabic_scores)} files)")
        
        if dialect_scores:
            scores_data['averages']['dialect'][comet_key] = round(sum(dialect_scores) / len(dialect_scores), 4)
            print(f"   📊 Average {comet_key} (dialect): {scores_data['averages']['dialect'][comet_key]:.4f} (based on {len(dialect_scores)} files)")
    
    elif direction in ['reverse', 'roundtrip']:
        # For reverse/roundtrip, compute single average
        scores = []
        for result in scores_data['results']:
            if result.get('type') == 'file':
                if comet_key in result:
                    scores.append(result[comet_key])
        
        if scores:
            scores_data['averages'][comet_key] = round(sum(scores) / len(scores), 4)
            print(f"   📊 Average {comet_key}: {scores_data['averages'][comet_key]:.4f} (based on {len(scores)} files)")


def compute_dialect_comet_scores(scores_data: Dict, segment_scores_dict: Dict, comet_key: str):
    """
    Compute dialect-level COMET scores by averaging segment-level scores for all files in each dialect.
    
    Args:
        scores_data: The scores.json data structure
        segment_scores_dict: Dictionary mapping (filename, file_type) to segment-level scores
        comet_key: The COMET key name to use (e.g., 'COMET_wmt22-comet-da' or 'COMET_wmt22-comet-da_ref-less')
    """
    if 'results' not in scores_data:
        return
    
    # Compute average COMET score for each dialect and update merged dialect entries
    for result in scores_data['results']:
        if result.get('type') == 'dialect_merged':
            dialect_code = result.get('dialect_code')
            dialect_name = result.get('dialect_name', dialect_code)
            source_files = result.get('source_files', [])
            
            if not source_files:
                continue
            
            # Collect segment scores only from files in this merged dialect result
            arabic_scores = []
            dialect_scores = []
            
            for filename in source_files:
                # Collect arabic_general scores (keys are now (filename, file_type, comet_key))
                key_arabic = (filename, 'arabic_general', comet_key)
                if key_arabic in segment_scores_dict:
                    arabic_scores.extend(segment_scores_dict[key_arabic])
                
                # Collect dialect scores
                key_dialect = (filename, 'dialect', comet_key)
                if key_dialect in segment_scores_dict:
                    dialect_scores.extend(segment_scores_dict[key_dialect])
            
            # Compute arabic_general COMET
            if arabic_scores:
                avg_comet = sum(arabic_scores) / len(arabic_scores)
                if 'arabic_general' not in result:
                    result['arabic_general'] = {}
                result['arabic_general'][comet_key] = round(avg_comet, 4)
                print(f"   📊 Computed arabic_general {comet_key} for {dialect_name}: {avg_comet:.4f} ({len(arabic_scores)} segments)")
            
            # Compute dialect COMET
            if dialect_scores:
                avg_comet = sum(dialect_scores) / len(dialect_scores)
                if 'dialect' not in result:
                    result['dialect'] = {}
                result['dialect'][comet_key] = round(avg_comet, 4)
                print(f"   📊 Computed dialect {comet_key} for {dialect_name}: {avg_comet:.4f} ({len(dialect_scores)} segments)")


def find_translation_file(translation_dir: Path, filename: str, file_type: str, direction: str) -> Optional[Path]:
    """
    Find the translation file corresponding to a result entry.
    
    Args:
        translation_dir: Directory containing translation files
        filename: Original filename from scores.json (e.g., "madar.test.glf.0.qa.en")
        file_type: Type of translation ('arabic_general', 'dialect', 'reverse', 'roundtrip')
        direction: Translation direction ('forward', 'reverse', 'roundtrip')
    """
    if direction == 'forward':
        if file_type == 'arabic_general':
            # Example: madar.test.glf.0.qa.en -> madar.test.glf.0.qa.ar.general
            trans_name = filename.replace('.en', '.ar.general')
        elif file_type == 'dialect':
            # Extract dialect code from filename
            # Example: madar.test.glf.0.qa.en -> madar.test.glf.0.qa.ar.qa
            match = re.search(r'\.([a-z]{2})\.en$', filename)
            if match:
                dialect_code = match.group(1)
                trans_name = filename.replace('.en', f'.ar.{dialect_code}')
            else:
                return None
        else:
            return None
        trans_file = translation_dir / trans_name
        
    elif direction == 'reverse':
        # For reverse, filename is .ar, translation is .en.translated
        # Example: madar.test.glf.0.qa.ar -> madar.test.glf.0.qa.en.translated
        trans_name = filename.replace('.ar', '.en.translated')
        trans_file = translation_dir / trans_name
        
    elif direction == 'roundtrip':
        # For roundtrip, filename is .en, translation is .en.roundtrip
        # Example: madar.test.glf.0.qa.en -> madar.test.glf.0.qa.en.roundtrip
        trans_name = filename.replace('.en', '.en.roundtrip')
        trans_file = translation_dir / trans_name
    else:
        return None
    
    if trans_file.exists():
        return trans_file
    
    return None


def compute_comet_score(model, sources: List[str], hypotheses: List[str], references: List[str], reference_less: bool = False) -> Tuple[float, List[float]]:
    """
    Compute COMET score for a batch of translations.
    
    Args:
        model: Loaded COMET model
        sources: Source sentences (for source-aware metrics, can be same as references for reference-based)
        hypotheses: Translated sentences
        references: Reference sentences (ignored if reference_less=True)
        reference_less: If True, use reference-less COMET (don't pass references)
    
    Returns:
        Tuple of (average COMET score, list of segment-level scores)
    """
    # COMET can work with or without references
    if reference_less:
        # Reference-less COMET: only use source and hypothesis
        data = [
            {
                "src": src,
                "mt": hyp
            }
            for src, hyp in zip(sources, hypotheses)
        ]
    else:
        # Reference-based COMET: include references
        data = [
            {
                "src": src,
                "mt": hyp,
                "ref": ref
            }
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]
    
    # model.predict() returns scores - handle different return formats
    result = model.predict(data, batch_size=8, gpus=1)
    
    # Handle different return formats
    if isinstance(result, tuple):
        # If it's a tuple, the first element should be scores
        scores = result[0]
    elif isinstance(result, dict):
        # If it's a dict, look for 'scores' key
        scores = result.get('scores', result)
    else:
        scores = result
    
    # Debug: check what type we got
    if isinstance(scores, str):
        raise ValueError(f"Unexpected string value for scores: {scores}. Result type: {type(result)}, Result: {result}")
    
    # Convert to numpy array if needed
    if not isinstance(scores, np.ndarray):
        try:
            scores = np.array(scores)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not convert scores to numpy array. Type: {type(scores)}, Value: {scores}, Error: {e}")
    
    # Ensure it's 1D array of numeric values
    if scores.ndim == 0:
        scores = np.array([scores])
    elif scores.ndim > 1:
        scores = scores.flatten()
    
    # Convert numpy array to list of floats
    segment_scores = scores.tolist()
    # Ensure all values are floats
    try:
        segment_scores = [float(score) for score in segment_scores]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not convert scores to floats. Scores type: {type(segment_scores)}, First few values: {segment_scores[:5] if len(segment_scores) > 0 else 'empty'}, Error: {e}")
    
    return float(scores.mean()), segment_scores


def process_scores_file(scores_file: Path, dataset_dir: Path, comet_model, model_name: str, skip_existing: bool = False, reference_less: bool = False):
    """
    Process a scores.json file and add COMET scores.
    
    Args:
        scores_file: Path to scores.json file
        dataset_dir: Path to dataset directory
        comet_model: Loaded COMET model
        model_name: COMET model name (for key generation)
        skip_existing: If True, skip files that already have COMET scores
        reference_less: If True, use reference-less COMET (don't pass references)
    
    Returns:
        Dictionary mapping (filename, file_type) to segment-level scores
    """
    translation_dir = scores_file.parent
    
    print(f"\n{'='*80}")
    print(f"📂 Processing: {translation_dir.name}")
    print(f"   Directory: {translation_dir}")
    print(f"{'='*80}\n")
    
    # Load scores.json
    try:
        with open(scores_file, 'r', encoding='utf-8') as f:
            scores_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading {scores_file}: {e}")
        return {}
    
    direction = scores_data.get('direction', 'forward')
    translation_model_name = scores_data.get('model', 'unknown')
    
    # Generate COMET key based on model name and reference mode
    comet_key = get_comet_key(model_name, reference_less)
    
    print(f"   Translation Model: {translation_model_name}")
    print(f"   COMET Model: {model_name} ({'reference-less' if reference_less else 'reference-based'})")
    print(f"   COMET Key: {comet_key}")
    print(f"   Direction: {direction}")
    
    if 'results' not in scores_data:
        print(f"⚠️  No results found in scores.json")
        return {}
    
    updated = False
    total_files = 0
    processed_files = 0
    
    # Dictionary to store segment-level scores: {(filename, file_type): [scores]}
    segment_scores_dict = {}
    
    # Process each result entry
    for result in scores_data['results']:
        result_type = result.get('type', 'file')
        
        # Skip merged dialects and overall scores for now (can be added later)
        if result_type != 'file':
            continue
        
        filename = result.get('filename', '')
        if not filename:
            continue
        
        total_files += 1
        
        # For forward direction, process both arabic_general and dialect
        if direction == 'forward':
            for file_type in ['arabic_general', 'dialect']:
                if file_type not in result:
                    continue
                
                # Check if COMET already exists
                if skip_existing and comet_key in result[file_type]:
                    print(f"   ⏭️  Skipping {filename} ({file_type}) - {comet_key} already exists")
                    continue
                
                # Find translation file
                trans_file = find_translation_file(translation_dir, filename, file_type, direction)
                if not trans_file:
                    print(f"   ⚠️  Translation file not found for {filename} ({file_type})")
                    continue
                
                # Find reference file
                ref_file = find_reference_file(trans_file, dataset_dir, file_type, direction)
                if not ref_file:
                    print(f"   ⚠️  Reference file not found for {filename} ({file_type})")
                    continue
                
                # Read files
                try:
                    trans_lines = trans_file.read_text(encoding='utf-8').strip().split('\n')
                    ref_lines = ref_file.read_text(encoding='utf-8').strip().split('\n')
                    # For forward, source is the English file
                    source_file = dataset_dir / filename
                    if source_file.exists():
                        source_lines = source_file.read_text(encoding='utf-8').strip().split('\n')
                    else:
                        source_lines = ref_lines  # Fallback to references
                except Exception as e:
                    print(f"   ❌ Error reading files for {filename} ({file_type}): {e}")
                    continue
                
                # Ensure same length
                min_len = min(len(trans_lines), len(ref_lines), len(source_lines))
                trans_lines = trans_lines[:min_len]
                ref_lines = ref_lines[:min_len]
                source_lines = source_lines[:min_len]
                
                if min_len == 0:
                    print(f"   ⚠️  Empty files for {filename} ({file_type})")
                    continue
                
                # Compute COMET score
                print(f"   📊 Computing {comet_key} for {filename} ({file_type}) - {min_len} sentences...", end=' ', flush=True)
                try:
                    comet_score, segment_scores = compute_comet_score(comet_model, source_lines, trans_lines, ref_lines, reference_less)
                    result[file_type][comet_key] = comet_score
                    # Store segment-level scores with key info
                    segment_scores_dict[(filename, file_type, comet_key)] = segment_scores
                    print(f"✓ {comet_score:.4f}")
                    updated = True
                    processed_files += 1
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        # For reverse and roundtrip directions
        elif direction in ['reverse', 'roundtrip']:
            # Check if COMET already exists
            if skip_existing and comet_key in result:
                print(f"   ⏭️  Skipping {filename} - {comet_key} already exists")
                continue
            
            # Find translation file
            # For reverse/roundtrip, file_type is the same as direction
            trans_file = find_translation_file(translation_dir, filename, direction, direction)
            if not trans_file:
                print(f"   ⚠️  Translation file not found for {filename}")
                continue
            
            # Find reference file
            # For reverse/roundtrip, file_type is the same as direction
            ref_file = find_reference_file(trans_file, dataset_dir, direction, direction)
            if not ref_file:
                print(f"   ⚠️  Reference file not found for {filename}")
                continue
            
            # Read files
            try:
                trans_lines = trans_file.read_text(encoding='utf-8').strip().split('\n')
                ref_lines = ref_file.read_text(encoding='utf-8').strip().split('\n')
                # For reverse/roundtrip, filename is already the source file
                # (reverse: .ar file, roundtrip: .en file)
                source_file = dataset_dir / filename
                if source_file.exists():
                    source_lines = source_file.read_text(encoding='utf-8').strip().split('\n')
                else:
                    source_lines = ref_lines  # Fallback to references
            except Exception as e:
                print(f"   ❌ Error reading files for {filename}: {e}")
                continue
            
            # Ensure same length
            min_len = min(len(trans_lines), len(ref_lines), len(source_lines))
            trans_lines = trans_lines[:min_len]
            ref_lines = ref_lines[:min_len]
            source_lines = source_lines[:min_len]
            
            if min_len == 0:
                print(f"   ⚠️  Empty files for {filename}")
                continue
            
            # Compute COMET score
            print(f"   📊 Computing {comet_key} for {filename} - {min_len} sentences...", end=' ', flush=True)
            try:
                comet_score, segment_scores = compute_comet_score(comet_model, source_lines, trans_lines, ref_lines, reference_less)
                result[comet_key] = comet_score
                # Store segment-level scores (use direction as file_type for reverse/roundtrip)
                segment_scores_dict[(filename, direction, comet_key)] = segment_scores
                print(f"✓ {comet_score:.4f}")
                updated = True
                processed_files += 1
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Save segment-level scores to a separate file
    if segment_scores_dict:
        # Use comet_key in filename to distinguish different COMET variants
        safe_comet_key = comet_key.replace('/', '_').replace('\\', '_')
        segment_scores_file = translation_dir / f"comet_segment_scores_{safe_comet_key}.json"
        try:
            # Convert keys to strings for JSON serialization
            # Keys are now (filename, file_type, comet_key) tuples
            segment_scores_json = {
                f"{filename}|{file_type}": scores
                for (filename, file_type, key), scores in segment_scores_dict.items()
                if key == comet_key  # Only save scores for current COMET variant
            }
            with open(segment_scores_file, 'w', encoding='utf-8') as f:
                json.dump(segment_scores_json, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved segment-level {comet_key} scores to {segment_scores_file.name}")
        except Exception as e:
            print(f"⚠️  Error saving segment-level scores: {e}")
    
    # Compute dialect-level scores for forward direction
    if direction == 'forward' and segment_scores_dict:
        compute_dialect_comet_scores(scores_data, segment_scores_dict, comet_key)
    
    # Compute averages for COMET scores
    compute_comet_averages(scores_data, comet_key, direction)
    
    # Save updated scores.json
    if updated:
        try:
            with open(scores_file, 'w', encoding='utf-8') as f:
                json.dump(scores_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Updated {scores_file} with COMET scores ({processed_files}/{total_files} files processed)")
        except Exception as e:
            print(f"❌ Error saving {scores_file}: {e}")
    else:
        print(f"\nℹ️  No updates needed for {scores_file}")
    
    return segment_scores_dict


def main():
    parser = argparse.ArgumentParser(
        description="Compute COMET scores for all translations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_comet_scores.py --dataset-dir /path/to/arabench --root-dir translations
  python compute_comet_scores.py --dataset-dir /path/to/arabench --root-dir translations --model-name wmt22-comet-da
  python compute_comet_scores.py --dataset-dir /path/to/arabench --root-dir translations --skip-existing
  python compute_comet_scores.py --dataset-dir /path/to/arabench --root-dir translations --reference-less
        """
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True,
        help='Path to the AraBench dataset directory'
    )
    
    parser.add_argument(
        '--root-dir',
        type=str,
        default='translations',
        help='Top-level directory to search for translation directories (containing scores.json files)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='Unbabel/wmt22-comet-da',
        help='COMET model name (default: Unbabel/wmt22-comet-da)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already have COMET scores'
    )
    
    parser.add_argument(
        '--reference-less',
        action='store_true',
        help='Use reference-less COMET (do not pass references to the model)'
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir).resolve()
    root_dir = Path(args.root_dir).resolve()
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory does not exist: {dataset_dir}")
        return
    
    if not root_dir.exists():
        print(f"❌ Root directory does not exist: {root_dir}")
        return
    
    print(f"🔍 Searching for scores.json files in: {root_dir}")
    translation_dirs = find_translation_directories(root_dir)
    
    if not translation_dirs:
        print(f"❌ No translation directories found in {root_dir}")
        return
    
    print(f"📊 Found {len(translation_dirs)} translation directory(ies)")
    
    # Load COMET model
    print(f"\n📦 Loading COMET model: {args.model_name}")
    if args.reference_less:
        print(f"   Mode: Reference-less COMET (references will not be used)")
    else:
        print(f"   Mode: Reference-based COMET (references will be used)")
    try:
        model_path = download_model(args.model_name)
        comet_model = load_from_checkpoint(model_path)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading COMET model: {e}")
        return
    
    # Process each translation directory
    for translation_dir in translation_dirs:
        scores_file = translation_dir / "scores.json"
        if scores_file.exists():
            process_scores_file(scores_file, dataset_dir, comet_model, args.model_name, args.skip_existing, args.reference_less)
    
    print(f"\n{'='*80}")
    print(f"✅ Finished processing all translation directories")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


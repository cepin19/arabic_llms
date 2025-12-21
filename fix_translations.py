#!/usr/bin/env python3
"""
Filter and fix suspicious translations.

This script:
1. Finds all translation files (similar to visualize_scores.py)
2. Checks line by line for suspicious translations
3. Applies fixes (e.g., truncate if >5x longer than source in tokens or characters)
4. Saves fixed translations preserving the original directory structure
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


def find_translation_directories(root_dir: Path) -> List[Path]:
    """Find all directories containing translation files (by looking for scores.json)."""
    translation_dirs = []
    
    # Look for scores.json files to identify model output directories
    for scores_file in root_dir.rglob("scores.json"):
        translation_dir = scores_file.parent
        if translation_dir not in translation_dirs:
            translation_dirs.append(translation_dir)
    
    return sorted(translation_dirs)


def find_translation_files(translation_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all translation files in a directory.
    
    Returns:
        Dict mapping file_type -> list of file paths
        file_type can be: 'forward_arabic', 'forward_dialect', 'reverse', 'roundtrip'
    """
    files = {
        'forward_arabic': [],
        'forward_dialect': [],
        'reverse': [],
        'roundtrip': []
    }
    
    # Forward translations: .ar.general and .ar.{dialect_code}
    for ar_file in translation_dir.glob("*.ar.general"):
        files['forward_arabic'].append(ar_file)
    
    # Dialect translations: .ar.{dialect_code} (but not .ar.general)
    dialect_codes = ['ma', 'tn', 'ms', 'sy', 'eg', 'jo', 'ps', 'lv', 'qa', 'lb', 
                     'iq', 'om', 'sa', 'ye', 'pa', 'dz', 'ly', 'sd']
    for dialect_code in dialect_codes:
        for dialect_file in translation_dir.glob(f"*.ar.{dialect_code}"):
            files['forward_dialect'].append(dialect_file)
    
    # Reverse translations: .en.translated
    for en_file in translation_dir.glob("*.en.translated"):
        files['reverse'].append(en_file)
    
    # Roundtrip translations: .en.roundtrip
    for roundtrip_file in translation_dir.glob("*.en.roundtrip"):
        files['roundtrip'].append(roundtrip_file)
    
    return files


def find_source_file(translation_file: Path, dataset_dir: Path, file_type: str) -> Optional[Path]:
    """
    Find the source file corresponding to a translation file.
    
    Args:
        translation_file: Path to the translation file
        dataset_dir: Path to the dataset directory
        file_type: Type of translation file
    """
    filename = translation_file.name
    
    if file_type == 'forward_arabic':
        # Forward Arabic: source is .en file
        # Example: madar.test.glf.0.qa.ar.general -> madar.test.glf.0.qa.en
        source_name = filename.replace('.ar.general', '.en')
        source_file = dataset_dir / source_name
        
    elif file_type == 'forward_dialect':
        # Forward Dialect: source is .en file
        # Example: madar.test.glf.0.qa.ar.qa -> madar.test.glf.0.qa.en
        # Remove .ar.{dialect_code} and add .en
        source_name = re.sub(r'\.ar\.[a-z]{2}$', '.en', filename)
        source_file = dataset_dir / source_name
        
    elif file_type == 'reverse':
        # Reverse: source is .ar file
        # Example: madar.test.glf.0.qa.en.translated -> madar.test.glf.0.qa.ar
        source_name = filename.replace('.en.translated', '.ar')
        source_file = dataset_dir / source_name
        
    elif file_type == 'roundtrip':
        # Roundtrip: source is .en file (original English)
        # Example: madar.test.glf.0.qa.en.roundtrip -> madar.test.glf.0.qa.en
        source_name = filename.replace('.en.roundtrip', '.en')
        source_file = dataset_dir / source_name
    
    else:
        return None
    
    if source_file.exists():
        return source_file
    
    return None


def count_tokens(text: str) -> int:
    """Count tokens (words) in text."""
    return len(text.split())


def fix_translation_line(source_line: str, translation_line: str, 
                        max_ratio: float = 5.0) -> str:
    """
    Fix a suspicious translation line.
    
    Args:
        source_line: Source text
        translation_line: Translation text
        max_ratio: Maximum ratio of translation tokens/characters to source tokens/characters
    
    Returns:
        Fixed translation line
    """
    source_tokens = count_tokens(source_line.strip())
    translation_tokens = count_tokens(translation_line.strip())
    source_chars = len(source_line.strip())
    translation_chars = len(translation_line.strip())
    
    if source_tokens == 0 and source_chars == 0:
        # If source is empty, return empty translation
        return ""
    
    if translation_tokens == 0 and translation_chars == 0:
        # If translation is empty, return source (as fallback)
        return source_line
    
    fixed_line = translation_line
    
    # Check token ratio
    if source_tokens > 0:
        token_ratio = translation_tokens / source_tokens
        if token_ratio > max_ratio:
            # Truncate translation to same number of tokens as source
            translation_words = translation_line.split()
            fixed_words = translation_words[:source_tokens]
            fixed_line = ' '.join(fixed_words)
    
    # Check character ratio (on the potentially already fixed line)
    if source_chars > 0:
        char_ratio = len(fixed_line.strip()) / source_chars
        if char_ratio > max_ratio:
            # Truncate translation to same character length as source
            fixed_line = fixed_line[:source_chars].rstrip()
    
    return fixed_line


def process_translation_file(translation_file: Path, source_file: Path, 
                            output_file: Path, file_type: str,
                            max_ratio: float = 5.0) -> Tuple[int, int]:
    """
    Process a translation file line by line and fix suspicious translations.
    
    Returns:
        (total_lines, fixed_lines)
    """
    # Read source and translation
    source_lines = source_file.read_text(encoding='utf-8').strip().split('\n')
    translation_lines = translation_file.read_text(encoding='utf-8').strip().split('\n')
    
    # Ensure same length
    min_len = min(len(source_lines), len(translation_lines))
    source_lines = source_lines[:min_len]
    translation_lines = translation_lines[:min_len]
    
    # Process each line
    fixed_lines = []
    fixed_count = 0
    
    for i, (source_line, translation_line) in enumerate(zip(source_lines, translation_lines)):
        original = translation_line
        fixed = fix_translation_line(source_line, translation_line, max_ratio)
        
        if fixed != original:
            fixed_count += 1
        
        fixed_lines.append(fixed)
    
    # Write fixed translation
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text('\n'.join(fixed_lines) + '\n', encoding='utf-8')
    
    return min_len, fixed_count


def main():
    parser = argparse.ArgumentParser(
        description="Filter and fix suspicious translations"
    )
    parser.add_argument(
        '--root-dir',
        type=str,
        default='translations',
        help='Root directory to search for translation files (default: current directory)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        required=True,
        help='Path to AraBench dataset directory (to find source files)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='fixed_translations',
        help='Output directory for fixed translations (preserves subdirectory structure) (default: fixed_translations)'
    )
    parser.add_argument(
        '--max-ratio',
        type=float,
        default=5.0,
        help='Maximum ratio of translation tokens/characters to source tokens/characters (default: 5.0)'
    )
    parser.add_argument(
        '--file-types',
        nargs='+',
        choices=['forward_arabic', 'forward_dialect', 'reverse', 'roundtrip', 'all'],
        default=['all'],
        help='Types of translation files to process (default: all)'
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir)
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Determine which file types to process
    if 'all' in args.file_types:
        file_types_to_process = ['forward_arabic', 'forward_dialect', 'reverse', 'roundtrip']
    else:
        file_types_to_process = args.file_types
    
    print(f"🔍 Searching for translation directories in: {root_dir}")
    translation_dirs = find_translation_directories(root_dir)
    
    if not translation_dirs:
        print(f"❌ No translation directories found in {root_dir}")
        return
    
    print(f"📊 Found {len(translation_dirs)} translation directory(ies)")
    
    total_files = 0
    total_fixed = 0
    total_lines = 0
    total_fixed_lines = 0
    
    # Process each translation directory
    for translation_dir in translation_dirs:
        # Get relative path from root_dir to preserve directory structure
        try:
            relative_path = translation_dir.relative_to(root_dir)
        except ValueError:
            # If translation_dir is not under root_dir, use just the directory name
            relative_path = Path(translation_dir.name)
        
        # Extract model name from directory path for display
        model_name = translation_dir.name
        
        print(f"\n{'='*80}")
        print(f"📂 Processing: {model_name}")
        print(f"   Directory: {translation_dir}")
        print(f"   Relative path: {relative_path}")
        print(f"{'='*80}")
        
        # Find all translation files
        translation_files = find_translation_files(translation_dir)
        
        # Process each file type
        for file_type in file_types_to_process:
            if file_type not in translation_files or not translation_files[file_type]:
                continue
            
            print(f"\n📄 Processing {file_type} files...")
            
            for translation_file in translation_files[file_type]:
                # Find corresponding source file
                source_file = find_source_file(translation_file, dataset_dir, file_type)
                
                if not source_file:
                    print(f"   ⚠️  Skipping {translation_file.name}: source file not found")
                    continue
                
                # Determine output path (preserve directory structure)
                output_file = output_dir / relative_path / translation_file.name
                
                # Process file
                try:
                    lines_processed, lines_fixed = process_translation_file(
                        translation_file, source_file, output_file, file_type, args.max_ratio
                    )
                    
                    total_files += 1
                    total_lines += lines_processed
                    total_fixed_lines += lines_fixed
                    
                    if lines_fixed > 0:
                        total_fixed += 1
                        print(f"   ✅ {translation_file.name}: {lines_fixed}/{lines_processed} lines fixed")
                    else:
                        print(f"   ✓  {translation_file.name}: {lines_processed} lines (no fixes needed)")
                
                except Exception as e:
                    print(f"   ❌ Error processing {translation_file.name}: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ Processing complete!")
    print(f"📊 Summary:")
    print(f"   Total files processed: {total_files}")
    print(f"   Files with fixes: {total_fixed}")
    print(f"   Total lines processed: {total_lines}")
    print(f"   Total lines fixed: {total_fixed_lines}")
    if total_lines > 0:
        fix_percentage = (total_fixed_lines / total_lines) * 100
        print(f"   Fix rate: {fix_percentage:.2f}%")
    print(f"📁 Output directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()


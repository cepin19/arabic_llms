#!/usr/bin/env python3
"""
Normalize Arabic translations using camel_tools normalization functions.

This script:
1. Finds all translation files (similar to fix_translations.py)
2. Applies Arabic text normalization:
   - Normalize alef maksura
   - Normalize Unicode (compatibility mode)
   - Normalize teh marbuta
   - Normalize alef
   - Convert Arabic numerals to Latin numerals
3. Saves normalized translations preserving the original directory structure
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

try:
    from camel_tools.utils.normalize import (
        normalize_alef_maksura_ar,
        normalize_unicode,
        normalize_teh_marbuta_ar,
        normalize_alef_ar
    )
except ImportError:
    print("❌ Error: camel_tools not installed. Please install it with: pip install camel-tools")
    exit(1)

"""
Normalize
Utility functions used by to prepare an arabic text to search and index.
@author: Taha Zerrouki <taha_zerrouki at gmail dot com>
@author: Taha Zerrouki
@contact: taha dot zerrouki at gmail dot com
@copyright: Arabtechies, Arabeyes, Taha Zerrouki
@license: GPL
@date:2017/02/15
@version:0.3
"""
import re
try:
    import araby as arabconst
except ImportError:
    print("❌ Error: Could not import araby module. Please ensure araby.py is in the same directory or in your Python path.")
    exit(1)

######################################################################
#{ Indivudual Functions
######################################################################

#--------------------------------------
def strip_tashkeel(text):
    """Strip vowel from a text and return a result text.
    The striped marks are :
        - FATHA, DAMMA, KASRA
        - SUKUN
        - SHADDA
        - FATHATAN, DAMMATAN, KASRATAN, , , .
    Example:
        >>> text=u"الْعَرَبِيّةُ"
        >>> strip_tashkeel(text)
        العربية

    @param text: arabic text.
    @type text: unicode.
    @return: return a striped text.
    @rtype: unicode.
    """
    return arabconst.strip_tashkeel(text)


#strip tatweel from a text and return a result text
#--------------------------------------
def strip_tatweel(text):
    """
    Strip tatweel from a text and return a result text.

    Example:
        >>> text=u"العـــــربية"
        >>> strip_tatweel(text)
        العربية

    @param text: arabic text.
    @type text: unicode.
    @return: return a striped text.
    @rtype: unicode.
    """
    return arabconst.strip_tatweel(text)


#--------------------------------------
def normalize_hamza(text):
    """Normalize Hamza forms into one form, and return a result text.
    The converted letters are :
        - The converted lettersinto HAMZA are: WAW_HAMZA,YEH_HAMZA
        - The converted lettersinto ALEF are: ALEF_MADDA,
        ALEF_HAMZA_ABOVE, ALEF_HAMZA_BELOW ,HAMZA_ABOVE, HAMZA_BELOW

    Example:
        >>> text=u"أهؤلاء من أولئكُ"
        >>> normalize_hamza(text)
        اهءلاء من اولءكُ

    @param text: arabic text.
    @type text: unicode.
    @return: return a converted text.
    @rtype: unicode.
    """
    text = arabconst.ALEFAT_PATTERN.sub(arabconst.ALEF, text)
    return arabconst.HAMZAT_PATTERN.sub(arabconst.HAMZA, text)

#--------------------------------------
def normalize_lamalef(text):
    """Normalize Lam Alef ligatures into two letters (LAM and ALEF),
    and return a result text.
    Some systems present lamAlef ligature as a single letter,
    this function convert it into two letters,
    The converted letters into  LAM and ALEF are :
        - LAM_ALEF, LAM_ALEF_HAMZA_ABOVE, LAM_ALEF_HAMZA_BELOW,
         LAM_ALEF_MADDA_ABOVE

    Example:
        >>> text=u"لانها لالئ الاسلام"
        >>> normalize_lamalef(text)
        لانها لالئ الاسلام

    @param text: arabic text.
    @type text: unicode.
    @return: return a converted text.
    @rtype: unicode.
    """
    return arabconst.normalize_ligature(text)
#--------------------------------------
def normalize_spellerrors(text):
    """Normalize some spellerrors like,
    TEH_MARBUTA into HEH,ALEF_MAKSURA into YEH, and return
    a result text.
    In some context users omit the difference between TEH_MARBUTA
    and HEH, and ALEF_MAKSURA and YEh.
    The conversions are:
        - TEH_MARBUTA into HEH
        - ALEF_MAKSURA into YEH

    Example:
        >>> text=u"اشترت سلمى دمية وحلوى"
        >>> normalize_spellerrors(text)
        اشترت سلمي دميه وحلوي

    @param text: arabic text.
    @type text: unicode.
    @return: return a converted text.
    @rtype: unicode.
    """
    text = re.sub(u'[%s]' % arabconst.TEH_MARBUTA, arabconst.HEH, text)
    return re.sub(u'[%s]' % arabconst.ALEF_MAKSURA, arabconst.YEH, text)

######################################################################
#{ Normalize One Function
######################################################################

def normalize_searchtext(text):
    """Normalize input text and return a result text.
    Normalize a text by :
        - strip tashkeel
        - strip tatweel
        - normalize  Hamza
        - normalize Lam Alef.
        - normalize Teh Marbuta and Alef Maksura
    Example:
        >>> text=u'أستشتري دمـــى آلية لأبنائك قبل الإغلاق'
        >>> normalize_searchtext(text)
        استشتري دمي اليه لابناءك قبل الاغلاق

    @param text: arabic text.
    @type text: unicode.
    @return: return a normalized text.
    @rtype: unicode.
    """
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    text = normalize_lamalef(text)
    text = normalize_hamza(text)
    text = normalize_spellerrors(text)
    return text


# Arabic to Latin numeral mapping
ARABIC_TO_LATIN_NUMERALS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}


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


def convert_arabic_numerals_to_latin(text: str) -> str:
    """
    Convert Arabic numerals (٠-٩) to Latin numerals (0-9).
    
    Args:
        text: Input text that may contain Arabic numerals
    
    Returns:
        Text with Arabic numerals converted to Latin numerals
    """
    result = []
    for char in text:
        if char in ARABIC_TO_LATIN_NUMERALS:
            result.append(ARABIC_TO_LATIN_NUMERALS[char])
        else:
            result.append(char)
    return ''.join(result)


def normalize_arabic_text(text: str) -> str:
    """
    Apply all Arabic text normalization functions.
    
    Args:
        text: Input Arabic text
    
    Returns:
        Normalized Arabic text
    """
    if not text:
        return text
    
    # Apply normalization functions in order
    normalized = text
    #normalized = normalize_searchtext(normalized)

    # 1. Normalize alef maksura
    normalized = normalize_alef_maksura_ar(normalized)
    
    # 2. Normalize Unicode (compatibility mode)
    # Note: User commented this out, keeping it commented
    normalized = normalize_unicode(normalized, compatibility=True)
    
    # 3. Normalize teh marbuta
    normalized = normalize_teh_marbuta_ar(normalized)
    
    # 4. Normalize alef
    normalized = normalize_alef_ar(normalized)
    normalized = normalize_hamza(normalized)
    normalized = normalize_spellerrors(normalized)
    # 5. Convert Arabic numerals to Latin numerals
    normalized = convert_arabic_numerals_to_latin(normalized)
    
    return normalized


def normalize_translation_line(translation_line: str, file_type: str) -> str:
    """
    Normalize a translation line.
    
    Args:
        translation_line: Translation text
        file_type: Type of translation file ('forward_arabic', 'forward_dialect', 'reverse', 'roundtrip')
    
    Returns:
        Normalized translation line
    """
    # Only normalize Arabic text (forward translations)
    # Reverse and roundtrip translations are English, so skip normalization
    if file_type in ['forward_arabic', 'forward_dialect']:
        return normalize_arabic_text(translation_line)
    else:
        # For English translations (reverse, roundtrip), just return as-is
        return translation_line


def process_translation_file(translation_file: Path, output_file: Path, 
                            file_type: str) -> Tuple[int, int]:
    """
    Process a translation file line by line and normalize Arabic text.
    
    Returns:
        (total_lines, normalized_lines)
    """
    # Read translation file
    translation_lines = translation_file.read_text(encoding='utf-8').strip().split('\n')
    
    # Process each line
    normalized_lines = []
    normalized_count = 0
    
    for translation_line in translation_lines:
        original = translation_line
        normalized = normalize_translation_line(translation_line, file_type)
        
        if normalized != original:
            normalized_count += 1
        
        normalized_lines.append(normalized)
    
    # Write normalized translation
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text('\n'.join(normalized_lines) + '\n', encoding='utf-8')
    
    return len(translation_lines), normalized_count


def find_dataset_files(dataset_dir: Path) -> Dict[str, List[Path]]:
    """
    Find all files in the dataset directory.
    
    Returns:
        Dict with 'arabic' and 'english' keys, each containing list of file paths
    """
    files = {
        'arabic': [],
        'english': []
    }
    
    # Find all .ar files (Arabic references)
    for ar_file in dataset_dir.glob("*.ar"):
        # Skip .ar.general and .ar.{dialect_code} files (these are translations, not references)
        if not ar_file.name.endswith('.ar.general') and not re.search(r'\.ar\.[a-z]{2}$', ar_file.name):
            files['arabic'].append(ar_file)
    
    # Find all .en files (English references)
    for en_file in dataset_dir.glob("*.en"):
        # Skip .en.translated and .en.roundtrip files (these are translations, not references)
        if not en_file.name.endswith('.en.translated') and not en_file.name.endswith('.en.roundtrip'):
            files['english'].append(en_file)
    
    return files


def process_reference_file(reference_file: Path, output_file: Path, is_arabic: bool = True) -> Tuple[int, int]:
    """
    Process a reference file line by line and normalize Arabic text (or copy English as-is).
    
    Args:
        reference_file: Path to the reference file (.ar or .en)
        output_file: Path to save the processed reference file
        is_arabic: Whether this is an Arabic file (True) or English file (False)
    
    Returns:
        (total_lines, normalized_lines)
    """
    # Read reference file
    reference_lines = reference_file.read_text(encoding='utf-8').strip().split('\n')
    
    if is_arabic:
        # Process each line for Arabic files
        normalized_lines = []
        normalized_count = 0
        
        for reference_line in reference_lines:
            original = reference_line
            normalized = normalize_arabic_text(reference_line)
            
            if normalized != original:
                normalized_count += 1
            
            normalized_lines.append(normalized)
        
        # Write normalized reference
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text('\n'.join(normalized_lines) + '\n', encoding='utf-8')
        
        return len(reference_lines), normalized_count
    else:
        # For English files, just copy as-is
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text('\n'.join(reference_lines) + '\n', encoding='utf-8')
        
        return len(reference_lines), 0


def main():
    parser = argparse.ArgumentParser(
        description="Normalize Arabic translations using camel_tools"
    )
    parser.add_argument(
        '--root-dir',
        type=str,
        default='translations',
        help='Root directory to search for translation files (default: translations)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='normalized_translations',
        help='Output directory for normalized translations (preserves subdirectory structure) (default: normalized_translations)'
    )
    parser.add_argument(
        '--file-types',
        nargs='+',
        choices=['forward_arabic', 'forward_dialect', 'reverse', 'roundtrip', 'all'],
        default=['all'],
        help='Types of translation files to process (default: all)'
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=None,
        help='Path to AraBench dataset directory (to normalize reference files)'
    )
    parser.add_argument(
        '--reference-output-dir',
        type=str,
        default='normalized_references',
        help='Output directory for normalized reference files (default: normalized_references)'
    )
    parser.add_argument(
        '--normalize-references',
        action='store_true',
        help='Also normalize reference files from dataset directory'
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir)
    
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
    total_normalized = 0
    total_lines = 0
    total_normalized_lines = 0
    
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
                # Determine output path (preserve directory structure)
                output_file = output_dir / relative_path / translation_file.name
                
                # Process file
                try:
                    lines_processed, lines_normalized = process_translation_file(
                        translation_file, output_file, file_type
                    )
                    
                    total_files += 1
                    total_lines += lines_processed
                    total_normalized_lines += lines_normalized
                    
                    if lines_normalized > 0:
                        total_normalized += 1
                        print(f"   ✅ {translation_file.name}: {lines_normalized}/{lines_processed} lines normalized")
                    else:
                        print(f"   ✓  {translation_file.name}: {lines_processed} lines (no normalization needed)")
                
                except Exception as e:
                    print(f"   ❌ Error processing {translation_file.name}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Process reference files if requested
    ref_total_files = 0
    ref_total_normalized = 0
    ref_total_lines = 0
    ref_total_normalized_lines = 0
    
    if args.normalize_references:
        if not args.dataset_dir:
            print(f"\n⚠️  --normalize-references specified but --dataset-dir not provided, skipping reference normalization")
        else:
            dataset_dir = Path(args.dataset_dir).resolve()
            reference_output_dir = Path(args.reference_output_dir)
            
            if not dataset_dir.exists():
                print(f"\n⚠️  Dataset directory not found: {dataset_dir}, skipping reference normalization")
            else:
                print(f"\n{'='*80}")
                print(f"📚 Processing reference files from dataset directory")
                print(f"   Dataset directory: {dataset_dir}")
                print(f"   Output directory: {reference_output_dir}")
                print(f"{'='*80}")
                
                dataset_files = find_dataset_files(dataset_dir)
                arabic_files = dataset_files['arabic']
                english_files = dataset_files['english']
                
                if not arabic_files and not english_files:
                    print(f"⚠️  No reference files found in {dataset_dir}")
                else:
                    print(f"📊 Found {len(arabic_files)} Arabic file(s) and {len(english_files)} English file(s)")
                    
                    # Process Arabic files (normalize)
                    for arabic_file in arabic_files:
                        # Determine output path (preserve filename and structure)
                        # If dataset_dir has subdirectories, preserve them
                        try:
                            relative_path = arabic_file.relative_to(dataset_dir)
                            output_file = reference_output_dir / relative_path
                        except ValueError:
                            # If file is not under dataset_dir, use just the filename
                            output_file = reference_output_dir / arabic_file.name
                        
                        # Process file
                        try:
                            lines_processed, lines_normalized = process_reference_file(
                                arabic_file, output_file, is_arabic=True
                            )
                            
                            ref_total_files += 1
                            ref_total_lines += lines_processed
                            ref_total_normalized_lines += lines_normalized
                            
                            if lines_normalized > 0:
                                ref_total_normalized += 1
                                print(f"   ✅ {arabic_file.name}: {lines_normalized}/{lines_processed} lines normalized")
                            else:
                                print(f"   ✓  {arabic_file.name}: {lines_processed} lines (no normalization needed)")
                        
                        except Exception as e:
                            print(f"   ❌ Error processing {arabic_file.name}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Process English files (copy as-is)
                    for english_file in english_files:
                        # Determine output path (preserve filename and structure)
                        try:
                            relative_path = english_file.relative_to(dataset_dir)
                            output_file = reference_output_dir / relative_path
                        except ValueError:
                            # If file is not under dataset_dir, use just the filename
                            output_file = reference_output_dir / english_file.name
                        
                        # Process file
                        try:
                            lines_processed, lines_normalized = process_reference_file(
                                english_file, output_file, is_arabic=False
                            )
                            
                            ref_total_files += 1
                            ref_total_lines += lines_processed
                            
                            print(f"   📋 {english_file.name}: {lines_processed} lines (copied)")
                        
                        except Exception as e:
                            print(f"   ❌ Error processing {english_file.name}: {e}")
                            import traceback
                            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"✅ Processing complete!")
    print(f"📊 Summary - Translation Files:")
    print(f"   Total files processed: {total_files}")
    print(f"   Files with normalization: {total_normalized}")
    print(f"   Total lines processed: {total_lines}")
    print(f"   Total lines normalized: {total_normalized_lines}")
    if total_lines > 0:
        normalize_percentage = (total_normalized_lines / total_lines) * 100
        print(f"   Normalization rate: {normalize_percentage:.2f}%")
    print(f"📁 Translation output directory: {output_dir}")
    
    if args.normalize_references and args.dataset_dir and Path(args.dataset_dir).exists():
        print(f"\n📊 Summary - Reference Files:")
        print(f"   Total files processed: {ref_total_files}")
        print(f"   Files with normalization: {ref_total_normalized}")
        print(f"   Total lines processed: {ref_total_lines}")
        print(f"   Total lines normalized: {ref_total_normalized_lines}")
        if ref_total_lines > 0:
            ref_normalize_percentage = (ref_total_normalized_lines / ref_total_lines) * 100
            print(f"   Normalization rate: {ref_normalize_percentage:.2f}%")
        print(f"📁 Reference output directory: {reference_output_dir}")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()


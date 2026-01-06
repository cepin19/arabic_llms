#!/usr/bin/env python3
"""
Generate LaTeX tables and figure references from fixed_translations results
and visualizations_fixed directory.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse
import sys
import numpy as np

# Import functions from visualize_scores.py
sys.path.insert(0, str(Path(__file__).parent))
try:
    from visualize_scores import (
        collect_all_scores,
        calculate_average_rankings,
        find_comet_keys,
        find_all_comet_variants,
        get_comet_score_for_variant,
        find_comet_score
    )
except ImportError:
    print("Warning: Could not import functions from visualize_scores.py")
    print("Average ranking tables will not be generated.")
    collect_all_scores = None
    calculate_average_rankings = None


def extract_model_name(dir_name: str) -> str:
    """Extract clean model name from directory name."""
    # Remove prefixes
    name = dir_name.replace('arabench_translations_', '')
    name = name.replace('arabench_aren_translations_', '')
    name = name.replace('arabench_rtt_translations_', '')
    
    # Handle roundtrip (has two model names)
    if '_' in name and name.count('_') > 1:
        parts = name.split('_')
        # Try to identify if it's a roundtrip
        if any('rtt' in p for p in parts):
            # Roundtrip format: model1_model2
            # For now, use the first model
            return parts[0] if len(parts) > 0 else name
    
    return name


def categorize_model(model_name: str) -> str:
    """Categorize model into groups."""
    model_lower = model_name.lower()
    
    if 'jais' in model_lower:
        return 'Arabic-Specialized'
    elif 'nile' in model_lower:
        return 'Arabic-Specialized'
    elif 'c4ai-command-r7b-arabic' in model_lower or 'command-r7b-arabic' in model_lower:
        return 'Arabic-Specialized'
    elif 'aya' in model_lower:
        return 'Multilingual'
    elif 'command' in model_lower:
        return 'Multilingual'
    elif 'gpt-4' in model_lower:
        return 'Commercial API'
    elif 'gemma' in model_lower or 'eurollm' in model_lower or 'mistral' in model_lower or 'qwen' in model_lower or 'llama' in model_lower or 'falcon' in model_lower:
        return 'Multilingual'
    else:
        return 'Other'


def get_model_size(model_name: str) -> Optional[str]:
    """Get model size from model name."""
    model_lower = model_name.lower()
    
    # Arabic-Specialized
    if 'jais-2-8b' in model_lower:
        return '8B'
    elif 'jais-2-70b' in model_lower:
        return '70B'
    elif 'nile-chat-12b' in model_lower or 'nile-chat-4b' in model_lower:
        if '12b' in model_lower:
            return '12B'
        elif '4b' in model_lower:
            return '4B'
    elif 'c4ai-command-r7b-arabic' in model_lower or 'command-r7b-arabic' in model_lower:
        return '7B'
    
    # Multilingual
    elif 'aya-expanse-8b' in model_lower or 'aya-expanse-32b' in model_lower:
        if '8b' in model_lower:
            return '8B'
        elif '32b' in model_lower:
            return '32B'
    elif 'c4ai-command-r-08-2024' in model_lower or 'command-r-08-2024' in model_lower:
        return '32B'
    elif 'c4ai-command-r-v01' in model_lower or 'command-r-v01' in model_lower:
        return '35B'
    elif 'command-a-translate' in model_lower:
        return '111B'
    elif 'gemma-3-4b' in model_lower:
        return '4B'
    elif 'gemma-3-27b' in model_lower:
        return '27B'
    elif 'eurollm-9b' in model_lower:
        return '9B'
    elif 'mistral-small-3.2-24b' in model_lower:
        return '24B'
    elif 'qwen3-4b' in model_lower:
        return '4B'
    elif 'llama-3.3-70b' in model_lower:
        return '70B'
    elif 'falcon-h1-34b' in model_lower:
        return '34B'
    
    # Commercial API
    elif 'gpt-4.1-nano' in model_lower:
        return 'N/A'
    elif 'gpt-4.1-mini' in model_lower:
        return 'N/A'
    
    return None


def load_scores_file(scores_file: Path) -> Optional[Dict]:
    """Load and parse a scores.json file."""
    try:
        with open(scores_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {scores_file}: {e}")
        return None


def extract_comet_keys(data: Dict) -> List[str]:
    """Extract all COMET key names from the data."""
    comet_keys = []
    
    def search_dict(d):
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(key, str) and key.startswith('COMET_'):
                    if key not in comet_keys:
                        comet_keys.append(key)
                if isinstance(value, (dict, list)):
                    search_dict(value)
        elif isinstance(d, list):
            for item in d:
                search_dict(item)
    
    search_dict(data)
    return sorted(comet_keys)


def get_metric_value(data: Dict, metric: str, score_type: Optional[str] = None) -> Optional[float]:
    """Get a metric value from data structure."""
    if score_type:
        if score_type in data and isinstance(data[score_type], dict):
            if metric in data[score_type]:
                return data[score_type][metric]
    else:
        if metric in data:
            return data[metric]
    return None


def process_forward_translations(fixed_translations_dir: Path) -> Dict:
    """Process forward translation results."""
    results = {}
    dialect_results = defaultdict(lambda: defaultdict(dict))  # dialect -> model -> scores
    
    forward_dirs = [d for d in fixed_translations_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('arabench_translations_') 
                   and 'aren' not in d.name and 'rtt' not in d.name]
    
    for trans_dir in sorted(forward_dirs):
        scores_file = trans_dir / 'scores.json'
        if not scores_file.exists():
            continue
        
        data = load_scores_file(scores_file)
        if not data:
            continue
        
        model_name = extract_model_name(trans_dir.name)
        model_category = categorize_model(model_name)
        
        # Extract averages
        if 'averages' in data:
            avg_data = data['averages']
            
            model_results = {
                'model': model_name,
                'category': model_category,
                'arabic_general': {},
                'dialect': {}
            }
            
            # Arabic general scores
            if 'arabic_general' in avg_data:
                ag = avg_data['arabic_general']
                if 'BLEU' in ag:
                    model_results['arabic_general']['BLEU'] = ag['BLEU']
                if 'CHRF' in ag:
                    model_results['arabic_general']['CHRF'] = ag['CHRF']
                # COMET scores
                comet_keys = extract_comet_keys(ag)
                for key in comet_keys:
                    if key in ag:
                        model_results['arabic_general'][key] = ag[key]
            
            # Dialect scores
            if 'dialect' in avg_data:
                dial = avg_data['dialect']
                if 'BLEU' in dial:
                    model_results['dialect']['BLEU'] = dial['BLEU']
                if 'CHRF' in dial:
                    model_results['dialect']['CHRF'] = dial['CHRF']
                # COMET scores
                comet_keys = extract_comet_keys(dial)
                for key in comet_keys:
                    if key in dial:
                        model_results['dialect'][key] = dial[key]
            
            results[model_name] = model_results
        
        # Extract dialect-specific results from merged dialects (both regular and madar_only)
        if 'results' in data:
            for result in data['results']:
                if result.get('type') in ('dialect_merged', 'dialect_merged_madar_only'):
                    dialect_code = result.get('dialect_code', '')
                    dialect_name = result.get('dialect_name', dialect_code)
                    # For madar_only, append suffix to distinguish
                    if result.get('type') == 'dialect_merged_madar_only':
                        dialect_name = f"{dialect_name} "
                    
                    if 'arabic_general' in result:
                        dialect_results[dialect_name][model_name] = {
                            'arabic_general': result['arabic_general'].copy(),
                            'dialect': result.get('dialect', {}).copy(),
                            'category': model_category,  # Preserve category
                            'is_madar_only': (result.get('type') == 'dialect_merged_madar_only')
                        }
    
    results['_dialect_breakdown'] = dict(dialect_results)
    return results


def process_reverse_translations(fixed_translations_dir: Path) -> Dict:
    """Process reverse translation results."""
    results = {}
    
    reverse_dirs = [d for d in fixed_translations_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('arabench_aren_translations_')]
    
    for trans_dir in sorted(reverse_dirs):
        scores_file = trans_dir / 'scores.json'
        if not scores_file.exists():
            continue
        
        data = load_scores_file(scores_file)
        if not data:
            continue
        
        model_name = extract_model_name(trans_dir.name)
        model_category = categorize_model(model_name)
        
        # Extract averages
        if 'averages' in data:
            avg_data = data['averages']
            
            model_results = {
                'model': model_name,
                'category': model_category,
                'scores': {}
            }
            
            if 'BLEU' in avg_data:
                model_results['scores']['BLEU'] = avg_data['BLEU']
            if 'CHRF' in avg_data:
                model_results['scores']['CHRF'] = avg_data['CHRF']
            
            # COMET scores
            comet_keys = extract_comet_keys(avg_data)
            for key in comet_keys:
                if key in avg_data:
                    model_results['scores'][key] = avg_data[key]
            
            results[model_name] = model_results
    
    return results


def process_roundtrip_translations(fixed_translations_dir: Path) -> Dict:
    """Process roundtrip translation results."""
    results = {}
    
    roundtrip_dirs = [d for d in fixed_translations_dir.iterdir() 
                     if d.is_dir() and d.name.startswith('arabench_rtt_translations_')]
    
    for trans_dir in sorted(roundtrip_dirs):
        scores_file = trans_dir / 'scores.json'
        if not scores_file.exists():
            continue
        
        data = load_scores_file(scores_file)
        if not data:
            continue
        
        # Extract model pair from directory name
        dir_name = trans_dir.name.replace('arabench_rtt_translations_', '')
        parts = dir_name.split('_')
        # Simple heuristic: find where model names might be
        # For now, use directory name as identifier
        model_pair = dir_name
        
        # Extract averages
        if 'averages' in data:
            avg_data = data['averages']
            
            model_results = {
                'model_pair': model_pair,
                'scores': {}
            }
            
            if 'BLEU' in avg_data:
                model_results['scores']['BLEU'] = avg_data['BLEU']
            if 'CHRF' in avg_data:
                model_results['scores']['CHRF'] = avg_data['CHRF']
            
            # COMET scores
            comet_keys = extract_comet_keys(avg_data)
            for key in comet_keys:
                if key in avg_data:
                    model_results['scores'][key] = avg_data[key]
            
            results[model_pair] = model_results
    
    return results


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '\\': r'\textbackslash{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_number(value: Optional[float], decimals: int = 2) -> str:
    """Format a number for LaTeX table."""
    if value is None:
        return '---'
    return f"{value:.{decimals}f}"


def generate_forward_table_latex(forward_results: Dict, metrics: List[str] = None) -> str:
    """Generate LaTeX table for forward translation results."""
    if metrics is None:
        metrics = ['BLEU', 'CHRF']
    
    # Group by category (skip dialect breakdown)
    by_category = defaultdict(list)
    for model_name, results in forward_results.items():
        if model_name == '_dialect_breakdown':
            continue
        if 'category' not in results:
            continue
        by_category[results['category']].append((model_name, results))
    
    # Sort categories
    category_order = ['Arabic-Specialized', 'Multilingual', 'Commercial API', 'Other']
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Forward Translation Results (English → Arabic): Overall Performance}")
    latex.append("\\label{tab:forward_overall}")
    # Use 3pt spacing between all columns
    col_spec = "l@{\\hspace{3pt}}c"  # Model and Size columns
    for i, metric in enumerate(metrics):
        col_spec += "@{\\hspace{3pt}}cc"  # Two columns per metric (General and Dialect)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header
    header = "Model & Size & "
    for metric in metrics:
        header += f"\\multicolumn{{2}}{{c@{{\\hspace{{3pt}}}}}}{{{metric}}} & "
    header = header.rstrip(" & ") + " \\\\"
    header += "\\cmidrule(lr){3-4} " if len(metrics) > 0 else ""
    for i, metric in enumerate(metrics):
        if i > 0:
            header += "\\cmidrule(lr){" + str(5 + i*2) + "-" + str(6 + i*2) + "} "
    header += " & General & Dialect " * len(metrics) + "\\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for category in category_order:
        if category not in by_category:
            continue
        
        # Category header
        latex.append(f"\\multicolumn{{{2 + len(metrics)*2}}}{{l}}{{\\textit{{{escape_latex(category)}}}}} \\\\")
        latex.append("\\midrule")
        
        # Models in category
        models_in_cat = sorted(by_category[category], key=lambda x: x[0])
        for model_name, results in models_in_cat:
            model_size = get_model_size(model_name)
            size_display = model_size if model_size else "---"
            row = f"{escape_latex(model_name)} & {size_display} & "
            for metric in metrics:
                ag_val = results['arabic_general'].get(metric)
                dial_val = results['dialect'].get(metric)
                row += f"{format_number(ag_val)} & {format_number(dial_val)} & "
            row = row.rstrip(" & ") + " \\\\"
            latex.append(row)
        
        latex.append("\\midrule")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_prompt_comparison_table_latex(forward_results: Dict, metrics: List[str] = None) -> str:
    """
    Generate LaTeX table comparing average merged dialect scores using 
    general prompt (arabic_general) vs dialect-specific prompt (dialect).
    
    Args:
        forward_results: Dictionary from process_forward_translations
        metrics: List of metrics to include (default: ['BLEU', 'CHRF'])
    
    Returns:
        LaTeX table string
    """
    if metrics is None:
        metrics = ['BLEU', 'CHRF']
    
    # Extract dialect breakdown
    dialect_breakdown = forward_results.get('_dialect_breakdown', {})
    if not dialect_breakdown:
        return "% No merged dialect data available for prompt comparison"
    
    # Collect scores for each model across all dialects
    model_scores = defaultdict(lambda: {
        'arabic_general': defaultdict(list),  # metric -> [scores]
        'dialect': defaultdict(list),  # metric -> [scores]
        'category': None
    })
    
    # Iterate through all dialects and collect scores
    for dialect_name, models_data in dialect_breakdown.items():
        for model_name, scores_data in models_data.items():
            if model_name in forward_results:
                category = forward_results[model_name].get('category', 'Other')
            else:
                category = scores_data.get('category', 'Other')
            
            model_scores[model_name]['category'] = category
            
            # Collect arabic_general scores
            if 'arabic_general' in scores_data:
                ag_scores = scores_data['arabic_general']
                for metric in metrics:
                    if metric in ag_scores:
                        model_scores[model_name]['arabic_general'][metric].append(ag_scores[metric])
            
            # Collect dialect scores
            if 'dialect' in scores_data:
                dial_scores = scores_data['dialect']
                for metric in metrics:
                    if metric in dial_scores:
                        model_scores[model_name]['dialect'][metric].append(dial_scores[metric])
    
    # Compute averages for each model
    model_averages = {}
    for model_name, scores in model_scores.items():
        category = scores['category']
        averages = {
            'category': category,
            'arabic_general': {},
            'dialect': {}
        }
        
        for metric in metrics:
            # Average for arabic_general
            ag_scores = scores['arabic_general'][metric]
            if ag_scores:
                averages['arabic_general'][metric] = np.mean(ag_scores)
            else:
                averages['arabic_general'][metric] = None
            
            # Average for dialect
            dial_scores = scores['dialect'][metric]
            if dial_scores:
                averages['dialect'][metric] = np.mean(dial_scores)
            else:
                averages['dialect'][metric] = None
        
        model_averages[model_name] = averages
    
    if not model_averages:
        return "% No scores found for prompt comparison"
    
    # Group by category
    by_category = defaultdict(list)
    for model_name, averages in model_averages.items():
        category = averages['category']
        by_category[category].append((model_name, averages))
    
    # Sort categories
    category_order = ['Arabic-Specialized', 'Multilingual', 'Commercial API', 'Other']
    
    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Average Merged Dialect Scores: General vs. Dialect-Specific Prompt}")
    latex.append("\\label{tab:prompt_comparison}")
    
    # Column specification
    col_spec = "l@{\\hspace{3pt}}c"  # Model and Size
    for metric in metrics:
        col_spec += "@{\\hspace{3pt}}cc"  # Two columns per metric (General and Dialect)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header
    header = "Model & Size & "
    for metric in metrics:
        header += f"\\multicolumn{{2}}{{c@{{\\hspace{{3pt}}}}}}{{{metric}}} & "
    header = header.rstrip(" & ") + " \\\\"
    
    # Add cmidrules
    if len(metrics) > 0:
        start_col = 3
        for i, metric in enumerate(metrics):
            if i > 0:
                header += " "
            header += f"\\cmidrule(lr){{{start_col + i*2}-{start_col + i*2 + 1}}}"
    header += " & "
    
    # Sub-header
    sub_header = " & "
    for metric in metrics:
        sub_header += "General & Dialect & "
    sub_header = sub_header.rstrip(" & ") + " \\\\"
    header += sub_header
    
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for category in category_order:
        if category not in by_category:
            continue
        
        models_in_category = sorted(by_category[category], key=lambda x: x[0])
        
        for model_name, averages in models_in_category:
            model_size = get_model_size(model_name)
            size_display = model_size if model_size else "---"
            
            row = f"{escape_latex(model_name)} & {size_display} & "
            
            for metric in metrics:
                ag_score = averages['arabic_general'].get(metric)
                dial_score = averages['dialect'].get(metric)
                
                ag_display = format_number(ag_score, 2)
                dial_display = format_number(dial_score, 2)
                
                row += f"{ag_display} & {dial_display} & "
            
            row = row.rstrip(" & ") + " \\\\"
            latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_dialect_prompt_comparison_table_latex(forward_results: Dict, metrics: List[str] = None) -> str:
    """
    Generate LaTeX table comparing average merged dialect scores using 
    general prompt (arabic_general) vs dialect-specific prompt (dialect),
    averaged across all models for each dialect.
    
    Args:
        forward_results: Dictionary from process_forward_translations
        metrics: List of metrics to include (default: ['BLEU', 'CHRF'])
    
    Returns:
        LaTeX table string
    """
    if metrics is None:
        metrics = ['BLEU', 'CHRF']
    
    # Extract dialect breakdown
    dialect_breakdown = forward_results.get('_dialect_breakdown', {})
    if not dialect_breakdown:
        return "% No merged dialect data available for dialect prompt comparison"
    
    # Collect scores for each dialect across all models
    # Deduplicate dialects by removing trailing spaces (MADAR-only indicator)
    dialect_scores = defaultdict(lambda: {
        'arabic_general': defaultdict(list),  # metric -> [scores]
        'dialect': defaultdict(list),  # metric -> [scores]
    })
    
    # Iterate through all dialects and collect scores
    # Filter to only include non-MADAR-only dialects (those without trailing space)
    for dialect_name, models_data in dialect_breakdown.items():
        # Skip MADAR-only dialects (they have trailing space)
        if dialect_name.endswith(' '):
            continue
        
        for model_name, scores_data in models_data.items():
            # Collect arabic_general scores
            if 'arabic_general' in scores_data:
                ag_scores = scores_data['arabic_general']
                for metric in metrics:
                    if metric in ag_scores:
                        dialect_scores[dialect_name]['arabic_general'][metric].append(ag_scores[metric])
            
            # Collect dialect scores
            if 'dialect' in scores_data:
                dial_scores = scores_data['dialect']
                for metric in metrics:
                    if metric in dial_scores:
                        dialect_scores[dialect_name]['dialect'][metric].append(dial_scores[metric])
    
    # Compute averages for each dialect
    dialect_averages = {}
    for dialect_name, scores in dialect_scores.items():
        averages = {
            'arabic_general': {},
            'dialect': {}
        }
        
        for metric in metrics:
            # Average for arabic_general
            ag_scores = scores['arabic_general'][metric]
            if ag_scores:
                averages['arabic_general'][metric] = np.mean(ag_scores)
            else:
                averages['arabic_general'][metric] = None
            
            # Average for dialect
            dial_scores = scores['dialect'][metric]
            if dial_scores:
                averages['dialect'][metric] = np.mean(dial_scores)
            else:
                averages['dialect'][metric] = None
        
        dialect_averages[dialect_name] = averages
    
    if not dialect_averages:
        return "% No scores found for dialect prompt comparison"
    
    # Sort dialects by name
    sorted_dialects = sorted(dialect_averages.items(), key=lambda x: x[0])
    
    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Average Merged Dialect Scores Across Models: General vs. Dialect-Specific Prompt}")
    latex.append("\\label{tab:dialect_prompt_comparison}")
    
    # Column specification (removed # Models column)
    col_spec = "l"  # Dialect
    for metric in metrics:
        col_spec += "@{\\hspace{3pt}}cc"  # Two columns per metric (General and Dialect)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header
    header = "Dialect & "
    for metric in metrics:
        header += f"\\multicolumn{{2}}{{c@{{\\hspace{{3pt}}}}}}{{{metric}}} & "
    header = header.rstrip(" & ") + " \\\\"
    
    # Add cmidrules
    if len(metrics) > 0:
        start_col = 2
        for i, metric in enumerate(metrics):
            if i > 0:
                header += " "
            header += f"\\cmidrule(lr){{{start_col + i*2}-{start_col + i*2 + 1}}}"
    header += " & "
    
    # Sub-header
    sub_header = ""
    for metric in metrics:
        sub_header += "General & Dialect & "
    sub_header = sub_header.rstrip(" & ") + " \\\\"
    header += sub_header
    
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for dialect_name, averages in sorted_dialects:
        row = f"{escape_latex(dialect_name)} & "
        
        for metric in metrics:
            ag_score = averages['arabic_general'].get(metric)
            dial_score = averages['dialect'].get(metric)
            
            ag_display = format_number(ag_score, 1)
            dial_display = format_number(dial_score, 1)
            
            row += f"{ag_display} & {dial_display} & "
        
        row = row.rstrip(" & ") + " \\\\"
        latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_model_dialect_prompt_comparison_table_latex(forward_results: Dict, model_name: str, metrics: List[str] = None) -> str:
    """
    Generate LaTeX table comparing merged dialect scores using 
    general prompt (arabic_general) vs dialect-specific prompt (dialect),
    for a specific model across all dialects.
    
    Args:
        forward_results: Dictionary from process_forward_translations
        model_name: Name of the model to generate table for
        metrics: List of metrics to include (default: ['BLEU', 'CHRF'])
    
    Returns:
        LaTeX table string
    """
    if metrics is None:
        metrics = ['BLEU', 'CHRF']
    
    # Extract dialect breakdown
    dialect_breakdown = forward_results.get('_dialect_breakdown', {})
    if not dialect_breakdown:
        return "% No merged dialect data available for model dialect prompt comparison"
    
    # Collect scores for this model across all dialects
    dialect_scores = {}
    
    # Iterate through all dialects and collect scores for this model
    for dialect_name, models_data in dialect_breakdown.items():
        # Skip MADAR-only dialects (they have trailing space)
        if dialect_name.endswith(' '):
            continue
        
        # Check if this model has scores for this dialect
        if model_name not in models_data:
            continue
        
        scores_data = models_data[model_name]
        dialect_scores[dialect_name] = {
            'arabic_general': {},
            'dialect': {}
        }
        
        # Collect arabic_general scores
        if 'arabic_general' in scores_data:
            ag_scores = scores_data['arabic_general']
            for metric in metrics:
                if metric in ag_scores:
                    dialect_scores[dialect_name]['arabic_general'][metric] = ag_scores[metric]
        
        # Collect dialect scores
        if 'dialect' in scores_data:
            dial_scores = scores_data['dialect']
            for metric in metrics:
                if metric in dial_scores:
                    dialect_scores[dialect_name]['dialect'][metric] = dial_scores[metric]
    
    if not dialect_scores:
        return f"% No scores found for model {model_name} in dialect prompt comparison"
    
    # Sort dialects by name
    sorted_dialects = sorted(dialect_scores.items(), key=lambda x: x[0])
    
    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{Merged Dialect Scores: General vs. Dialect-Specific Prompt ({escape_latex(model_name)})}}")
    safe_model_name = model_name.replace(' ', '_').replace('-', '_').lower()
    latex.append(f"\\label{{tab:model_dialect_prompt_comparison_{safe_model_name}}}")
    
    # Column specification
    col_spec = "l"  # Dialect
    for metric in metrics:
        col_spec += "@{\\hspace{3pt}}cc"  # Two columns per metric (General and Dialect)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header
    header = "Dialect & "
    for metric in metrics:
        header += f"\\multicolumn{{2}}{{c@{{\\hspace{{3pt}}}}}}{{{metric}}} & "
    header = header.rstrip(" & ") + " \\\\"
    
    # Add cmidrules
    if len(metrics) > 0:
        start_col = 2
        for i, metric in enumerate(metrics):
            if i > 0:
                header += " "
            header += f"\\cmidrule(lr){{{start_col + i*2}-{start_col + i*2 + 1}}}"
    header += " & "
    
    # Sub-header
    sub_header = ""
    for metric in metrics:
        sub_header += "General & Dialect & "
    sub_header = sub_header.rstrip(" & ") + " \\\\"
    header += sub_header
    
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for dialect_name, scores in sorted_dialects:
        row = f"{escape_latex(dialect_name)} & "
        
        for metric in metrics:
            ag_score = scores['arabic_general'].get(metric)
            dial_score = scores['dialect'].get(metric)
            
            ag_display = format_number(ag_score, 1)
            dial_display = format_number(dial_score, 1)
            
            row += f"{ag_display} & {dial_display} & "
        
        row = row.rstrip(" & ") + " \\\\"
        latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_dialect_table_latex(forward_results: Dict, metrics: List[str] = None, top_n_dialects: int = 8) -> str:
    """Generate LaTeX table for dialect-specific forward translation results."""
    if metrics is None:
        metrics = ['BLEU', 'CHRF']
    
    dialect_breakdown = forward_results.get('_dialect_breakdown', {})
    if not dialect_breakdown:
        return "% No dialect breakdown data available"
    
    # Select top N dialects by number of models
    dialects = sorted(dialect_breakdown.keys(), 
                     key=lambda d: len(dialect_breakdown[d]), 
                     reverse=True)[:top_n_dialects]
    
    # Get all models that appear in at least one dialect
    all_models = set()
    for dialect_data in dialect_breakdown.values():
        all_models.update(dialect_data.keys())
    all_models = sorted(all_models)
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Forward Translation: Dialect-Specific Performance (Dialect Variant)}")
    latex.append("\\label{tab:forward_dialects}")
    # Use 3pt spacing between all columns
    col_spec = "l@{\\hspace{3pt}}c"  # Model and Size columns
    for dialect in dialects:
        for metric in metrics:
            col_spec += "@{\\hspace{3pt}}c"
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header
    header = "Model & Size & "
    for dialect in dialects:
        header += f"\\multicolumn{{{len(metrics)}}}{{c@{{\\hspace{{3pt}}}}}}{{{escape_latex(dialect)}}} & "
    header = header.rstrip(" & ") + " \\\\"
    latex.append(header)
    
    # Sub-header for metrics
    subheader = " & & "
    for dialect in dialects:
        for metric in metrics:
            subheader += f"{escape_latex(metric)} & "
    subheader = subheader.rstrip(" & ") + " \\\\"
    latex.append(subheader)
    latex.append("\\midrule")
    
    # Data rows
    for model in all_models:
        model_size = get_model_size(model)
        size_display = model_size if model_size else "---"
        row = f"{escape_latex(model)} & {size_display} & "
        for dialect in dialects:
            if model in dialect_breakdown[dialect]:
                scores = dialect_breakdown[dialect][model].get('dialect', {})
                for metric in metrics:
                    val = scores.get(metric)
                    row += f"{format_number(val)} & "
            else:
                row += "--- & " * len(metrics)
        row = row.rstrip(" & ") + " \\\\"
        latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_reverse_table_latex(reverse_results: Dict, metrics: List[str] = None) -> str:
    """Generate LaTeX table for reverse translation results."""
    if metrics is None:
        metrics = ['BLEU', 'CHRF']
    
    # Group by category
    by_category = defaultdict(list)
    for model_name, results in reverse_results.items():
        by_category[results['category']].append((model_name, results))
    
    # Sort categories
    category_order = ['Arabic-Specialized', 'Multilingual', 'Commercial API', 'Other']
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Reverse Translation Results (Arabic → English): Overall Performance}")
    latex.append("\\label{tab:reverse_overall}")
    # Use 3pt spacing between all columns
    col_spec = "l@{\\hspace{3pt}}c"  # Model and Size columns
    for metric in metrics:
        col_spec += "@{\\hspace{3pt}}c"
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header
    header = "Model & Size & " + " & ".join([escape_latex(m) for m in metrics]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for category in category_order:
        if category not in by_category:
            continue
        
        # Category header
        latex.append(f"\\multicolumn{{{2 + len(metrics)}}}{{l}}{{\\textit{{{escape_latex(category)}}}}} \\\\")
        latex.append("\\midrule")
        
        # Models in category
        models_in_cat = sorted(by_category[category], key=lambda x: x[0])
        for model_name, results in models_in_cat:
            model_size = get_model_size(model_name)
            size_display = model_size if model_size else "---"
            row = f"{escape_latex(model_name)} & {size_display} & "
            for metric in metrics:
                val = results['scores'].get(metric)
                row += f"{format_number(val)} & "
            row = row.rstrip(" & ") + " \\\\"
            latex.append(row)
        
        latex.append("\\midrule")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_average_rankings_table_latex(test_sets: Dict, models: List[str], 
                                          score_type: Optional[str], metric: str,
                                          comet_variant: Optional[str] = None,
                                          title: str = "") -> str:
    """
    Generate LaTeX table for average rankings.
    
    Args:
        test_sets: Dictionary mapping test_set -> {model -> scores}
        models: List of model names
        score_type: 'arabic_general', 'dialect', or None
        metric: 'BLEU', 'CHRF', or 'COMET'
        comet_variant: If metric is 'COMET', the specific COMET variant
        title: Title for the table
    """
    if not calculate_average_rankings:
        return "% Average ranking calculation not available"
    
    avg_rankings = calculate_average_rankings(test_sets, models, score_type, metric, comet_variant)
    
    if not avg_rankings:
        return f"% No {metric} scores found for average ranking calculation"
    
    # Sort by average rank (ascending - lower is better)
    sorted_models = sorted(avg_rankings.items(), key=lambda x: x[1])
    
    # Count test sets for each model
    model_counts = {}
    key = f'{score_type}_{metric}' if score_type else None
    for model in models:
        count = 0
        for test_set, model_scores in test_sets.items():
            if model in model_scores:
                if key and metric != 'COMET' and key in model_scores[model]:
                    count += 1
                elif key and metric == 'COMET':
                    if comet_variant:
                        score = get_comet_score_for_variant(model_scores[model], comet_variant, score_type)
                        if score is not None:
                            count += 1
                    else:
                        if find_comet_score(model_scores[model], score_type) is not None:
                            count += 1
                elif not key:
                    if metric == 'COMET':
                        if comet_variant:
                            score = get_comet_score_for_variant(model_scores[model], comet_variant)
                            if score is not None:
                                count += 1
                        else:
                            if find_comet_score(model_scores[model]) is not None:
                                count += 1
                    elif metric in model_scores[model] or \
                       f'arabic_general_{metric}' in model_scores[model] or \
                       f'dialect_{metric}' in model_scores[model]:
                        count += 1
        model_counts[model] = count
    
    # Calculate wins (number of test sets where model has highest score)
    model_wins = {model: 0 for model in models}
    for test_set, model_scores in test_sets.items():
        best_score = None
        best_model = None
        for model in models:
            if model not in model_scores:
                continue
            
            score = None
            if key and metric != 'COMET':
                score = model_scores[model].get(key)
            elif key and metric == 'COMET':
                if comet_variant:
                    score = get_comet_score_for_variant(model_scores[model], comet_variant, score_type)
                else:
                    score = find_comet_score(model_scores[model], score_type)
            elif not key:
                if metric == 'COMET':
                    if comet_variant:
                        score = get_comet_score_for_variant(model_scores[model], comet_variant)
                    else:
                        score = find_comet_score(model_scores[model])
                else:
                    score = model_scores[model].get(metric)
                    if score is None:
                        score = model_scores[model].get(f'arabic_general_{metric}')
                    if score is None:
                        score = model_scores[model].get(f'dialect_{metric}')
            
            if score is not None:
                if best_score is None or score > best_score:
                    best_score = score
                    best_model = model
        
        if best_model is not None:
            model_wins[best_model] = model_wins.get(best_model, 0) + 1
    
    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    if not title:
        title_label = ''
        if score_type == 'arabic_general':
            title_label = 'General Arabic '
        elif score_type == 'dialect':
            title_label = 'Dialect '
        metric_label = comet_variant if (metric == 'COMET' and comet_variant) else metric
        title = f"Average Rankings for {title_label}{metric_label} Score"
    latex.append(f"\\caption{{{escape_latex(title)}}}")
    label = f"avg_rankings_{metric.lower()}"
    if score_type:
        label += f"_{score_type}"
    if comet_variant:
        safe_comet = comet_variant.replace('-', '_').replace(' ', '_').lower()
        label += f"_{safe_comet}"
    latex.append(f"\\label{{tab:{label}}}")
    # Use 3pt spacing between all columns (removed Rank column)
    latex.append("\\begin{tabular}{l@{\\hspace{3pt}}c@{\\hspace{3pt}}c@{\\hspace{3pt}}c@{\\hspace{3pt}}c}")
    latex.append("\\toprule")
    latex.append("Model & Size & Wins & Avg. Rank \\\\")
    latex.append("\\midrule")
    
    for model, avg_rank in sorted_models:
        count = model_counts.get(model, 0)
        wins = model_wins.get(model, 0)
        model_size = get_model_size(model)
        size_display = model_size if model_size else "---"
        latex.append(f"{escape_latex(model)} & {size_display} & {wins} & {avg_rank:.1f} ({count} test sets) \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_overall_scores_table_latex(overall_scores: Dict[str, Dict], models: List[str],
                                       direction: str, score_type: Optional[str] = None,
                                       metrics: List[str] = None, comet_variants: List[str] = None) -> str:
    """
    Generate LaTeX table for overall scores (all test sets concatenated).
    
    Args:
        overall_scores: Dict mapping model_name -> overall scores dict
        models: List of model names
        direction: Translation direction ('forward', 'reverse', 'roundtrip')
        score_type: For forward direction, 'arabic_general' or 'dialect' (None for reverse/roundtrip)
        metrics: List of metrics to include (default: ['BLEU', 'CHRF'])
        comet_variants: List of COMET variant keys to include
    """
    if metrics is None:
        metrics = ['BLEU', 'CHRF']
    
    if not overall_scores:
        return f"% No overall scores found for {direction} direction"
    
    # Group by category
    by_category = defaultdict(list)
    for model_name in models:
        if model_name not in overall_scores:
            continue
        category = categorize_model(model_name)
        by_category[category].append((model_name, overall_scores[model_name]))
    
    # Sort categories
    category_order = ['Arabic-Specialized', 'Multilingual', 'Commercial API', 'Other']
    
    # Determine all metrics to include
    all_metrics = list(metrics)
    if comet_variants:
        all_metrics.extend(comet_variants)
    
    # Determine title and label
    if direction == 'forward' and score_type:
        title_label = 'General Arabic' if score_type == 'arabic_general' else 'Dialect'
        title = f"Overall Scores (All Test Sets Concatenated): Forward Translation ({title_label})"
        label = f"overall_scores_forward_{score_type}"
    elif direction == 'reverse':
        title = "Overall Scores (All Test Sets Concatenated): Reverse Translation"
        label = "overall_scores_reverse"
    elif direction == 'roundtrip':
        title = "Overall Scores (All Test Sets Concatenated): Roundtrip Translation"
        label = "overall_scores_roundtrip"
    else:
        title = f"Overall Scores (All Test Sets Concatenated): {direction.title()}"
        label = f"overall_scores_{direction}"
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{escape_latex(title)}}}")
    latex.append(f"\\label{{tab:{label}}}")
    # Use 3pt spacing between all columns
    col_spec = "l@{\\hspace{3pt}}c"  # Model and Size columns
    for metric in all_metrics:
        col_spec += "@{\\hspace{3pt}}c"
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header
    header = "Model & Size & " + " & ".join([escape_latex(m) for m in all_metrics]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for category in category_order:
        if category not in by_category:
            continue
        
        # Category header
        latex.append(f"\\multicolumn{{{2 + len(all_metrics)}}}{{l}}{{\\textit{{{escape_latex(category)}}}}} \\\\")
        latex.append("\\midrule")
        
        # Models in category
        models_in_cat = sorted(by_category[category], key=lambda x: x[0])
        for model_name, model_scores in models_in_cat:
            model_size = get_model_size(model_name)
            size_display = model_size if model_size else "---"
            row = f"{escape_latex(model_name)} & {size_display} & "
            
            # Extract scores based on direction and score_type
            scores_dict = {}
            if direction == 'forward' and score_type:
                if score_type in model_scores:
                    scores_dict = model_scores[score_type]
            elif direction == 'reverse':
                scores_dict = model_scores
            elif direction == 'roundtrip':
                if 'roundtrip' in model_scores:
                    scores_dict = model_scores['roundtrip']
                else:
                    scores_dict = model_scores
            
            for metric in all_metrics:
                val = scores_dict.get(metric)
                row += f"{format_number(val, decimals=1)} & "
            
            row = row.rstrip(" & ") + " \\\\"
            latex.append(row)
        
        latex.append("\\midrule")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_testset_ranking_table_latex(results_by_direction: Dict, direction: str = 'forward', 
                                          score_type: Optional[str] = None, 
                                          use_merged: bool = False,
                                          metrics: List[str] = None) -> str:
    """
    Generate LaTeX table ranking test sets/dialects by difficulty (average scores).
    
    Args:
        results_by_direction: Dictionary with direction -> {file_test_sets, merged_test_sets, ...}
        direction: Translation direction ('forward' or 'reverse')
        score_type: For forward direction, 'arabic_general' or 'dialect' (None for reverse)
        use_merged: If True, use merged_test_sets; if False, use file_test_sets
        metrics: List of metrics to include (default: ['BLEU', 'CHRF'])
    
    Returns:
        LaTeX table string
    """
    if metrics is None:
        metrics = ['BLEU', 'CHRF']
    
    if direction not in results_by_direction:
        return "% No data available for this direction"
    
    direction_data = results_by_direction[direction]
    
    # Select which data source to use
    if use_merged:
        test_sets_data = direction_data.get('merged_test_sets', {})
        data_type = "Merged Dialects"
    else:
        test_sets_data = direction_data.get('file_test_sets', {})
        data_type = "Test Sets"
    
    if not test_sets_data:
        return f"% No {data_type.lower()} available for this direction"
    
    # Collect all test sets/dialects with their scores
    testset_scores = defaultdict(lambda: defaultdict(list))  # testset -> metric -> [scores]
    
    # Process test sets/dialects
    for test_set, model_scores in test_sets_data.items():
        for model, scores in model_scores.items():
            for metric in metrics:
                score = None
                if direction == 'forward' and score_type:
                    # Forward direction with score_type
                    score_key = f'{score_type}_{metric}'
                    score = scores.get(score_key)
                else:
                    # Reverse direction or no score_type
                    score = scores.get(metric)
                
                if score is not None:
                    testset_scores[test_set][metric].append(score)
    
    if not testset_scores:
        return f"% No {data_type.lower()} scores available"
    
    # Compute average scores for each test set/dialect
    testset_averages = []
    for testset_name, metric_scores in testset_scores.items():
        avg_scores = {}
        for metric in metrics:
            scores_list = metric_scores.get(metric, [])
            if scores_list:
                avg_scores[metric] = np.mean(scores_list)
            else:
                avg_scores[metric] = None
        
        # Use average of all metrics for ranking (or BLEU if available)
        ranking_score = avg_scores.get('BLEU') or avg_scores.get('CHRF') or 0.0
        if ranking_score > 0:
            testset_averages.append((testset_name, avg_scores, ranking_score))
    
    # Sort by ranking score (descending - higher scores = easier)
    testset_averages.sort(key=lambda x: x[2], reverse=True)
    
    # Generate LaTeX table
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    
    title_parts = []
    if use_merged:
        title_parts.append("Merged Dialects")
    else:
        title_parts.append("Test Sets")
    
    if direction == 'forward':
        if score_type == 'arabic_general':
            title_parts.append("General Arabic")
        elif score_type == 'dialect':
            title_parts.append("Dialect")
        title_parts.append("Forward Translation")
    else:
        title_parts.append("Reverse Translation")
    title_parts.append("Rankings")
    
    title = ": ".join(title_parts)
    latex.append(f"\\caption{{{escape_latex(title)}}}")
    
    label = f"testset_ranking_{direction}"
    if use_merged:
        label += "_merged"
    if score_type:
        label += f"_{score_type}"
    latex.append(f"\\label{{tab:{label}}}")
    
    # Use 3pt spacing between all columns
    col_spec = "l@{\\hspace{3pt}}c"  # Rank and Test Set/Dialect columns
    for metric in metrics:
        col_spec += "@{\\hspace{3pt}}c"
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header
    header = "Rank & Test Set/Dialect & " + " & ".join([escape_latex(m) for m in metrics]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for rank, (testset_name, avg_scores, _) in enumerate(testset_averages, 1):
        row = f"{rank} & {escape_latex(testset_name)} & "
        for metric in metrics:
            score = avg_scores.get(metric)
            if score is not None:
                row += f"{format_number(score, decimals=1)} & "
            else:
                row += "--- & "
        row = row.rstrip(" & ") + " \\\\"
        latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def generate_roundtrip_table_latex(roundtrip_results: Dict, metrics: List[str] = None) -> str:
    """Generate LaTeX table for roundtrip translation results."""
    if metrics is None:
        metrics = ['BLEU', 'CHRF']
    
    if not roundtrip_results:
        return "% No roundtrip results available"
    
    # Sort by BLEU score (descending)
    sorted_pairs = sorted(roundtrip_results.items(), 
                         key=lambda x: x[1]['scores'].get('BLEU', 0), 
                         reverse=True)
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Roundtrip Translation Results (English → Arabic → English): Top Model Pairs}")
    latex.append("\\label{tab:roundtrip}")
    # Use 3pt spacing between all columns
    col_spec = "l"  # Model Pair column
    for metric in metrics:
        col_spec += "@{\\hspace{3pt}}c"
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header
    header = "Model Pair & " + " & ".join([escape_latex(m) for m in metrics]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows (top 15)
    for model_pair, results in sorted_pairs[:15]:
        # Clean up model pair name
        pair_name = model_pair.replace('_', ' ').replace('-', ' ')
        row = f"{escape_latex(pair_name)} & "
        for metric in metrics:
            val = results['scores'].get(metric)
            row += f"{format_number(val)} & "
        row = row.rstrip(" & ") + " \\\\"
        latex.append(row)
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def find_key_figures(visualizations_dir: Path) -> Dict[str, List[Tuple]]:
    """Find key figures from visualizations directory."""
    figures = {
        'forward': [],
        'reverse': [],
        'roundtrip': []
    }
    
    # Forward visualizations - prioritize combined/overview plots
    forward_dir = visualizations_dir / 'forward'
    if forward_dir.exists():
        # Combined plots (highest priority)
        for subdir in ['arabic_general', 'dialect']:
            subdir_path = forward_dir / subdir
            if subdir_path.exists():
                combined_bleu = subdir_path / f'all_testsets_{subdir}_bleu_combined.png'
                combined_chrf = subdir_path / f'all_testsets_{subdir}_chrf_combined.png'
                if combined_bleu.exists():
                    figures['forward'].append(('combined', f'forward/{subdir}/all_testsets_{subdir}_bleu_combined.png'))
                if combined_chrf.exists():
                    figures['forward'].append(('combined', f'forward/{subdir}/all_testsets_{subdir}_chrf_combined.png'))
        
        # Dialect-specific plots from merged directory (medium priority)
        for subdir in ['arabic_general', 'dialect']:
            subdir_path = forward_dir / subdir
            if subdir_path.exists():
                merged_dir = subdir_path / 'merged'
                if merged_dir.exists():
                    for fig_file in sorted(merged_dir.glob('dialect_*.png')):
                        figures['forward'].append(('dialect', f'forward/{subdir}/merged/{fig_file.name}'))
    
    # Reverse visualizations - prioritize combined plots
    reverse_dir = visualizations_dir / 'reverse'
    if reverse_dir.exists():
        # Prioritize combined/overview plots
        combined_files = []
        other_files = []
        for fig_file in sorted(reverse_dir.glob('*.png')):
            if 'all_testsets' in fig_file.name or 'combined' in fig_file.name:
                combined_files.append(('combined', f'reverse/{fig_file.name}'))
            else:
                other_files.append(('other', f'reverse/{fig_file.name}'))
        figures['reverse'] = combined_files + other_files[:3]  # Limit others
    
    # Roundtrip visualizations - prioritize combined plots
    roundtrip_dir = visualizations_dir / 'roundtrip'
    if roundtrip_dir.exists():
        for subdir in sorted(roundtrip_dir.iterdir()):
            if subdir.is_dir():
                for fig_file in sorted(subdir.glob('*.png')):
                    if 'all_testsets' in fig_file.name or 'combined' in fig_file.name:
                        figures['roundtrip'].append(('combined', f'roundtrip/{subdir.name}/{fig_file.name}'))
    
    return figures


def generate_figure_latex(figures: Dict[str, List[Tuple]], base_path: str = 'visualizations_fixed') -> str:
    """Generate LaTeX code for including figures."""
    latex = []
    
    # Forward figures
    if figures['forward']:
        latex.append("\\subsection{Forward Translation Visualizations}")
        
        # Separate combined and dialect-specific
        combined_figs = [f for f in figures['forward'] if f[0] == 'combined']
        dialect_figs = [f for f in figures['forward'] if f[0] == 'dialect']
        
        # Combined overview figures
        if combined_figs:
            latex.append("\\subsubsection{Overall Performance}")
            for fig_type, fig_path in combined_figs[:4]:  # Top 4 combined
                fig_name = Path(fig_path).stem.replace('_', ' ').title()
                label = Path(fig_path).stem.replace(' ', '_').replace('-', '_').lower()
                metric = 'BLEU' if 'bleu' in fig_path.lower() else 'CHRF'
                score_type = 'Arabic General' if 'arabic_general' in fig_path else 'Dialect'
                latex.append(f"\\begin{{figure}}[htbp]")
                latex.append(f"\\centering")
                latex.append(f"\\includegraphics[width=0.9\\textwidth]{{{base_path}/{fig_path}}}")
                latex.append(f"\\caption{{{score_type} translation performance across all models ({metric})}}")
                latex.append(f"\\label{{fig:{label}}}")
                latex.append(f"\\end{{figure}}")
                latex.append("")
        
        # Dialect-specific figures (sample)
        if dialect_figs:
            latex.append("\\subsubsection{Dialect-Specific Performance}")
            # Select a few representative dialects
            selected_dialects = ['Modern Standard Arabic', 'Moroccan Arabic', 'Egyptian Arabic', 'Iraqi Arabic']
            for fig_type, fig_path in dialect_figs[:8]:  # Limit to 8
                fig_name = Path(fig_path).stem
                # Extract dialect name
                if 'dialect_' in fig_name:
                    dialect = fig_name.replace('dialect_', '').replace('_arabic_general', '').replace('_dialect', '').replace('_', ' ')
                    if any(d in dialect for d in selected_dialects) or len(dialect_figs) <= 8:
                        label = fig_name.replace(' ', '_').replace('-', '_').lower()
                        metric = 'BLEU' if 'bleu' in fig_path.lower() else 'CHRF'
                        score_type = 'Arabic General' if 'arabic_general' in fig_path else 'Dialect'
                        latex.append(f"\\begin{{figure}}[htbp]")
                        latex.append(f"\\centering")
                        latex.append(f"\\includegraphics[width=0.8\\textwidth]{{{base_path}/{fig_path}}}")
                        latex.append(f"\\caption{{{dialect} translation performance ({score_type}, {metric})}}")
                        latex.append(f"\\label{{fig:{label}}}")
                        latex.append(f"\\end{{figure}}")
                        latex.append("")
    
    # Reverse figures
    if figures['reverse']:
        latex.append("\\subsection{Reverse Translation Visualizations}")
        for fig_type, fig_path in figures['reverse'][:4]:  # Limit to top 4
            fig_name = Path(fig_path).stem.replace('_', ' ').title()
            label = Path(fig_path).stem.replace(' ', '_').replace('-', '_').lower()
            metric = 'BLEU' if 'bleu' in fig_path.lower() else 'CHRF'
            latex.append(f"\\begin{{figure}}[htbp]")
            latex.append(f"\\centering")
            latex.append(f"\\includegraphics[width=0.9\\textwidth]{{{base_path}/{fig_path}}}")
            latex.append(f"\\caption{{Reverse translation performance across all models ({metric})}}")
            latex.append(f"\\label{{fig:{label}}}")
            latex.append(f"\\end{{figure}}")
            latex.append("")
    
    # Roundtrip figures
    if figures['roundtrip']:
        latex.append("\\subsection{Roundtrip Translation Visualizations}")
        for fig_type, fig_path in figures['roundtrip'][:4]:  # Limit to top 4
            fig_name = Path(fig_path).stem.replace('_', ' ').title()
            label = Path(fig_path).stem.replace(' ', '_').replace('-', '_').lower()
            metric = 'BLEU' if 'bleu' in fig_path.lower() else 'CHRF'
            model_pair = Path(fig_path).parent.name
            latex.append(f"\\begin{{figure}}[htbp]")
            latex.append(f"\\centering")
            latex.append(f"\\includegraphics[width=0.9\\textwidth]{{{base_path}/{fig_path}}}")
            latex.append(f"\\caption{{Roundtrip translation performance for {escape_latex(model_pair)} ({metric})}}")
            latex.append(f"\\label{{fig:{label}}}")
            latex.append(f"\\end{{figure}}")
            latex.append("")
    
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables and figures from results')
    parser.add_argument('--fixed-translations-dir', type=str, 
                       default='fixed_translations',
                       help='Directory containing fixed translation results')
    parser.add_argument('--visualizations-dir', type=str,
                       default='visualizations_fixed',
                       help='Directory containing visualization figures')
    parser.add_argument('--output-dir', type=str,
                       default='paper_tables',
                       help='Output directory for generated LaTeX files')
    parser.add_argument('--include-comet', action='store_true',
                       help='Include COMET scores in tables')
    
    args = parser.parse_args()
    
    fixed_translations_dir = Path(args.fixed_translations_dir)
    visualizations_dir = Path(args.visualizations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Processing forward translations...")
    forward_results = process_forward_translations(fixed_translations_dir)
    print(f"Found {len(forward_results)} forward translation models")
    
    print("Processing reverse translations...")
    reverse_results = process_reverse_translations(fixed_translations_dir)
    print(f"Found {len(reverse_results)} reverse translation models")
    
    print("Processing roundtrip translations...")
    roundtrip_results = process_roundtrip_translations(fixed_translations_dir)
    print(f"Found {len(roundtrip_results)} roundtrip translation pairs")
    
    print("Finding figures...")
    figures = find_key_figures(visualizations_dir)
    print(f"Found {len(figures['forward'])} forward, {len(figures['reverse'])} reverse, {len(figures['roundtrip'])} roundtrip figures")
    
    # Determine metrics
    metrics = ['BLEU', 'CHRF']
    if args.include_comet:
        # Find COMET keys from first result
        if forward_results:
            first_result = list(forward_results.values())[0]
            comet_keys = []
            for key in first_result['arabic_general'].keys():
                if key.startswith('COMET_'):
                    comet_keys.append(key)
            metrics.extend(comet_keys[:2])  # Add up to 2 COMET variants
    
    # Generate tables
    print("Generating LaTeX tables...")
    forward_table = generate_forward_table_latex(forward_results, metrics)
    with open(output_dir / 'forward_table.tex', 'w') as f:
        f.write(forward_table)
    
    dialect_table = generate_dialect_table_latex(forward_results, metrics)
    with open(output_dir / 'dialect_table.tex', 'w') as f:
        f.write(dialect_table)
    
    prompt_comparison_table = generate_prompt_comparison_table_latex(forward_results, metrics)
    with open(output_dir / 'prompt_comparison_table.tex', 'w') as f:
        f.write(prompt_comparison_table)
    
    dialect_prompt_comparison_table = generate_dialect_prompt_comparison_table_latex(forward_results, metrics)
    with open(output_dir / 'dialect_prompt_comparison_table.tex', 'w') as f:
        f.write(dialect_prompt_comparison_table)
    
    # Generate per-model dialect prompt comparison tables
    print("Generating per-model dialect prompt comparison tables...")
    dialect_breakdown = forward_results.get('_dialect_breakdown', {})
    if dialect_breakdown:
        # Get all unique models from dialect breakdown
        all_models = set()
        for models_data in dialect_breakdown.values():
            all_models.update(models_data.keys())
        
        for model_name in sorted(all_models):
            model_table = generate_model_dialect_prompt_comparison_table_latex(forward_results, model_name, metrics)
            safe_model_name = model_name.replace(' ', '_').replace('-', '_').replace('/', '_').lower()
            filename = f'model_dialect_prompt_comparison_{safe_model_name}.tex'
            with open(output_dir / filename, 'w') as f:
                f.write(model_table)
    
    reverse_table = generate_reverse_table_latex(reverse_results, metrics)
    with open(output_dir / 'reverse_table.tex', 'w') as f:
        f.write(reverse_table)
    
    roundtrip_table = generate_roundtrip_table_latex(roundtrip_results, metrics)
    with open(output_dir / 'roundtrip_table.tex', 'w') as f:
        f.write(roundtrip_table)
    
    # Generate overall scores tables
    if collect_all_scores:
        print("Generating overall scores tables...")
        
        # Find all scores.json files
        scores_files = []
        for path in fixed_translations_dir.rglob("scores.json"):
            scores_files.append(path)
        
        if scores_files:
            # Collect all scores
            results_by_direction, all_models = collect_all_scores(scores_files)
            
            # Forward direction - overall scores by score type
            if 'forward' in results_by_direction:
                forward_data = results_by_direction['forward']
                overall_scores = forward_data.get('overall_scores', {})
                
                if overall_scores:
                    # Combined table with all metrics
                    table = generate_overall_scores_table_latex(
                        overall_scores, all_models, 'forward', 'arabic_general',
                        metrics=['BLEU', 'CHRF']
                    )
                    filename = 'forward_overall_scores_arabic_general.tex'
                    with open(output_dir / filename, 'w') as f:
                        f.write(table)
                    
                    table = generate_overall_scores_table_latex(
                        overall_scores, all_models, 'forward', 'dialect',
                        metrics=['BLEU', 'CHRF']
                    )
                    filename = 'forward_overall_scores_dialect.tex'
                    with open(output_dir / filename, 'w') as f:
                        f.write(table)
                    
                    # COMET variants for regular overall scores
                    if args.include_comet:
                        comet_variants = set()
                        for model_scores in overall_scores.values():
                            for score_type_key in ['arabic_general', 'dialect']:
                                if score_type_key in model_scores:
                                    for key in model_scores[score_type_key].keys():
                                        if isinstance(key, str) and key.startswith('COMET_'):
                                            comet_variants.add(key)
                        
                        for comet_variant in sorted(comet_variants):
                            for score_type in ['arabic_general', 'dialect']:
                                table = generate_overall_scores_table_latex(
                                    overall_scores, all_models, 'forward', score_type,
                                    metrics=['BLEU', 'CHRF'], comet_variants=[comet_variant]
                                )
                                safe_comet = comet_variant.replace('-', '_').replace(' ', '_').lower()
                                filename = f'forward_overall_scores_{score_type}_comet_{safe_comet}.tex'
                                with open(output_dir / filename, 'w') as f:
                                    f.write(table)
                
                # MADAR-only overall scores
                overall_scores_madar_only = forward_data.get('overall_scores_madar_only', {})
                if overall_scores_madar_only:
                    table = generate_overall_scores_table_latex(
                        overall_scores_madar_only, all_models, 'forward', 'arabic_general',
                        metrics=['BLEU', 'CHRF']
                    )
                    filename = 'forward_overall_scores_madar_only_arabic_general.tex'
                    with open(output_dir / filename, 'w') as f:
                        f.write(table)
                    
                    table = generate_overall_scores_table_latex(
                        overall_scores_madar_only, all_models, 'forward', 'dialect',
                        metrics=['BLEU', 'CHRF']
                    )
                    filename = 'forward_overall_scores_madar_only_dialect.tex'
                    with open(output_dir / filename, 'w') as f:
                        f.write(table)
                    
                    # COMET variants for MADAR-only
                    if args.include_comet:
                        # Find COMET variants from MADAR-only overall scores
                        comet_variants = set()
                        for model_scores in overall_scores_madar_only.values():
                            for score_type_key in ['arabic_general', 'dialect']:
                                if score_type_key in model_scores:
                                    for key in model_scores[score_type_key].keys():
                                        if isinstance(key, str) and key.startswith('COMET_'):
                                            comet_variants.add(key)
                        
                        for comet_variant in sorted(comet_variants):
                            for score_type in ['arabic_general', 'dialect']:
                                table = generate_overall_scores_table_latex(
                                    overall_scores_madar_only, all_models, 'forward', score_type,
                                    metrics=['BLEU', 'CHRF'], comet_variants=[comet_variant]
                                )
                                safe_comet = comet_variant.replace('-', '_').replace(' ', '_').lower()
                                filename = f'forward_overall_scores_madar_only_{score_type}_comet_{safe_comet}.tex'
                                with open(output_dir / filename, 'w') as f:
                                    f.write(table)
            
            # Reverse direction - overall scores
            if 'reverse' in results_by_direction:
                reverse_data = results_by_direction['reverse']
                overall_scores = reverse_data.get('overall_scores', {})
                
                if overall_scores:
                    # Combined table with all metrics
                    table = generate_overall_scores_table_latex(
                        overall_scores, all_models, 'reverse', None,
                        metrics=['BLEU', 'CHRF']
                    )
                    filename = 'reverse_overall_scores.tex'
                    with open(output_dir / filename, 'w') as f:
                        f.write(table)
                    
                    # COMET variants
                    if args.include_comet:
                        comet_variants = set()
                        for model_scores in overall_scores.values():
                            for key in model_scores.keys():
                                if isinstance(key, str) and key.startswith('COMET_'):
                                    comet_variants.add(key)
                        
                        for comet_variant in sorted(comet_variants):
                            table = generate_overall_scores_table_latex(
                                overall_scores, all_models, 'reverse', None,
                                metrics=['BLEU', 'CHRF'], comet_variants=[comet_variant]
                            )
                            safe_comet = comet_variant.replace('-', '_').replace(' ', '_').lower()
                            filename = f'reverse_overall_scores_comet_{safe_comet}.tex'
                            with open(output_dir / filename, 'w') as f:
                                f.write(table)
                
                # MADAR-only overall scores
                overall_scores_madar_only = reverse_data.get('overall_scores_madar_only', {})
                if overall_scores_madar_only:
                    table = generate_overall_scores_table_latex(
                        overall_scores_madar_only, all_models, 'reverse', None,
                        metrics=['BLEU', 'CHRF']
                    )
                    filename = 'reverse_overall_scores_madar_only.tex'
                    with open(output_dir / filename, 'w') as f:
                        f.write(table)
                    
                    # COMET variants for MADAR-only
                    if args.include_comet:
                        comet_variants = set()
                        for model_scores in overall_scores_madar_only.values():
                            for key in model_scores.keys():
                                if isinstance(key, str) and key.startswith('COMET_'):
                                    comet_variants.add(key)
                        
                        for comet_variant in sorted(comet_variants):
                            table = generate_overall_scores_table_latex(
                                overall_scores_madar_only, all_models, 'reverse', None,
                                metrics=['BLEU', 'CHRF'], comet_variants=[comet_variant]
                            )
                            safe_comet = comet_variant.replace('-', '_').replace(' ', '_').lower()
                            filename = f'reverse_overall_scores_madar_only_comet_{safe_comet}.tex'
                            with open(output_dir / filename, 'w') as f:
                                f.write(table)
    else:
        print("⚠️  Skipping overall scores tables (functions not available)")
    
    # Generate average ranking tables
    if collect_all_scores and calculate_average_rankings:
        print("Generating average ranking tables...")
        
        # Find all scores.json files
        scores_files = []
        for path in fixed_translations_dir.rglob("scores.json"):
            scores_files.append(path)
        
        if scores_files:
            # Collect all scores
            results_by_direction, all_models = collect_all_scores(scores_files)
            
            # Forward direction - overall and by score type
            if 'forward' in results_by_direction:
                forward_data = results_by_direction['forward']
                file_test_sets = forward_data.get('file_test_sets', {})
                
                if file_test_sets:
                    # Overall rankings (no score_type)
                    for metric in ['BLEU', 'CHRF']:
                        table = generate_average_rankings_table_latex(
                            file_test_sets, all_models, None, metric,
                            title=f"Average Rankings: Forward Translation ({metric})"
                        )
                        filename = f'forward_avg_rankings_{metric.lower()}.tex'
                        with open(output_dir / filename, 'w') as f:
                            f.write(table)
                    
                    # By score type (arabic_general and dialect)
                    for score_type in ['arabic_general', 'dialect']:
                        for metric in ['BLEU', 'CHRF']:
                            table = generate_average_rankings_table_latex(
                                file_test_sets, all_models, score_type, metric,
                                title=f"Average Rankings: Forward Translation ({score_type}, {metric})"
                            )
                            filename = f'forward_avg_rankings_{score_type}_{metric.lower()}.tex'
                            with open(output_dir / filename, 'w') as f:
                                f.write(table)
                    
                    # COMET variants
                    if args.include_comet:
                        comet_variants = find_all_comet_variants(file_test_sets, all_models)
                        for comet_variant in comet_variants:
                            # Overall
                            table = generate_average_rankings_table_latex(
                                file_test_sets, all_models, None, 'COMET', comet_variant,
                                title=f"Average Rankings: Forward Translation ({comet_variant})"
                            )
                            safe_comet = comet_variant.replace('-', '_').replace(' ', '_').lower()
                            filename = f'forward_avg_rankings_comet_{safe_comet}.tex'
                            with open(output_dir / filename, 'w') as f:
                                f.write(table)
                            
                            # By score type
                            for score_type in ['arabic_general', 'dialect']:
                                table = generate_average_rankings_table_latex(
                                    file_test_sets, all_models, score_type, 'COMET', comet_variant,
                                    title=f"Average Rankings: Forward Translation ({score_type}, {comet_variant})"
                                )
                                filename = f'forward_avg_rankings_{score_type}_comet_{safe_comet}.tex'
                                with open(output_dir / filename, 'w') as f:
                                    f.write(table)
            
            # Reverse direction
            if 'reverse' in results_by_direction:
                reverse_data = results_by_direction['reverse']
                file_test_sets = reverse_data.get('file_test_sets', {})
                
                if file_test_sets:
                    for metric in ['BLEU', 'CHRF']:
                        table = generate_average_rankings_table_latex(
                            file_test_sets, all_models, None, metric,
                            title=f"Average Rankings: Reverse Translation ({metric})"
                        )
                        filename = f'reverse_avg_rankings_{metric.lower()}.tex'
                        with open(output_dir / filename, 'w') as f:
                            f.write(table)
                    
                    # COMET variants
                    if args.include_comet:
                        comet_variants = find_all_comet_variants(file_test_sets, all_models)
                        for comet_variant in comet_variants:
                            table = generate_average_rankings_table_latex(
                                file_test_sets, all_models, None, 'COMET', comet_variant,
                                title=f"Average Rankings: Reverse Translation ({comet_variant})"
                            )
                            safe_comet = comet_variant.replace('-', '_').replace(' ', '_').lower()
                            filename = f'reverse_avg_rankings_comet_{safe_comet}.tex'
                            with open(output_dir / filename, 'w') as f:
                                f.write(table)
            
            # Merged dialects (regular)
            if 'forward' in results_by_direction:
                forward_data = results_by_direction['forward']
                merged_test_sets = forward_data.get('merged_test_sets', {})
                
                # Separate regular and madar_only merged scores
                merged_test_sets_regular = {k: v for k, v in merged_test_sets.items() if not k.endswith('')}
                merged_test_sets_madar_only = {k: v for k, v in merged_test_sets.items() if k.endswith('')}
                
                if merged_test_sets_regular:
                    for score_type in ['arabic_general', 'dialect']:
                        for metric in ['BLEU', 'CHRF']:
                            table = generate_average_rankings_table_latex(
                                merged_test_sets_regular, all_models, score_type, metric,
                                title=f"Average Rankings: Merged Dialects ({score_type}, {metric})"
                            )
                            filename = f'forward_merged_avg_rankings_{score_type}_{metric.lower()}.tex'
                            with open(output_dir / filename, 'w') as f:
                                f.write(table)
                
                # MADAR-only merged dialects
                if merged_test_sets_madar_only:
                    for score_type in ['arabic_general', 'dialect']:
                        for metric in ['BLEU', 'CHRF']:
                            table = generate_average_rankings_table_latex(
                                merged_test_sets_madar_only, all_models, score_type, metric,
                                title=f"Average Rankings: Merged Dialects - MADAR Only ({score_type}, {metric})"
                            )
                            filename = f'forward_merged_madar_only_avg_rankings_{score_type}_{metric.lower()}.tex'
                            with open(output_dir / filename, 'w') as f:
                                f.write(table)
            
            if 'reverse' in results_by_direction:
                reverse_data = results_by_direction['reverse']
                merged_test_sets = reverse_data.get('merged_test_sets', {})
                
                # Separate regular and madar_only merged scores
                merged_test_sets_regular = {k: v for k, v in merged_test_sets.items() if not k.endswith('')}
                merged_test_sets_madar_only = {k: v for k, v in merged_test_sets.items() if k.endswith('')}
                
                if merged_test_sets_regular:
                    for metric in ['BLEU', 'CHRF']:
                        table = generate_average_rankings_table_latex(
                            merged_test_sets_regular, all_models, None, metric,
                            title=f"Average Rankings: Merged Dialects - Reverse ({metric})"
                        )
                        filename = f'reverse_merged_avg_rankings_{metric.lower()}.tex'
                        with open(output_dir / filename, 'w') as f:
                            f.write(table)
                
                # MADAR-only merged dialects
                if merged_test_sets_madar_only:
                    for metric in ['BLEU', 'CHRF']:
                        table = generate_average_rankings_table_latex(
                            merged_test_sets_madar_only, all_models, None, metric,
                            title=f"Average Rankings: Merged Dialects - Reverse, MADAR Only ({metric})"
                        )
                        filename = f'reverse_merged_madar_only_avg_rankings_{metric.lower()}.tex'
                        with open(output_dir / filename, 'w') as f:
                            f.write(table)
    else:
        print("⚠️  Skipping average ranking tables (functions not available)")
    
    # Generate test set/dialect ranking tables
    if collect_all_scores and results_by_direction:
        print("\nGenerating test set/dialect ranking tables...")
        
        # Forward direction - file test sets
        if 'forward' in results_by_direction:
            forward_data = results_by_direction['forward']
            file_test_sets = forward_data.get('file_test_sets', {})
            
            if file_test_sets:
                # General Arabic
                table = generate_testset_ranking_table_latex(
                    results_by_direction, 'forward', 'arabic_general', use_merged=False
                )
                filename = 'forward_testset_ranking_arabic_general.tex'
                with open(output_dir / filename, 'w') as f:
                    f.write(table)
                print(f"   ✅ {filename}")
                
                # Dialect
                table = generate_testset_ranking_table_latex(
                    results_by_direction, 'forward', 'dialect', use_merged=False
                )
                filename = 'forward_testset_ranking_dialect.tex'
                with open(output_dir / filename, 'w') as f:
                    f.write(table)
                print(f"   ✅ {filename}")
            
            # Merged dialects (regular)
            merged_test_sets = forward_data.get('merged_test_sets', {})
            # Separate regular and madar_only
            merged_test_sets_regular = {k: v for k, v in merged_test_sets.items() if not k.endswith('')}
            merged_test_sets_madar_only = {k: v for k, v in merged_test_sets.items() if k.endswith('')}
            
            if merged_test_sets_regular:
                # Temporarily replace merged_test_sets with regular only for table generation
                forward_data['merged_test_sets'] = merged_test_sets_regular
                # General Arabic
                table = generate_testset_ranking_table_latex(
                    results_by_direction, 'forward', 'arabic_general', use_merged=True
                )
                filename = 'forward_merged_dialect_ranking_arabic_general.tex'
                with open(output_dir / filename, 'w') as f:
                    f.write(table)
                print(f"   ✅ {filename}")
                
                # Dialect
                table = generate_testset_ranking_table_latex(
                    results_by_direction, 'forward', 'dialect', use_merged=True
                )
                filename = 'forward_merged_dialect_ranking_dialect.tex'
                with open(output_dir / filename, 'w') as f:
                    f.write(table)
                print(f"   ✅ {filename}")
                # Restore original merged_test_sets
                forward_data['merged_test_sets'] = merged_test_sets
            
            # MADAR-only merged dialects
            if merged_test_sets_madar_only:
                # Temporarily replace merged_test_sets with madar_only for table generation
                forward_data['merged_test_sets'] = merged_test_sets_madar_only
                # General Arabic
                table = generate_testset_ranking_table_latex(
                    results_by_direction, 'forward', 'arabic_general', use_merged=True
                )
                filename = 'forward_merged_madar_only_dialect_ranking_arabic_general.tex'
                with open(output_dir / filename, 'w') as f:
                    f.write(table)
                print(f"   ✅ {filename}")
                
                # Dialect
                table = generate_testset_ranking_table_latex(
                    results_by_direction, 'forward', 'dialect', use_merged=True
                )
                filename = 'forward_merged_madar_only_dialect_ranking_dialect.tex'
                with open(output_dir / filename, 'w') as f:
                    f.write(table)
                print(f"   ✅ {filename}")
                # Restore original merged_test_sets
                forward_data['merged_test_sets'] = merged_test_sets
        
        # Reverse direction
        if 'reverse' in results_by_direction:
            reverse_data = results_by_direction['reverse']
            file_test_sets = reverse_data.get('file_test_sets', {})
            
            if file_test_sets:
                table = generate_testset_ranking_table_latex(
                    results_by_direction, 'reverse', None, use_merged=False
                )
                filename = 'reverse_testset_ranking.tex'
                with open(output_dir / filename, 'w') as f:
                    f.write(table)
                print(f"   ✅ {filename}")
            
            # Merged dialects (regular)
            merged_test_sets = reverse_data.get('merged_test_sets', {})
            # Separate regular and madar_only
            merged_test_sets_regular = {k: v for k, v in merged_test_sets.items() if not k.endswith('')}
            merged_test_sets_madar_only = {k: v for k, v in merged_test_sets.items() if k.endswith('')}
            
            if merged_test_sets_regular:
                # Temporarily replace merged_test_sets with regular only for table generation
                reverse_data['merged_test_sets'] = merged_test_sets_regular
                table = generate_testset_ranking_table_latex(
                    results_by_direction, 'reverse', None, use_merged=True
                )
                filename = 'reverse_merged_dialect_ranking.tex'
                with open(output_dir / filename, 'w') as f:
                    f.write(table)
                print(f"   ✅ {filename}")
                # Restore original merged_test_sets
                reverse_data['merged_test_sets'] = merged_test_sets
            
            # MADAR-only merged dialects
            if merged_test_sets_madar_only:
                # Temporarily replace merged_test_sets with madar_only for table generation
                reverse_data['merged_test_sets'] = merged_test_sets_madar_only
                table = generate_testset_ranking_table_latex(
                    results_by_direction, 'reverse', None, use_merged=True
                )
                filename = 'reverse_merged_madar_only_dialect_ranking.tex'
                with open(output_dir / filename, 'w') as f:
                    f.write(table)
                print(f"   ✅ {filename}")
                # Restore original merged_test_sets
                reverse_data['merged_test_sets'] = merged_test_sets
    
    # Generate figure LaTeX
    print("Generating LaTeX figure code...")
    figure_latex = generate_figure_latex(figures)
    with open(output_dir / 'figures.tex', 'w') as f:
        f.write(figure_latex)
    
    # Generate summary document
    summary = []
    summary.append("\\section{Results Tables and Figures}")
    summary.append("")
    summary.append("\\subsection{Forward Translation Results}")
    summary.append("\\input{paper_tables/forward_table.tex}")
    summary.append("")
    summary.append("\\input{paper_tables/dialect_table.tex}")
    summary.append("")
    summary.append("\\subsection{Reverse Translation Results}")
    summary.append("\\input{paper_tables/reverse_table.tex}")
    summary.append("")
    summary.append("\\subsection{Roundtrip Translation Results}")
    summary.append("\\input{paper_tables/roundtrip_table.tex}")
    summary.append("")
    summary.append("\\subsection{Visualizations}")
    summary.append("\\input{paper_tables/figures.tex}")
    
    with open(output_dir / 'results_section.tex', 'w') as f:
        f.write("\n".join(summary))
    
    print(f"\n✅ Generated LaTeX files in {output_dir}/")
    print("   - forward_table.tex")
    print("   - dialect_table.tex")
    print("   - reverse_table.tex")
    print("   - roundtrip_table.tex")
    print("   - figures.tex")
    print("   - results_section.tex")
    if collect_all_scores:
        print("   - overall_scores tables (forward_*, reverse_*)")
        if calculate_average_rankings:
            print("   - average_ranking tables (forward_*, reverse_*, *_merged_*)")


if __name__ == '__main__':
    main()


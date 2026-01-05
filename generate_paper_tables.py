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
    elif 'aya' in model_lower:
        return 'Multilingual (Arabic-focused)'
    elif 'command' in model_lower:
        return 'Multilingual (Arabic-focused)'
    elif 'gpt-4' in model_lower:
        return 'Commercial API'
    elif 'gemma' in model_lower or 'eurollm' in model_lower or 'mistral' in model_lower or 'qwen' in model_lower or 'llama' in model_lower or 'falcon' in model_lower:
        return 'General Multilingual'
    else:
        return 'Other'


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
        
        # Extract dialect-specific results from merged dialects
        if 'results' in data:
            for result in data['results']:
                if result.get('type') == 'dialect_merged':
                    dialect_code = result.get('dialect_code', '')
                    dialect_name = result.get('dialect_name', dialect_code)
                    
                    if 'arabic_general' in result:
                        dialect_results[dialect_name][model_name] = {
                            'arabic_general': result['arabic_general'].copy(),
                            'dialect': result.get('dialect', {}).copy(),
                            'category': model_category  # Preserve category
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
    category_order = ['Arabic-Specialized', 'Multilingual (Arabic-focused)', 
                     'General Multilingual', 'Commercial API', 'Other']
    
    latex = []
    latex.append("\\begin{table*}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Forward Translation Results (English → Arabic): Overall Performance}")
    latex.append("\\label{tab:forward_overall}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{l" + "c" * len(metrics) * 2 + "}")
    latex.append("\\toprule")
    
    # Header
    header = "Model & "
    for metric in metrics:
        header += f"\\multicolumn{{2}}{{c}}{{{metric}}} & "
    header = header.rstrip(" & ") + " \\\\"
    header += "\\cmidrule(lr){2-3} " if len(metrics) > 0 else ""
    for i, metric in enumerate(metrics):
        if i > 0:
            header += "\\cmidrule(lr){" + str(2 + i*2) + "-" + str(3 + i*2) + "} "
    header += " & General & Dialect " * len(metrics) + "\\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for category in category_order:
        if category not in by_category:
            continue
        
        # Category header
        latex.append(f"\\multicolumn{{{1 + len(metrics)*2}}}{{l}}{{\\textit{{{escape_latex(category)}}}}} \\\\")
        latex.append("\\midrule")
        
        # Models in category
        models_in_cat = sorted(by_category[category], key=lambda x: x[0])
        for model_name, results in models_in_cat:
            row = f"{escape_latex(model_name)} & "
            for metric in metrics:
                ag_val = results['arabic_general'].get(metric)
                dial_val = results['dialect'].get(metric)
                row += f"{format_number(ag_val)} & {format_number(dial_val)} & "
            row = row.rstrip(" & ") + " \\\\"
            latex.append(row)
        
        latex.append("\\midrule")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table*}")
    
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
    latex.append("\\begin{table*}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Forward Translation: Dialect-Specific Performance (Dialect Variant)}")
    latex.append("\\label{tab:forward_dialects}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{l" + "c" * len(metrics) * len(dialects) + "}")
    latex.append("\\toprule")
    
    # Header
    header = "Model & "
    for dialect in dialects:
        header += f"\\multicolumn{{{len(metrics)}}}{{c}}{{{escape_latex(dialect)}}} & "
    header = header.rstrip(" & ") + " \\\\"
    latex.append(header)
    
    # Sub-header for metrics
    subheader = " & "
    for dialect in dialects:
        for metric in metrics:
            subheader += f"{escape_latex(metric)} & "
    subheader = subheader.rstrip(" & ") + " \\\\"
    latex.append(subheader)
    latex.append("\\midrule")
    
    # Data rows
    for model in all_models:
        row = f"{escape_latex(model)} & "
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
    latex.append("}")
    latex.append("\\end{table*}")
    
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
    category_order = ['Arabic-Specialized', 'Multilingual (Arabic-focused)', 
                     'General Multilingual', 'Commercial API', 'Other']
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Reverse Translation Results (Arabic → English): Overall Performance}")
    latex.append("\\label{tab:reverse_overall}")
    latex.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    latex.append("\\toprule")
    
    # Header
    header = "Model & " + " & ".join([escape_latex(m) for m in metrics]) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Data rows
    for category in category_order:
        if category not in by_category:
            continue
        
        # Category header
        latex.append(f"\\multicolumn{{{1 + len(metrics)}}}{{l}}{{\\textit{{{escape_latex(category)}}}}} \\\\")
        latex.append("\\midrule")
        
        # Models in category
        models_in_cat = sorted(by_category[category], key=lambda x: x[0])
        for model_name, results in models_in_cat:
            row = f"{escape_latex(model_name)} & "
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
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("Rank & Model & Avg. Rank \\\\")
    latex.append("\\midrule")
    
    for rank, (model, avg_rank) in enumerate(sorted_models, 1):
        count = model_counts.get(model, 0)
        latex.append(f"{rank} & {escape_latex(model)} & {avg_rank:.1f} ({count} test sets) \\\\")
    
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
    latex.append("\\resizebox{0.9\\textwidth}{!}{")
    latex.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
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
    latex.append("}")
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
    
    reverse_table = generate_reverse_table_latex(reverse_results, metrics)
    with open(output_dir / 'reverse_table.tex', 'w') as f:
        f.write(reverse_table)
    
    roundtrip_table = generate_roundtrip_table_latex(roundtrip_results, metrics)
    with open(output_dir / 'roundtrip_table.tex', 'w') as f:
        f.write(roundtrip_table)
    
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
            
            # Merged dialects
            if 'forward' in results_by_direction:
                forward_data = results_by_direction['forward']
                merged_test_sets = forward_data.get('merged_test_sets', {})
                
                if merged_test_sets:
                    for score_type in ['arabic_general', 'dialect']:
                        for metric in ['BLEU', 'CHRF']:
                            table = generate_average_rankings_table_latex(
                                merged_test_sets, all_models, score_type, metric,
                                title=f"Average Rankings: Merged Dialects ({score_type}, {metric})"
                            )
                            filename = f'forward_merged_avg_rankings_{score_type}_{metric.lower()}.tex'
                            with open(output_dir / filename, 'w') as f:
                                f.write(table)
            
            if 'reverse' in results_by_direction:
                reverse_data = results_by_direction['reverse']
                merged_test_sets = reverse_data.get('merged_test_sets', {})
                
                if merged_test_sets:
                    for metric in ['BLEU', 'CHRF']:
                        table = generate_average_rankings_table_latex(
                            merged_test_sets, all_models, None, metric,
                            title=f"Average Rankings: Merged Dialects - Reverse ({metric})"
                        )
                        filename = f'reverse_merged_avg_rankings_{metric.lower()}.tex'
                        with open(output_dir / filename, 'w') as f:
                            f.write(table)
    else:
        print("⚠️  Skipping average ranking tables (functions not available)")
    
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
    if collect_all_scores and calculate_average_rankings:
        print("   - average_ranking tables (forward_*, reverse_*, *_merged_*)")


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Visualization script for comparing BLEU/CHRF scores across models and test sets.

This script:
1. Finds all scores.json files in result subdirectories
2. Extracts BLEU and CHRF scores for each model and test set
3. Creates bar plots comparing models for each test set
4. Handles both eval_arabench.py and eval_arabench_reverse.py formats
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def find_scores_files(root_dir: Path) -> List[Path]:
    """Find all scores.json files in result subdirectories."""
    scores_files = []
    
    # Look for scores.json in subdirectories
    for path in root_dir.rglob("scores.json"):
        scores_files.append(path)
    
    return sorted(scores_files)


def load_scores_file(scores_path: Path) -> Optional[Dict]:
    """Load and parse a scores.json file."""
    try:
        with open(scores_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading {scores_path}: {e}")
        return None


def extract_model_name(scores_data: Dict, scores_path: Path) -> str:
    """Extract model name from scores data or path."""
    if 'model' in scores_data:
        model = scores_data['model']
        # Clean up model name (remove path prefixes)
        if '/' in model:
            model = model.split('/')[-1]
        return model
    else:
        # Try to infer from directory name
        return scores_path.parent.name


def extract_scores(scores_data: Dict) -> Tuple[Dict, Dict, Optional[Dict]]:
    """
    Extract scores from scores.json data.
    
    Returns: 
        - file_scores: Dict mapping test_set_name -> {metric_type -> score, metadata}
        - merged_scores: Dict mapping dialect_name -> {metric_type -> score, metadata}
        - overall_scores: Dict with overall scores (or None if not present)
    """
    file_scores = {}
    merged_scores = {}
    overall_scores = None
    
    if 'results' not in scores_data:
        return file_scores, merged_scores, overall_scores
    
    for result in scores_data['results']:
        result_type = result.get('type', 'file')  # Default to 'file' for backward compatibility
        
        # Handle overall scores (all test sets concatenated)
        if result_type == 'overall':
            overall_scores = {}
            
            # Forward direction: has arabic_general and dialect
            if 'arabic_general' in result:
                overall_scores['arabic_general'] = {}
                if 'BLEU' in result['arabic_general']:
                    overall_scores['arabic_general']['BLEU'] = result['arabic_general']['BLEU']
                if 'CHRF' in result['arabic_general']:
                    overall_scores['arabic_general']['CHRF'] = result['arabic_general']['CHRF']
            
            if 'dialect' in result:
                overall_scores['dialect'] = {}
                if 'BLEU' in result['dialect']:
                    overall_scores['dialect']['BLEU'] = result['dialect']['BLEU']
                if 'CHRF' in result['dialect']:
                    overall_scores['dialect']['CHRF'] = result['dialect']['CHRF']
            
            # Reverse/Roundtrip: direct BLEU/CHRF
            if 'BLEU' in result and 'CHRF' in result:
                overall_scores['BLEU'] = result['BLEU']
                overall_scores['CHRF'] = result['CHRF']
            
            # Roundtrip: has roundtrip scores
            if 'roundtrip' in result:
                overall_scores['roundtrip'] = {}
                if 'BLEU' in result['roundtrip']:
                    overall_scores['roundtrip']['BLEU'] = result['roundtrip']['BLEU']
                if 'CHRF' in result['roundtrip']:
                    overall_scores['roundtrip']['CHRF'] = result['roundtrip']['CHRF']
            
            # Store metadata
            if 'num_test_sets' in result:
                overall_scores['num_test_sets'] = result['num_test_sets']
            if 'num_sentences' in result:
                overall_scores['num_sentences'] = result['num_sentences']
        
        # Handle merged dialect results
        elif result_type == 'dialect_merged':
            dialect_name = result.get('dialect_name', 'unknown')
            merged_scores[dialect_name] = {}
            
            if 'arabic_general' in result:
                if 'BLEU' in result['arabic_general']:
                    merged_scores[dialect_name]['arabic_general_BLEU'] = result['arabic_general']['BLEU']
                if 'CHRF' in result['arabic_general']:
                    merged_scores[dialect_name]['arabic_general_CHRF'] = result['arabic_general']['CHRF']
            
            if 'dialect' in result:
                if 'BLEU' in result['dialect']:
                    merged_scores[dialect_name]['dialect_BLEU'] = result['dialect']['BLEU']
                if 'CHRF' in result['dialect']:
                    merged_scores[dialect_name]['dialect_CHRF'] = result['dialect']['CHRF']
            
            # Store metadata
            if 'dialect_code' in result:
                merged_scores[dialect_name]['dialect_code'] = result['dialect_code']
            if 'dialect_name' in result:
                merged_scores[dialect_name]['dialect_name'] = result['dialect_name']
            if 'num_test_sets' in result:
                merged_scores[dialect_name]['num_test_sets'] = result['num_test_sets']
            if 'num_sentences' in result:
                merged_scores[dialect_name]['num_sentences'] = result['num_sentences']
            merged_scores[dialect_name]['is_merged'] = True
        
        # Handle individual file results
        else:
            test_set = result.get('filename', 'unknown')
            
            # Handle round-trip format
            if 'roundtrip' in result:
                file_scores[test_set] = {
                    'BLEU': result['roundtrip']['BLEU'],
                    'CHRF': result['roundtrip']['CHRF']
                }
                if 'dialect_code' in result:
                    file_scores[test_set]['dialect_code'] = result['dialect_code']
                if 'dialect_name' in result:
                    file_scores[test_set]['dialect_name'] = result['dialect_name']
            
            # Handle eval_arabench_reverse.py format (direct BLEU/CHRF)
            elif 'BLEU' in result and 'CHRF' in result and 'arabic_general' not in result:
                file_scores[test_set] = {
                    'BLEU': result['BLEU'],
                    'CHRF': result['CHRF']
                }
            
            # Handle eval_arabench.py format (arabic_general and dialect)
            elif 'arabic_general' in result:
                file_scores[test_set] = {}
                if 'BLEU' in result['arabic_general']:
                    file_scores[test_set]['arabic_general_BLEU'] = result['arabic_general']['BLEU']
                if 'CHRF' in result['arabic_general']:
                    file_scores[test_set]['arabic_general_CHRF'] = result['arabic_general']['CHRF']
                
                if 'dialect' in result:
                    if 'BLEU' in result['dialect']:
                        file_scores[test_set]['dialect_BLEU'] = result['dialect']['BLEU']
                    if 'CHRF' in result['dialect']:
                        file_scores[test_set]['dialect_CHRF'] = result['dialect']['CHRF']
                
                # Store metadata
                if 'dialect_code' in result:
                    file_scores[test_set]['dialect_code'] = result['dialect_code']
                if 'dialect_name' in result:
                    file_scores[test_set]['dialect_name'] = result['dialect_name']
                if 'num_sentences' in result:
                    file_scores[test_set]['num_sentences'] = result['num_sentences']
                file_scores[test_set]['is_merged'] = False
    
    return file_scores, merged_scores, overall_scores


def collect_all_scores(scores_files: List[Path]) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Collect all scores from all files, grouped by translation direction.
    
    Returns:
        - results_by_direction: {
              direction: {
                  'file_test_sets': {test_set -> {model -> metrics}},
                  'merged_test_sets': {dialect -> {model -> metrics}},
                  'overall_scores': {model -> metrics}
              }
          }
        - models: List of all model names
    """
    results_by_direction: Dict[str, Dict[str, Dict]] = {}
    models = set()
    
    for scores_path in scores_files:
        scores_data = load_scores_file(scores_path)
        if not scores_data:
            continue
        
        direction = scores_data.get('direction', 'forward')
        model_name = extract_model_name(scores_data, scores_path)
        models.add(model_name)
        
        if direction not in results_by_direction:
            results_by_direction[direction] = {
                'file_test_sets': defaultdict(lambda: defaultdict(dict)),
                'merged_test_sets': defaultdict(lambda: defaultdict(dict)),
                'overall_scores': defaultdict(dict),
            }
        
        file_scores, merged_scores, overall_scores = extract_scores(scores_data)
        
        # Add file scores
        for test_set, metrics in file_scores.items():
            results_by_direction[direction]['file_test_sets'][test_set][model_name] = metrics
        
        # Add merged dialect scores (only present for forward direction)
        for dialect_name, metrics in merged_scores.items():
            results_by_direction[direction]['merged_test_sets'][dialect_name][model_name] = metrics
            print(f"   Found merged dialect '{dialect_name}' for model '{model_name}' (direction: {direction})")
        
        # Add overall scores
        if overall_scores:
            results_by_direction[direction]['overall_scores'][model_name] = overall_scores
            print(f"   Found overall scores for model '{model_name}' (direction: {direction})")
    
    # Debug: Print summary of merged dialects found per direction
    for direction, data in results_by_direction.items():
        merged_test_sets = data['merged_test_sets']
        if merged_test_sets:
            print(f"\n📊 Merged dialects summary ({direction}):")
            for dialect_name in sorted(merged_test_sets.keys()):
                models_with_dialect = list(merged_test_sets[dialect_name].keys())
                print(f"   {dialect_name}: {len(models_with_dialect)} model(s) - {', '.join(models_with_dialect)}")
    
    # Convert nested defaultdicts to dicts
    for direction in list(results_by_direction.keys()):
        results_by_direction[direction]['file_test_sets'] = dict(results_by_direction[direction]['file_test_sets'])
        results_by_direction[direction]['merged_test_sets'] = dict(results_by_direction[direction]['merged_test_sets'])
    
    return results_by_direction, sorted(models)


def write_scores_table(test_sets: Dict, models: List[str], score_type: Optional[str], 
                      metric: str, stats_file, title: str = ""):
    """Write a table of scores to the stats file."""
    if title:
        stats_file.write(f"\n{title}\n")
        stats_file.write("=" * 80 + "\n\n")
    
    # Collect all test sets and their scores
    test_set_names = sorted(test_sets.keys())
    
    if not test_set_names:
        stats_file.write("No test sets with scores.\n\n")
        return
    
    # Determine the key to look for
    if score_type:
        key = f'{score_type}_{metric}'
    else:
        key = metric
    
    # Calculate column widths
    max_test_set_len = max(len(ts) for ts in test_set_names) if test_set_names else 20
    test_set_width = min(max(max_test_set_len, 30), 50)
    model_width = 15
    
    # Create header
    header = f"{'Test Set':<{test_set_width}}"
    for model in models:
        # Truncate model name if too long
        model_display = model[:model_width-2] + ".." if len(model) > model_width else model
        header += f"{model_display:>{model_width}}"
    stats_file.write(header + "\n")
    stats_file.write("-" * (test_set_width + model_width * len(models)) + "\n")
    
    # Write scores for each test set
    for test_set in test_set_names:
        model_scores = test_sets[test_set]
        # Truncate test set name if too long
        test_set_display = test_set[:test_set_width-2] + ".." if len(test_set) > test_set_width else test_set
        line = f"{test_set_display:<{test_set_width}}"
        
        for model in models:
            if model in model_scores:
                if score_type and key in model_scores[model]:
                    score = model_scores[model][key]
                    line += f"{score:>{model_width}.4f}"
                elif not score_type:
                    # Try to find the metric in various formats
                    score = None
                    if metric in model_scores[model]:
                        score = model_scores[model][metric]
                    elif f'arabic_general_{metric}' in model_scores[model]:
                        score = model_scores[model][f'arabic_general_{metric}']
                    elif f'dialect_{metric}' in model_scores[model]:
                        score = model_scores[model][f'dialect_{metric}']
                    
                    if score is not None:
                        line += f"{score:>{model_width}.4f}"
                    else:
                        line += f"{'N/A':>{model_width}}"
                else:
                    line += f"{'N/A':>{model_width}}"
            else:
                line += f"{'N/A':>{model_width}}"
        
        stats_file.write(line + "\n")
    
    stats_file.write("\n")


def create_bar_plot(test_set: str, model_scores: Dict[str, Dict[str, float]], 
                    models: List[str], output_dir: Path, metric: str = 'BLEU'):
    """Create a bar plot for a specific test set and metric."""
    # Extract scores for this metric
    model_values = []
    model_labels = []
    
    for model in models:
        if model in model_scores:
            # Try to find the metric in various formats
            score = None
            if metric in model_scores[model]:
                score = model_scores[model][metric]
            elif f'arabic_general_{metric}' in model_scores[model]:
                score = model_scores[model][f'arabic_general_{metric}']
            elif f'dialect_{metric}' in model_scores[model]:
                score = model_scores[model][f'dialect_{metric}']
            
            if score is not None:
                model_values.append(score)
                model_labels.append(model)
    
    if not model_values:
        print(f"⚠️  No {metric} scores found for test set: {test_set}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(model_labels) * 0.8), 6))
    
    # Create bars
    bars = ax.bar(range(len(model_labels)), model_values, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(model_labels))))
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Scores by Model\nTest Set: {test_set}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, model_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    safe_test_set = test_set.replace('/', '_').replace('.', '_')
    output_file = output_dir / f"{safe_test_set}_{metric.lower()}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def create_combined_plot(test_set: str, model_scores: Dict[str, Dict[str, float]], 
                         models: List[str], output_dir: Path):
    """Create a combined plot with both BLEU and CHRF scores."""
    # Extract BLEU and CHRF scores
    bleu_values = []
    chrf_values = []
    model_labels = []
    
    for model in models:
        if model in model_scores:
            bleu_score = None
            chrf_score = None
            
            # Try to find BLEU
            if 'BLEU' in model_scores[model]:
                bleu_score = model_scores[model]['BLEU']
            elif 'arabic_general_BLEU' in model_scores[model]:
                bleu_score = model_scores[model]['arabic_general_BLEU']
            elif 'dialect_BLEU' in model_scores[model]:
                bleu_score = model_scores[model]['dialect_BLEU']
            
            # Try to find CHRF
            if 'CHRF' in model_scores[model]:
                chrf_score = model_scores[model]['CHRF']
            elif 'arabic_general_CHRF' in model_scores[model]:
                chrf_score = model_scores[model]['arabic_general_CHRF']
            elif 'dialect_CHRF' in model_scores[model]:
                chrf_score = model_scores[model]['dialect_CHRF']
            
            if bleu_score is not None or chrf_score is not None:
                bleu_values.append(bleu_score if bleu_score is not None else 0)
                chrf_values.append(chrf_score if chrf_score is not None else 0)
                model_labels.append(model)
    
    if not model_labels:
        print(f"⚠️  No scores found for test set: {test_set}")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(model_labels))
    width = 0.6
    
    # BLEU plot
    bars1 = ax1.bar(x, bleu_values, width, color='steelblue', label='BLEU')
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
    ax1.set_title('BLEU Scores', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    
    # Add value labels on BLEU bars
    for bar, val in zip(bars1, bleu_values):
        if val > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    # CHRF plot
    bars2 = ax2.bar(x, chrf_values, width, color='coral', label='CHRF')
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('CHRF Score', fontsize=12, fontweight='bold')
    ax2.set_title('CHRF Scores', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    
    # Add value labels on CHRF bars
    for bar, val in zip(bars2, chrf_values):
        if val > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Overall title
    fig.suptitle(f'BLEU and CHRF Scores by Model\nTest Set: {test_set}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    safe_test_set = test_set.replace('/', '_').replace('.', '_')
    output_file = output_dir / f"{safe_test_set}_combined.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def create_summary_plot(test_sets: Dict, models: List[str], output_dir: Path, metric: str = 'BLEU'):
    """Create a summary plot showing all test sets and models."""
    # Prepare data matrix
    test_set_names = sorted(test_sets.keys())
    data_matrix = []
    
    for test_set in test_set_names:
        row = []
        for model in models:
            score = None
            if model in test_sets[test_set]:
                if metric in test_sets[test_set][model]:
                    score = test_sets[test_set][model][metric]
                elif f'arabic_general_{metric}' in test_sets[test_set][model]:
                    score = test_sets[test_set][model][f'arabic_general_{metric}']
                elif f'dialect_{metric}' in test_sets[test_set][model]:
                    score = test_sets[test_set][model][f'dialect_{metric}']
            row.append(score if score is not None else np.nan)
        data_matrix.append(row)
    
    if not data_matrix or all(all(np.isnan(x) for x in row) for row in data_matrix):
        print(f"⚠️  No {metric} scores found for summary plot")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(models) * 1.2), max(8, len(test_set_names) * 0.5)))
    
    im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(test_set_names)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(test_set_names, fontsize=9)
    
    # Add text annotations
    for i in range(len(test_set_names)):
        for j in range(len(models)):
            val = data_matrix[i][j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{metric} Score', rotation=270, labelpad=20, fontsize=11)
    
    ax.set_title(f'{metric} Scores: All Test Sets vs All Models', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Set', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"summary_{metric.lower()}_heatmap.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def create_average_plot(test_sets: Dict, models: List[str], output_dir: Path, metric: str = 'BLEU'):
    """Create a plot showing average scores across all test sets for each model."""
    # Calculate averages for each model
    model_averages = {}
    
    for model in models:
        scores = []
        for test_set, model_scores in test_sets.items():
            if model in model_scores:
                score = None
                if metric in model_scores[model]:
                    score = model_scores[model][metric]
                elif f'arabic_general_{metric}' in model_scores[model]:
                    score = model_scores[model][f'arabic_general_{metric}']
                elif f'dialect_{metric}' in model_scores[model]:
                    score = model_scores[model][f'dialect_{metric}']
                
                if score is not None:
                    scores.append(score)
        
        if scores:
            model_averages[model] = np.mean(scores)
    
    if not model_averages:
        print(f"⚠️  No {metric} scores found for average plot")
        return
    
    # Sort models by average score (descending)
    sorted_models = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
    model_names = [m[0] for m in sorted_models]
    avg_values = [m[1] for m in sorted_models]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(model_names) * 0.8), 6))
    
    # Create bars
    bars = ax.bar(range(len(model_names)), avg_values, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Average {metric} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Average {metric} Score Across All Test Sets\n(Models sorted by performance)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # Add number of test sets info
    num_test_sets = len(test_sets)
    ax.text(0.02, 0.98, f'Based on {num_test_sets} test set(s)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"average_{metric.lower()}_all_testsets.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def create_all_testsets_plot(test_sets: Dict, models: List[str], output_dir: Path, metric: str = 'BLEU'):
    """Create a grouped bar chart showing all test sets in a single plot."""
    test_set_names = sorted(test_sets.keys())
    
    if len(test_set_names) > 20:
        print(f"⚠️  Too many test sets ({len(test_set_names)}) for grouped bar chart. Skipping.")
        print(f"   Consider using the average plot or heatmap instead.")
        return
    
    # Prepare data
    data = {model: [] for model in models}
    valid_test_sets = []
    
    for test_set in test_set_names:
        has_data = False
        for model in models:
            score = None
            if model in test_sets[test_set]:
                if metric in test_sets[test_set][model]:
                    score = test_sets[test_set][model][metric]
                elif f'arabic_general_{metric}' in test_sets[test_set][model]:
                    score = test_sets[test_set][model][f'arabic_general_{metric}']
                elif f'dialect_{metric}' in test_sets[test_set][model]:
                    score = test_sets[test_set][model][f'dialect_{metric}']
            
            data[model].append(score if score is not None else np.nan)
            if score is not None:
                has_data = True
        
        if has_data:
            valid_test_sets.append(test_set)
    
    if not valid_test_sets:
        print(f"⚠️  No {metric} scores found for all test sets plot")
        return
    
    # Truncate test set names for display
    display_names = [name[:30] + '...' if len(name) > 30 else name for name in valid_test_sets]
    
    # Create figure
    x = np.arange(len(valid_test_sets))
    width = 0.8 / len(models)  # Width of bars
    fig, ax = plt.subplots(figsize=(max(16, len(valid_test_sets) * 1.5), 8))
    
    # Create bars for each model
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    bars_list = []
    
    for i, model in enumerate(models):
        values = [data[model][test_set_names.index(ts)] if ts in test_set_names else np.nan 
                  for ts in valid_test_sets]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
        bars_list.append(bars)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val) and val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=7, rotation=90)
    
    # Customize plot
    ax.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Scores: All Test Sets Compared\n(Grouped by Model)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"all_testsets_{metric.lower()}_grouped.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def create_all_testsets_combined_plot(test_sets: Dict, models: List[str], output_dir: Path, metric: str = 'BLEU'):
    """Create a single image with all test set plots side by side."""
    # Get all test sets that have data for this metric
    valid_test_sets = []
    for test_set, model_scores in test_sets.items():
        has_data = False
        for model in models:
            if model in model_scores:
                if metric in model_scores[model] or \
                   f'arabic_general_{metric}' in model_scores[model] or \
                   f'dialect_{metric}' in model_scores[model]:
                    has_data = True
                    break
        if has_data:
            valid_test_sets.append(test_set)
    
    if not valid_test_sets:
        print(f"⚠️  No {metric} scores found for combined plot")
        return
    
    # Sort test sets
    valid_test_sets = sorted(valid_test_sets)
    
    # Calculate grid dimensions
    n_plots = len(valid_test_sets)
    n_cols = min(4, n_plots)  # Max 4 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    # Create a plot for each test set
    for idx, test_set in enumerate(valid_test_sets):
        ax = axes[idx]
        model_scores = test_sets[test_set]
        
        # Extract scores for this metric
        model_values = []
        model_labels = []
        
        for model in models:
            if model in model_scores:
                score = None
                if metric in model_scores[model]:
                    score = model_scores[model][metric]
                elif f'arabic_general_{metric}' in model_scores[model]:
                    score = model_scores[model][f'arabic_general_{metric}']
                elif f'dialect_{metric}' in model_scores[model]:
                    score = model_scores[model][f'dialect_{metric}']
                
                if score is not None:
                    model_values.append(score)
                    model_labels.append(model)
        
        if not model_values:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(test_set[:40] + '...' if len(test_set) > 40 else test_set, fontsize=10)
            continue
        
        # Create bars
        bars = ax.bar(range(len(model_labels)), model_values,
                     color=plt.cm.viridis(np.linspace(0, 1, len(model_labels))))
        
        # Customize subplot
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel(f'{metric} Score', fontsize=9)
        ax.set_title(test_set[:40] + '...' if len(test_set) > 40 else test_set, 
                    fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Add value labels on bars (only if there's space)
        if len(model_labels) <= 8:  # Only add labels if not too crowded
            for bar, val in zip(bars, model_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}',
                       ha='center', va='bottom', fontsize=7)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle(f'{metric} Scores: All Test Sets\n(Each subplot shows one test set)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    output_file = output_dir / f"all_testsets_{metric.lower()}_combined.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def create_all_testsets_combined_plot_by_type(test_sets: Dict, models: List[str], output_dir: Path, 
                                             score_type: str, metric: str = 'BLEU'):
    """
    Create a single image with all test set plots side by side for a specific score type.
    
    Args:
        score_type: 'arabic_general' or 'dialect'
    """
    # Get all test sets that have data for this score type and metric
    valid_test_sets = []
    for test_set, model_scores in test_sets.items():
        has_data = False
        for model in models:
            if model in model_scores:
                key = f'{score_type}_{metric}'
                if key in model_scores[model]:
                    has_data = True
                    break
        if has_data:
            valid_test_sets.append(test_set)
    
    if not valid_test_sets:
        print(f"⚠️  No {score_type} {metric} scores found for combined plot")
        return
    
    # Sort test sets
    valid_test_sets = sorted(valid_test_sets)
    
    # Calculate grid dimensions
    n_plots = len(valid_test_sets)
    n_cols = min(4, n_plots)  # Max 4 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    # Create a plot for each test set
    for idx, test_set in enumerate(valid_test_sets):
        ax = axes[idx]
        model_scores = test_sets[test_set]
        
        # Extract scores for this metric and score type
        model_values = []
        model_labels = []
        
        for model in models:
            if model in model_scores:
                key = f'{score_type}_{metric}'
                if key in model_scores[model]:
                    score = model_scores[model][key]
                    model_values.append(score)
                    model_labels.append(model)
        
        if not model_values:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(test_set[:40] + '...' if len(test_set) > 40 else test_set, fontsize=10)
            continue
        
        # Create bars
        bars = ax.bar(range(len(model_labels)), model_values,
                     color=plt.cm.viridis(np.linspace(0, 1, len(model_labels))))
        
        # Customize subplot
        ax.set_xlabel('Model', fontsize=9)
        ax.set_ylabel(f'{metric} Score', fontsize=9)
        title_label = 'General Arabic' if score_type == 'arabic_general' else 'Dialect'
        ax.set_title(f"{test_set[:35] + '...' if len(test_set) > 35 else test_set}\n({title_label})", 
                    fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Add value labels on bars (only if there's space)
        if len(model_labels) <= 8:  # Only add labels if not too crowded
            for bar, val in zip(bars, model_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}',
                       ha='center', va='bottom', fontsize=7)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    title_label = 'General Arabic' if score_type == 'arabic_general' else 'Dialect'
    fig.suptitle(f'{metric} Scores: All Test Sets ({title_label})\n(Each subplot shows one test set)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    output_file = output_dir / f"all_testsets_{score_type}_{metric.lower()}_combined.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def calculate_general_dialect_difference(test_sets: Dict, models: List[str], metric: str = 'BLEU') -> Dict[str, float]:
    """
    Calculate average difference between general Arabic and dialect scores for each model.
    
    Returns: Dict mapping model_name -> average_difference (general - dialect)
    """
    differences = {model: [] for model in models}
    
    for test_set, model_scores in test_sets.items():
        for model in models:
            if model in model_scores:
                general_key = f'arabic_general_{metric}'
                dialect_key = f'dialect_{metric}'
                
                general_score = model_scores[model].get(general_key)
                dialect_score = model_scores[model].get(dialect_key)
                
                if general_score is not None and dialect_score is not None:
                    diff = general_score - dialect_score
                    differences[model].append(diff)
    
    # Calculate averages
    avg_differences = {}
    for model in models:
        if differences[model]:
            avg_differences[model] = np.mean(differences[model])
    
    return avg_differences


def print_general_dialect_difference(test_sets: Dict, models: List[str], metric: str = 'BLEU', stats_file=None):
    """Print average difference between general Arabic and dialect scores for each model."""
    avg_differences = calculate_general_dialect_difference(test_sets, models, metric)
    
    if not avg_differences:
        msg = f"\n⚠️  No general/dialect score pairs found for {metric}"
        print(msg)
        if stats_file:
            stats_file.write(msg + "\n")
        return
    
    header = f"\n{'='*80}\n📊 Average Difference: General Arabic - Dialect ({metric} Score)\n{'='*80}\nPositive values mean general Arabic scores are higher.\nNegative values mean dialect scores are higher.\n\n"
    print(header, end='')
    if stats_file:
        stats_file.write(header)
    
    # Sort by difference
    sorted_models = sorted(avg_differences.items(), key=lambda x: x[1], reverse=True)
    
    # Count how many test sets each model appears in
    model_counts = {}
    for test_set, model_scores in test_sets.items():
        for model in models:
            if model in model_scores:
                general_key = f'arabic_general_{metric}'
                dialect_key = f'dialect_{metric}'
                if general_key in model_scores[model] and dialect_key in model_scores[model]:
                    model_counts[model] = model_counts.get(model, 0) + 1
    
    for model, avg_diff in sorted_models:
        count = model_counts.get(model, 0)
        lines = [
            f"Model: {model}",
            f"  Average difference: {avg_diff:+.4f} ({metric} points)",
            f"  Based on {count} test set(s)"
        ]
        if avg_diff > 0:
            lines.append(f"  → General Arabic scores are {avg_diff:.4f} points higher on average")
        elif avg_diff < 0:
            lines.append(f"  → Dialect scores are {abs(avg_diff):.4f} points higher on average")
        else:
            lines.append(f"  → Scores are equal on average")
        lines.append("")
        
        output = "\n".join(lines)
        print(output, end='')
        if stats_file:
            stats_file.write(output)


def calculate_ranking_stats_by_type(test_sets: Dict, models: List[str], score_type: str, metric: str = 'BLEU') -> Dict[str, Dict[int, int]]:
    """
    Calculate ranking statistics for a specific score type (arabic_general or dialect).
    
    Returns: Dict mapping model_name -> Dict mapping rank -> count
    """
    rankings = {model: {} for model in models}
    key = f'{score_type}_{metric}'
    
    for test_set, model_scores in test_sets.items():
        # Collect scores for this test set and score type
        model_scores_list = []
        for model in models:
            if model in model_scores and key in model_scores[model]:
                score = model_scores[model][key]
                if score is not None:
                    model_scores_list.append((model, score))
        
        if not model_scores_list:
            continue
        
        # Sort by score (descending)
        model_scores_list.sort(key=lambda x: x[1], reverse=True)
        
        # Assign ranks (handle ties)
        current_rank = 1
        i = 0
        while i < len(model_scores_list):
            score = model_scores_list[i][1]
            # Count how many models have this same score
            tie_count = sum(1 for _, s in model_scores_list if s == score)
            
            # All models with this score get the same rank
            for j in range(i, i + tie_count):
                model = model_scores_list[j][0]
                if current_rank not in rankings[model]:
                    rankings[model][current_rank] = 0
                rankings[model][current_rank] += 1
            
            current_rank += tie_count
            i += tie_count
    
    return rankings


def calculate_average_rankings(test_sets: Dict, models: List[str], score_type: Optional[str] = None, metric: str = 'BLEU') -> Dict[str, float]:
    """
    Calculate average ranking for each model across all test sets.
    
    Args:
        score_type: 'arabic_general', 'dialect', or None (for combined)
        metric: 'BLEU' or 'CHRF'
    
    Returns: Dict mapping model_name -> average_rank
    """
    model_ranks = {model: [] for model in models}
    
    for test_set, model_scores in test_sets.items():
        # Collect scores for this test set
        model_scores_list = []
        for model in models:
            if model in model_scores:
                score = None
                if score_type:
                    key = f'{score_type}_{metric}'
                    score = model_scores[model].get(key)
                else:
                    # Try to find the metric in various formats
                    if metric in model_scores[model]:
                        score = model_scores[model][metric]
                    elif f'arabic_general_{metric}' in model_scores[model]:
                        score = model_scores[model][f'arabic_general_{metric}']
                    elif f'dialect_{metric}' in model_scores[model]:
                        score = model_scores[model][f'dialect_{metric}']
                
                if score is not None:
                    model_scores_list.append((model, score))
        
        if not model_scores_list:
            continue
        
        # Sort by score (descending)
        model_scores_list.sort(key=lambda x: x[1], reverse=True)
        
        # Assign ranks (handle ties)
        current_rank = 1
        i = 0
        while i < len(model_scores_list):
            score = model_scores_list[i][1]
            # Count how many models have this same score
            tie_count = sum(1 for _, s in model_scores_list if s == score)
            
            # All models with this score get the same rank
            # For ties, use the average rank (e.g., if 2 models tie for 1st, both get rank 1.5)
            avg_rank = current_rank + (tie_count - 1) / 2.0
            
            for j in range(i, i + tie_count):
                model = model_scores_list[j][0]
                model_ranks[model].append(avg_rank)
            
            current_rank += tie_count
            i += tie_count
    
    # Calculate averages
    avg_rankings = {}
    for model in models:
        if model_ranks[model]:
            avg_rankings[model] = np.mean(model_ranks[model])
    
    return avg_rankings


def print_average_rankings(test_sets: Dict, models: List[str], score_type: Optional[str] = None, metric: str = 'BLEU', stats_file=None):
    """Print average rankings for each model."""
    avg_rankings = calculate_average_rankings(test_sets, models, score_type, metric)
    
    if not avg_rankings:
        title_label = ''
        if score_type == 'arabic_general':
            title_label = 'General Arabic '
        elif score_type == 'dialect':
            title_label = 'Dialect '
        msg = f"\n⚠️  No {title_label}{metric} scores found for average ranking calculation"
        print(msg)
        if stats_file:
            stats_file.write(msg + "\n")
        return
    
    # Sort by average rank (ascending - lower is better)
    sorted_models = sorted(avg_rankings.items(), key=lambda x: x[1])
    
    title_label = ''
    if score_type == 'arabic_general':
        title_label = 'General Arabic '
    elif score_type == 'dialect':
        title_label = 'Dialect '
    
    header = f"\n{'='*80}\n📊 Average Rankings for {title_label}{metric} Score\n{'='*80}\nLower rank is better (1st = best)\n\n"
    print(header, end='')
    if stats_file:
        stats_file.write(header)
    
    lines = []
    for rank, (model, avg_rank) in enumerate(sorted_models, 1):
        # Count how many test sets this model appeared in
        count = 0
        key = f'{score_type}_{metric}' if score_type else None
        for test_set, model_scores in test_sets.items():
            if model in model_scores:
                if key and key in model_scores[model]:
                    count += 1
                elif not key:
                    if metric in model_scores[model] or \
                       f'arabic_general_{metric}' in model_scores[model] or \
                       f'dialect_{metric}' in model_scores[model]:
                        count += 1
        
        line = f"{rank}. {model}: {avg_rank:.2f} (based on {count} test set(s))"
        lines.append(line)
        print(line)
    
    lines.append("")
    if stats_file:
        stats_file.write("\n".join(lines) + "\n")


def print_ranking_stats_by_type(test_sets: Dict, models: List[str], score_type: str, metric: str = 'BLEU', stats_file=None):
    """Print ranking statistics for a specific score type."""
    rankings = calculate_ranking_stats_by_type(test_sets, models, score_type, metric)
    
    title_label = 'General Arabic' if score_type == 'arabic_general' else 'Dialect'
    header = f"\n{'='*80}\n📊 Ranking Statistics for {title_label} {metric} Score\n{'='*80}\n"
    print(header, end='')
    if stats_file:
        stats_file.write(header)
    
    # Calculate total test sets with data for this score type and metric
    test_sets_with_data = set()
    key = f'{score_type}_{metric}'
    for test_set, model_scores in test_sets.items():
        for model in models:
            if model in model_scores and key in model_scores[model]:
                test_sets_with_data.add(test_set)
                break
    
    total_test_sets = len(test_sets_with_data)
    info = f"Total test sets with {title_label} {metric} scores: {total_test_sets}\n\n"
    print(info, end='')
    if stats_file:
        stats_file.write(info)
    
    # Print stats for each model
    lines = []
    for model in sorted(models):
        model_rankings = rankings[model]
        if not model_rankings:
            continue
        
        total_appearances = sum(model_rankings.values())
        lines.append(f"Model: {model}")
        lines.append(f"  Total appearances: {total_appearances}/{total_test_sets}")
        
        # Print rank distribution
        rank_counts = []
        for rank in sorted(model_rankings.keys()):
            count = model_rankings[rank]
            rank_counts.append(f"  {rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'}: {count} time{'s' if count != 1 else ''}")
        
        if rank_counts:
            lines.extend(rank_counts)
        
        # Calculate win rate (1st place)
        wins = model_rankings.get(1, 0)
        win_rate = (wins / total_appearances * 100) if total_appearances > 0 else 0
        lines.append(f"  Win rate: {wins}/{total_appearances} ({win_rate:.1f}%)")
        
        # Calculate top-3 rate
        top3 = sum(model_rankings.get(r, 0) for r in [1, 2, 3])
        top3_rate = (top3 / total_appearances * 100) if total_appearances > 0 else 0
        lines.append(f"  Top-3 rate: {top3}/{total_appearances} ({top3_rate:.1f}%)")
        lines.append("")
    
    output = "\n".join(lines)
    print(output, end='')
    if stats_file:
        stats_file.write(output)


def calculate_average_scores_by_type(test_sets: Dict, models: List[str], score_type: str, metric: str = 'BLEU') -> Dict[str, float]:
    """
    Calculate average scores for a specific score type (arabic_general or dialect).
    
    Returns: Dict mapping model_name -> average_score
    """
    averages = {}
    key = f'{score_type}_{metric}'
    
    for model in models:
        scores = []
        for test_set, model_scores in test_sets.items():
            if model in model_scores and key in model_scores[model]:
                score = model_scores[model][key]
                if score is not None:
                    scores.append(score)
        
        if scores:
            averages[model] = np.mean(scores)
    
    return averages


def print_average_scores_by_type(test_sets: Dict, models: List[str], score_type: str, metric: str = 'BLEU', stats_file=None):
    """Print average scores for a specific score type."""
    averages = calculate_average_scores_by_type(test_sets, models, score_type, metric)
    
    if not averages:
        title_label = 'General Arabic' if score_type == 'arabic_general' else 'Dialect'
        msg = f"\n⚠️  No {title_label} {metric} scores found"
        print(msg)
        if stats_file:
            stats_file.write(msg + "\n")
        return
    
    title_label = 'General Arabic' if score_type == 'arabic_general' else 'Dialect'
    header = f"\n{'='*80}\n📊 Average {metric} Scores for {title_label} Translations\n{'='*80}\n\nModels sorted by average {metric} score:\n\n"
    print(header, end='')
    if stats_file:
        stats_file.write(header)
    
    # Count test sets for each model
    model_counts = {}
    key = f'{score_type}_{metric}'
    for test_set, model_scores in test_sets.items():
        for model in models:
            if model in model_scores and key in model_scores[model]:
                model_counts[model] = model_counts.get(model, 0) + 1
    
    # Sort by average score (descending)
    sorted_models = sorted(averages.items(), key=lambda x: x[1], reverse=True)
    
    lines = []
    for model, avg_score in sorted_models:
        count = model_counts.get(model, 0)
        lines.extend([
            f"Model: {model}",
            f"  Average {metric}: {avg_score:.4f}",
            f"  Based on {count} test set(s)",
            ""
        ])
    
    output = "\n".join(lines)
    print(output, end='')
    if stats_file:
        stats_file.write(output)


def print_overall_scores(overall_scores: Dict[str, Dict], models: List[str], direction: str, 
                         score_type: Optional[str] = None, stats_file=None):
    """
    Print overall scores (all test sets concatenated) for each model.
    
    Args:
        overall_scores: Dict mapping model_name -> overall scores dict
        models: List of model names
        direction: Translation direction ('forward', 'reverse', 'roundtrip')
        score_type: For forward direction, 'arabic_general' or 'dialect' (None for reverse/roundtrip)
        stats_file: Optional file handle to write to
    """
    if not overall_scores:
        msg = f"\n⚠️  No overall scores found"
        print(msg)
        if stats_file:
            stats_file.write(msg + "\n")
        return
    
    title_label = ''
    if direction == 'forward' and score_type:
        title_label = 'General Arabic' if score_type == 'arabic_general' else 'Dialect'
    elif direction == 'reverse':
        title_label = 'Reverse Translation'
    elif direction == 'roundtrip':
        title_label = 'Round-trip Translation'
    
    header = f"\n{'='*80}\n📊 Overall Scores (All Test Sets Concatenated)"
    if title_label:
        header += f" - {title_label}"
    header += f"\n{'='*80}\n\n"
    print(header, end='')
    if stats_file:
        stats_file.write(header)
    
    lines = []
    for model in sorted(models):
        if model not in overall_scores:
            continue
        
        model_scores = overall_scores[model]
        lines.append(f"Model: {model}")
        
        if direction == 'forward' and score_type:
            # Forward direction with specific score type
            key = score_type
            if key in model_scores:
                if 'BLEU' in model_scores[key]:
                    lines.append(f"  BLEU: {model_scores[key]['BLEU']:.4f}")
                if 'CHRF' in model_scores[key]:
                    lines.append(f"  CHRF: {model_scores[key]['CHRF']:.4f}")
        elif direction == 'reverse':
            # Reverse direction: direct BLEU/CHRF
            if 'BLEU' in model_scores:
                lines.append(f"  BLEU: {model_scores['BLEU']:.4f}")
            if 'CHRF' in model_scores:
                lines.append(f"  CHRF: {model_scores['CHRF']:.4f}")
        elif direction == 'roundtrip':
            # Roundtrip: roundtrip scores
            if 'roundtrip' in model_scores:
                if 'BLEU' in model_scores['roundtrip']:
                    lines.append(f"  BLEU: {model_scores['roundtrip']['BLEU']:.4f}")
                if 'CHRF' in model_scores['roundtrip']:
                    lines.append(f"  CHRF: {model_scores['roundtrip']['CHRF']:.4f}")
        
        # Add metadata if available
        if 'num_test_sets' in model_scores:
            lines.append(f"  Test sets: {model_scores['num_test_sets']}")
        if 'num_sentences' in model_scores:
            lines.append(f"  Sentences: {model_scores['num_sentences']}")
        
        lines.append("")
    
    output = "\n".join(lines)
    print(output, end='')
    if stats_file:
        stats_file.write(output)


def calculate_ranking_stats(test_sets: Dict, models: List[str], metric: str = 'BLEU') -> Dict[str, Dict[int, int]]:
    """
    Calculate ranking statistics for each model.
    
    Returns: Dict mapping model_name -> Dict mapping rank -> count
    """
    rankings = {model: {} for model in models}
    
    for test_set, model_scores in test_sets.items():
        # Collect scores for this test set
        model_scores_list = []
        for model in models:
            if model in model_scores:
                score = None
                if metric in model_scores[model]:
                    score = model_scores[model][metric]
                elif f'arabic_general_{metric}' in model_scores[model]:
                    score = model_scores[model][f'arabic_general_{metric}']
                elif f'dialect_{metric}' in model_scores[model]:
                    score = model_scores[model][f'dialect_{metric}']
                
                if score is not None:
                    model_scores_list.append((model, score))
        
        if not model_scores_list:
            continue
        
        # Sort by score (descending)
        model_scores_list.sort(key=lambda x: x[1], reverse=True)
        
        # Assign ranks (handle ties)
        current_rank = 1
        i = 0
        while i < len(model_scores_list):
            score = model_scores_list[i][1]
            # Count how many models have this same score
            tie_count = sum(1 for _, s in model_scores_list if s == score)
            
            # All models with this score get the same rank
            for j in range(i, i + tie_count):
                model = model_scores_list[j][0]
                if current_rank not in rankings[model]:
                    rankings[model][current_rank] = 0
                rankings[model][current_rank] += 1
            
            current_rank += tie_count
            i += tie_count
    
    return rankings


def print_ranking_stats(test_sets: Dict, models: List[str], metric: str = 'BLEU', stats_file=None):
    """Print ranking statistics for each model."""
    rankings = calculate_ranking_stats(test_sets, models, metric)
    
    header = f"\n{'='*80}\n📊 Ranking Statistics for {metric} Score\n{'='*80}\n"
    print(header, end='')
    if stats_file:
        stats_file.write(header)
    
    # Calculate total test sets with data for this metric
    test_sets_with_data = set()
    for test_set, model_scores in test_sets.items():
        for model in models:
            if model in model_scores:
                if metric in model_scores[model] or \
                   f'arabic_general_{metric}' in model_scores[model] or \
                   f'dialect_{metric}' in model_scores[model]:
                    test_sets_with_data.add(test_set)
                    break
    
    total_test_sets = len(test_sets_with_data)
    info = f"Total test sets with {metric} scores: {total_test_sets}\n\n"
    print(info, end='')
    if stats_file:
        stats_file.write(info)
    
    # Print stats for each model
    lines = []
    for model in sorted(models):
        model_rankings = rankings[model]
        if not model_rankings:
            continue
        
        total_appearances = sum(model_rankings.values())
        lines.append(f"Model: {model}")
        lines.append(f"  Total appearances: {total_appearances}/{total_test_sets}")
        
        # Print rank distribution
        rank_counts = []
        for rank in sorted(model_rankings.keys()):
            count = model_rankings[rank]
            rank_counts.append(f"  {rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'}: {count} time{'s' if count != 1 else ''}")
        
        if rank_counts:
            lines.extend(rank_counts)
        
        # Calculate win rate (1st place)
        wins = model_rankings.get(1, 0)
        win_rate = (wins / total_appearances * 100) if total_appearances > 0 else 0
        lines.append(f"  Win rate: {wins}/{total_appearances} ({win_rate:.1f}%)")
        
        # Calculate top-3 rate
        top3 = sum(model_rankings.get(r, 0) for r in [1, 2, 3])
        top3_rate = (top3 / total_appearances * 100) if total_appearances > 0 else 0
        lines.append(f"  Top-3 rate: {top3}/{total_appearances} ({top3_rate:.1f}%)")
        lines.append("")
    
    output = "\n".join(lines)
    print(output, end='')
    if stats_file:
        stats_file.write(output)


def create_bar_plot_by_type(test_set: str, model_scores: Dict[str, Dict[str, float]], 
                            models: List[str], output_dir: Path, score_type: str, metric: str = 'BLEU'):
    """Create a bar plot for a specific test set, score type (arabic_general or dialect), and metric."""
    key = f'{score_type}_{metric}'
    
    model_values = []
    model_labels = []
    
    for model in models:
        if model in model_scores and key in model_scores[model]:
            score = model_scores[model][key]
            if score is not None:
                model_values.append(score)
                model_labels.append(model)
    
    if not model_values:
        return  # Silently skip if no data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(model_labels) * 0.8), 6))
    
    # Create bars
    bars = ax.bar(range(len(model_labels)), model_values, 
                  color=plt.cm.viridis(np.linspace(0, 1, len(model_labels))))
    
    # Customize plot
    title_label = 'General Arabic' if score_type == 'arabic_general' else 'Dialect'
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Scores by Model ({title_label})\nTest Set: {test_set}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # Add value labels on bars
    for bar, val in zip(bars, model_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    safe_test_set = test_set.replace('/', '_').replace('.', '_')
    output_file = output_dir / f"{safe_test_set}_{score_type}_{metric.lower()}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def create_dialect_grouped_plot(test_sets: Dict, models: List[str], output_dir: Path, 
                                 score_type: str, metric: str = 'BLEU'):
    """
    Create plots grouped by dialect for a specific score type.
    Groups test sets by their dialect_name and creates a separate plot for each dialect.
    """
    # Group test sets by dialect
    dialect_groups: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    
    for test_set, model_scores in test_sets.items():
        # Get dialect name from any model's scores
        dialect_name = None
        for model in models:
            if model in model_scores and 'dialect_name' in model_scores[model]:
                dialect_name = model_scores[model]['dialect_name']
                break
        
        if dialect_name:
            dialect_groups[dialect_name][test_set] = model_scores
    
    if not dialect_groups:
        return
    
    title_label = 'General Arabic' if score_type == 'arabic_general' else 'Dialect'
    key = f'{score_type}_{metric}'
    
    # Create a plot for each dialect
    for dialect_name, dialect_test_sets in sorted(dialect_groups.items()):
        # Filter test sets that have data for this score type
        valid_test_sets = []
        for test_set, model_scores in dialect_test_sets.items():
            for model in models:
                if model in model_scores and key in model_scores[model]:
                    valid_test_sets.append(test_set)
                    break
        
        if not valid_test_sets:
            continue
        
        valid_test_sets = sorted(valid_test_sets)
        
        # Calculate grid dimensions
        n_plots = len(valid_test_sets)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = list(axes.flatten()) if hasattr(axes, 'flatten') else [axes]
        else:
            axes = list(axes.flatten())
        
        # Create a plot for each test set in this dialect
        for idx, test_set in enumerate(valid_test_sets):
            ax = axes[idx]
            model_scores = dialect_test_sets[test_set]
            
            model_values = []
            model_labels = []
            
            for model in models:
                if model in model_scores and key in model_scores[model]:
                    score = model_scores[model][key]
                    if score is not None:
                        model_values.append(score)
                        model_labels.append(model)
            
            if not model_values:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(test_set[:40] + '...' if len(test_set) > 40 else test_set, fontsize=10)
                continue
            
            # Create bars
            bars = ax.bar(range(len(model_labels)), model_values,
                         color=plt.cm.viridis(np.linspace(0, 1, len(model_labels))))
            
            # Customize subplot
            ax.set_xlabel('Model', fontsize=9)
            ax.set_ylabel(f'{metric} Score', fontsize=9)
            ax.set_title(test_set[:40] + '...' if len(test_set) > 40 else test_set, 
                        fontsize=10, fontweight='bold')
            ax.set_xticks(range(len(model_labels)))
            ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(bottom=0)
            
            # Add value labels
            if len(model_labels) <= 8:
                for bar, val in zip(bars, model_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1f}',
                           ha='center', va='bottom', fontsize=7)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        # Overall title
        fig.suptitle(f'{metric} Scores for {dialect_name} ({title_label})\n({n_plots} test set(s))', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save figure
        safe_dialect = dialect_name.replace('/', '_').replace('.', '_').replace(' ', '_')
        output_file = output_dir / f"dialect_{safe_dialect}_{score_type}_{metric.lower()}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")


def create_combined_average_plot(test_sets: Dict, models: List[str], output_dir: Path):
    """Create a combined plot showing average BLEU and CHRF across all test sets."""
    # Calculate averages for each model
    bleu_averages = {}
    chrf_averages = {}
    
    for model in models:
        bleu_scores = []
        chrf_scores = []
        
        for test_set, model_scores in test_sets.items():
            if model in model_scores:
                # Try to find BLEU
                bleu_score = None
                if 'BLEU' in model_scores[model]:
                    bleu_score = model_scores[model]['BLEU']
                elif 'arabic_general_BLEU' in model_scores[model]:
                    bleu_score = model_scores[model]['arabic_general_BLEU']
                elif 'dialect_BLEU' in model_scores[model]:
                    bleu_score = model_scores[model]['dialect_BLEU']
                
                # Try to find CHRF
                chrf_score = None
                if 'CHRF' in model_scores[model]:
                    chrf_score = model_scores[model]['CHRF']
                elif 'arabic_general_CHRF' in model_scores[model]:
                    chrf_score = model_scores[model]['arabic_general_CHRF']
                elif 'dialect_CHRF' in model_scores[model]:
                    chrf_score = model_scores[model]['dialect_CHRF']
                
                if bleu_score is not None:
                    bleu_scores.append(bleu_score)
                if chrf_score is not None:
                    chrf_scores.append(chrf_score)
        
        if bleu_scores:
            bleu_averages[model] = np.mean(bleu_scores)
        if chrf_scores:
            chrf_averages[model] = np.mean(chrf_scores)
    
    if not bleu_averages and not chrf_averages:
        print("⚠️  No scores found for combined average plot")
        return
    
    # Get all models that have at least one score
    all_models = sorted(set(list(bleu_averages.keys()) + list(chrf_averages.keys())))
    
    # Sort by BLEU if available, otherwise by CHRF
    if bleu_averages:
        sorted_models = sorted(all_models, key=lambda m: bleu_averages.get(m, 0), reverse=True)
    else:
        sorted_models = sorted(all_models, key=lambda m: chrf_averages.get(m, 0), reverse=True)
    
    bleu_values = [bleu_averages.get(m, 0) for m in sorted_models]
    chrf_values = [chrf_averages.get(m, 0) for m in sorted_models]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(sorted_models))
    width = 0.6
    
    # BLEU plot
    if bleu_averages:
        bars1 = ax1.bar(x, bleu_values, width, color='steelblue', label='BLEU')
        ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average BLEU Score', fontsize=12, fontweight='bold')
        ax1.set_title('Average BLEU Across All Test Sets', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(bottom=0)
        
        # Add value labels
        for bar, val in zip(bars1, bleu_values):
            if val > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No BLEU scores available', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax1.set_title('Average BLEU Across All Test Sets', fontsize=13, fontweight='bold')
    
    # CHRF plot
    if chrf_averages:
        bars2 = ax2.bar(x, chrf_values, width, color='coral', label='CHRF')
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average CHRF Score', fontsize=12, fontweight='bold')
        ax2.set_title('Average CHRF Across All Test Sets', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(bottom=0)
        
        # Add value labels
        for bar, val in zip(bars2, chrf_values):
            if val > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No CHRF scores available', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('Average CHRF Across All Test Sets', fontsize=13, fontweight='bold')
    
    # Overall title
    num_test_sets = len(test_sets)
    fig.suptitle(f'Average BLEU and CHRF Scores Across All Test Sets\n(Models sorted by BLEU, based on {num_test_sets} test set(s))', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / "average_combined_all_testsets.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize BLEU/CHRF scores from scores.json files in result subdirectories'
    )
    parser.add_argument(
        '--root-dir',
        type=str,
        default='.',
        help='Top-level directory to search for translation directories (containing scores.json files) (default: current directory)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Output directory for plots (default: visualizations)'
    )
    parser.add_argument(
        '--metric',
        type=str,
        choices=['BLEU', 'CHRF', 'both'],
        default='both',
        help='Which metric(s) to plot (default: both)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Also create summary heatmaps for all test sets'
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Searching for scores.json files in: {root_dir}")
    scores_files = find_scores_files(root_dir)
    
    if not scores_files:
        print(f"❌ No scores.json files found in {root_dir}")
        return
    
    print(f"📊 Found {len(scores_files)} scores.json file(s):")
    for f in scores_files:
        print(f"   - {f}")
    
    print(f"\n📈 Collecting scores...")
    results_by_direction, models = collect_all_scores(scores_files)
    
    if not results_by_direction:
        print("❌ No test sets found in scores files")
        return
    
    print(f"✅ Found {len(models)} model(s)")
    print(f"   Models: {', '.join(models)}")
    
    # Process each direction separately
    for direction, data in results_by_direction.items():
        if direction == 'forward':
            dir_label = "Forward (English -> Arabic)"
        elif direction == 'reverse':
            dir_label = "Reverse (Arabic -> English)"
        elif direction == 'roundtrip':
            dir_label = "Round-trip (English -> Arabic dialect -> English)"
        else:
            dir_label = direction
        print(f"\n{'='*80}")
        print(f"📂 Direction: {dir_label} [{direction}]")
        print(f"{'='*80}")
        
        file_test_sets = data.get('file_test_sets', {})
        merged_test_sets = data.get('merged_test_sets', {})
        
        if not file_test_sets and not merged_test_sets:
            print(f"⚠️  No test sets found for direction: {direction}")
            continue
        
        print(f"✅ Found {len(file_test_sets)} individual test set(s) and {len(merged_test_sets)} merged dialect(s)")
        if file_test_sets:
            print(f"   Individual test sets: {', '.join(sorted(file_test_sets.keys())[:5])}{'...' if len(file_test_sets) > 5 else ''}")
        if merged_test_sets:
            print(f"   Merged dialects: {', '.join(sorted(merged_test_sets.keys()))}")
        
        # Use a direction-specific output subdirectory to avoid overwriting
        direction_output_dir = output_dir / direction
        direction_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use file_test_sets as the main test_sets for backward compatibility with existing functions
        test_sets = file_test_sets
        
        # For forward direction, create separate outputs for general Arabic and dialect
        if direction == 'forward':
            for score_type in ['arabic_general', 'dialect']:
                type_label = 'General Arabic' if score_type == 'arabic_general' else 'Dialect'
                type_output_dir = direction_output_dir / score_type
                type_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create stats file for this type
                stats_file_path = type_output_dir / "statistics.txt"
                with open(stats_file_path, 'w', encoding='utf-8') as stats_file:
                    stats_file.write(f"{'='*80}\n")
                    stats_file.write(f"Statistics for {type_label} Translations\n")
                    stats_file.write(f"Direction: {dir_label}\n")
                    stats_file.write(f"{'='*80}\n\n")
                    
                    print(f"\n🎨 Creating visualizations for {type_label}...")
                    
                    # Create plots for each individual test set
                    print(f"\n📊 Creating per-test-set plots ({type_label})...")
                    for test_set, model_scores in file_test_sets.items():
                        if args.metric in ['BLEU', 'both']:
                            create_bar_plot_by_type(test_set, model_scores, models, type_output_dir, score_type, 'BLEU')
                        if args.metric in ['CHRF', 'both']:
                            create_bar_plot_by_type(test_set, model_scores, models, type_output_dir, score_type, 'CHRF')
                    
                    # Create combined plot with all test sets side by side
                    print(f"\n📊 Creating combined plot - all test sets ({type_label})...")
                    if args.metric in ['BLEU', 'both']:
                        create_all_testsets_combined_plot_by_type(test_sets, models, type_output_dir, score_type, 'BLEU')
                    if args.metric in ['CHRF', 'both']:
                        create_all_testsets_combined_plot_by_type(test_sets, models, type_output_dir, score_type, 'CHRF')
                    
                    # Create dialect-grouped plots
                    print(f"\n📊 Creating per-dialect grouped plots ({type_label})...")
                    if args.metric in ['BLEU', 'both']:
                        create_dialect_grouped_plot(test_sets, models, type_output_dir, score_type, 'BLEU')
                    if args.metric in ['CHRF', 'both']:
                        create_dialect_grouped_plot(test_sets, models, type_output_dir, score_type, 'CHRF')
                    
                    # Write score tables to stats file
                    stats_file.write("\n" + "=" * 80 + "\n")
                    stats_file.write("SCORE TABLES\n")
                    stats_file.write("=" * 80 + "\n")
                    if args.metric in ['BLEU', 'both']:
                        write_scores_table(test_sets, models, score_type, 'BLEU', stats_file, 
                                          f"\n{type_label} BLEU Scores by Test Set")
                    if args.metric in ['CHRF', 'both']:
                        write_scores_table(test_sets, models, score_type, 'CHRF', stats_file, 
                                          f"\n{type_label} CHRF Scores by Test Set")
                    
                    # Print average scores
                    print(f"\n📊 Average Scores ({type_label})...")
                    if args.metric in ['BLEU', 'both']:
                        print_average_scores_by_type(test_sets, models, score_type, 'BLEU', stats_file)
                    if args.metric in ['CHRF', 'both']:
                        print_average_scores_by_type(test_sets, models, score_type, 'CHRF', stats_file)
                    
                    # Print ranking statistics
                    print(f"\n📈 Ranking Statistics ({type_label})...")
                    if args.metric in ['BLEU', 'both']:
                        print_ranking_stats_by_type(test_sets, models, score_type, 'BLEU', stats_file)
                    if args.metric in ['CHRF', 'both']:
                        print_ranking_stats_by_type(test_sets, models, score_type, 'CHRF', stats_file)
                    
                    # Print average rankings
                    print(f"\n📊 Average Rankings ({type_label})...")
                    if args.metric in ['BLEU', 'both']:
                        print_average_rankings(test_sets, models, score_type, 'BLEU', stats_file)
                    if args.metric in ['CHRF', 'both']:
                        print_average_rankings(test_sets, models, score_type, 'CHRF', stats_file)
                    
                    # Print overall scores
                    overall_scores = data.get('overall_scores', {})
                    if overall_scores:
                        print(f"\n📊 Overall Scores ({type_label})...")
                        print_overall_scores(overall_scores, models, direction, score_type, stats_file)
                    
                    # Create summary plots (heatmaps)
                    if args.summary:
                        print(f"\n📊 Creating summary heatmaps ({type_label})...")
                        if args.metric in ['BLEU', 'both']:
                            create_summary_plot(test_sets, models, type_output_dir, 'BLEU')
                        if args.metric in ['CHRF', 'both']:
                            create_summary_plot(test_sets, models, type_output_dir, 'CHRF')
                
                print(f"✅ Statistics saved to: {stats_file_path}")
            
            # Print general/dialect difference statistics (comparison between the two)
            diff_stats_file_path = direction_output_dir / "general_vs_dialect_differences.txt"
            with open(diff_stats_file_path, 'w', encoding='utf-8') as diff_stats_file:
                diff_stats_file.write(f"{'='*80}\n")
                diff_stats_file.write(f"General Arabic vs Dialect Differences\n")
                diff_stats_file.write(f"Direction: {dir_label}\n")
                diff_stats_file.write(f"{'='*80}\n\n")
                
                print(f"\n📈 Calculating General Arabic vs Dialect differences...")
                if args.metric in ['BLEU', 'both']:
                    print_general_dialect_difference(test_sets, models, 'BLEU', diff_stats_file)
                if args.metric in ['CHRF', 'both']:
                    print_general_dialect_difference(test_sets, models, 'CHRF', diff_stats_file)
            
            print(f"✅ Difference statistics saved to: {diff_stats_file_path}")
        
        else:
            # For reverse/roundtrip, use the standard plots (no general/dialect split)
            stats_file_path = direction_output_dir / "statistics.txt"
            with open(stats_file_path, 'w', encoding='utf-8') as stats_file:
                stats_file.write(f"{'='*80}\n")
                stats_file.write(f"Statistics for {dir_label}\n")
                stats_file.write(f"{'='*80}\n\n")
                
                print(f"\n🎨 Creating visualizations for individual test sets...")
                
                # Create plots for each individual test set
                for test_set, model_scores in file_test_sets.items():
                    if args.metric in ['BLEU', 'both']:
                        create_bar_plot(test_set, model_scores, models, direction_output_dir, 'BLEU')
                    if args.metric in ['CHRF', 'both']:
                        create_bar_plot(test_set, model_scores, models, direction_output_dir, 'CHRF')
                
                # Create combined plot with all test sets side by side
                print(f"\n📊 Creating combined plot (all test sets side by side)...")
                if args.metric in ['BLEU', 'both']:
                    create_all_testsets_combined_plot(test_sets, models, direction_output_dir, 'BLEU')
                if args.metric in ['CHRF', 'both']:
                    create_all_testsets_combined_plot(test_sets, models, direction_output_dir, 'CHRF')
                
                # Write score tables to stats file
                stats_file.write("\n" + "=" * 80 + "\n")
                stats_file.write("SCORE TABLES\n")
                stats_file.write("=" * 80 + "\n")
                if args.metric in ['BLEU', 'both']:
                    write_scores_table(test_sets, models, None, 'BLEU', stats_file, 
                                      "\nBLEU Scores by Test Set")
                if args.metric in ['CHRF', 'both']:
                    write_scores_table(test_sets, models, None, 'CHRF', stats_file, 
                                      "\nCHRF Scores by Test Set")
                
                # Print overall ranking statistics
                print(f"\n📈 Overall Ranking Statistics...")
                if args.metric in ['BLEU', 'both']:
                    print_ranking_stats(test_sets, models, 'BLEU', stats_file)
                if args.metric in ['CHRF', 'both']:
                    print_ranking_stats(test_sets, models, 'CHRF', stats_file)
                
                # Print overall average rankings
                print(f"\n📊 Overall Average Rankings...")
                if args.metric in ['BLEU', 'both']:
                    print_average_rankings(test_sets, models, None, 'BLEU', stats_file)
                if args.metric in ['CHRF', 'both']:
                    print_average_rankings(test_sets, models, None, 'CHRF', stats_file)
                
                # Print overall scores
                overall_scores = data.get('overall_scores', {})
                if overall_scores:
                    print(f"\n📊 Overall Scores (All Test Sets Concatenated)...")
                    print_overall_scores(overall_scores, models, direction, None, stats_file)
                
                # Create aggregated plots (averages and grouped)
                print(f"\n📊 Creating aggregated plots...")
                if args.metric in ['BLEU', 'both']:
                    create_average_plot(test_sets, models, direction_output_dir, 'BLEU')
                    create_all_testsets_plot(test_sets, models, direction_output_dir, 'BLEU')
                if args.metric in ['CHRF', 'both']:
                    create_average_plot(test_sets, models, direction_output_dir, 'CHRF')
                    create_all_testsets_plot(test_sets, models, direction_output_dir, 'CHRF')
                
                # Create summary plots (heatmaps)
                if args.summary:
                    print(f"\n📊 Creating summary heatmaps...")
                    if args.metric in ['BLEU', 'both']:
                        create_summary_plot(test_sets, models, direction_output_dir, 'BLEU')
                    if args.metric in ['CHRF', 'both']:
                        create_summary_plot(test_sets, models, direction_output_dir, 'CHRF')
            
            print(f"✅ Statistics saved to: {stats_file_path}")
        
        # Create visualizations for merged dialects (only forward direction has these)
        if merged_test_sets and direction == 'forward':
            print(f"\n🎨 Creating visualizations for merged dialects...")
            
            # Check which models have merged dialect results
            models_with_merged = set()
            for dialect_name, model_scores in merged_test_sets.items():
                models_with_merged.update(model_scores.keys())
            
            models_missing_merged = set(models) - models_with_merged
            if models_missing_merged:
                print(f"⚠️  Warning: {len(models_missing_merged)} model(s) missing merged dialect results:")
                print(f"   {', '.join(sorted(models_missing_merged))}")
                print(f"   These models need to re-run the evaluation script to compute merged dialect scores.")
                print(f"   Only {len(models_with_merged)} model(s) will appear in merged dialect visualizations.\n")
            
            # Create separate outputs for general Arabic and dialect (merged)
            for score_type in ['arabic_general', 'dialect']:
                type_label = 'General Arabic' if score_type == 'arabic_general' else 'Dialect'
                type_output_dir = direction_output_dir / score_type / 'merged'
                type_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create stats file for merged dialects
                stats_file_path = type_output_dir / "statistics.txt"
                with open(stats_file_path, 'w', encoding='utf-8') as stats_file:
                    stats_file.write(f"{'='*80}\n")
                    stats_file.write(f"Statistics for Merged Dialects ({type_label})\n")
                    stats_file.write(f"Direction: {dir_label}\n")
                    stats_file.write(f"{'='*80}\n\n")
                    
                    print(f"\n📊 Creating merged dialect plots ({type_label})...")
                    
                    # Create plots for each merged dialect
                    for dialect_name, model_scores in merged_test_sets.items():
                        if args.metric in ['BLEU', 'both']:
                            create_bar_plot_by_type(f"{dialect_name} (merged)", model_scores, models, type_output_dir, score_type, 'BLEU')
                        if args.metric in ['CHRF', 'both']:
                            create_bar_plot_by_type(f"{dialect_name} (merged)", model_scores, models, type_output_dir, score_type, 'CHRF')
                    
                    # Create combined plot with all merged dialects side by side
                    print(f"\n📊 Creating combined plot - all merged dialects ({type_label})...")
                    if args.metric in ['BLEU', 'both']:
                        create_all_testsets_combined_plot_by_type(merged_test_sets, models, type_output_dir, score_type, 'BLEU')
                    if args.metric in ['CHRF', 'both']:
                        create_all_testsets_combined_plot_by_type(merged_test_sets, models, type_output_dir, score_type, 'CHRF')
                    
                    # Write score tables to stats file
                    stats_file.write("\n" + "=" * 80 + "\n")
                    stats_file.write("MERGED DIALECT SCORE TABLES\n")
                    stats_file.write("=" * 80 + "\n")
                    if args.metric in ['BLEU', 'both']:
                        write_scores_table(merged_test_sets, models, score_type, 'BLEU', stats_file, 
                                          f"\n{type_label} BLEU Scores for Merged Dialects")
                    if args.metric in ['CHRF', 'both']:
                        write_scores_table(merged_test_sets, models, score_type, 'CHRF', stats_file, 
                                          f"\n{type_label} CHRF Scores for Merged Dialects")
                    
                    # Print statistics for merged dialects
                    print(f"\n📊 Average Scores for Merged Dialects ({type_label})...")
                    if args.metric in ['BLEU', 'both']:
                        print_average_scores_by_type(merged_test_sets, models, score_type, 'BLEU', stats_file)
                    if args.metric in ['CHRF', 'both']:
                        print_average_scores_by_type(merged_test_sets, models, score_type, 'CHRF', stats_file)
                    
                    # Print ranking statistics for merged dialects
                    print(f"\n📈 Ranking Statistics for Merged Dialects ({type_label})...")
                    if args.metric in ['BLEU', 'both']:
                        print_ranking_stats_by_type(merged_test_sets, models, score_type, 'BLEU', stats_file)
                    if args.metric in ['CHRF', 'both']:
                        print_ranking_stats_by_type(merged_test_sets, models, score_type, 'CHRF', stats_file)
                    
                    # Print average rankings for merged dialects
                    print(f"\n📊 Average Rankings for Merged Dialects ({type_label})...")
                    if args.metric in ['BLEU', 'both']:
                        print_average_rankings(merged_test_sets, models, score_type, 'BLEU', stats_file)
                    if args.metric in ['CHRF', 'both']:
                        print_average_rankings(merged_test_sets, models, score_type, 'CHRF', stats_file)
                
                print(f"✅ Merged dialect statistics saved to: {stats_file_path}")
        
        print(f"\n{'='*80}")
        print(f"✅ Visualization complete for direction: {direction}")
        print(f"📁 Output directory: {direction_output_dir}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()


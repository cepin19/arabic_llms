# Generated LaTeX Tables and Figures

This directory contains automatically generated LaTeX tables and figures extracted from the experimental results in `fixed_translations/` and visualizations in `visualizations_fixed/`.

## Files Generated

### Tables

1. **forward_table.tex** - Overall forward translation (English → Arabic) performance across all models, showing both Arabic General and Dialect-specific scores for BLEU and CHRF metrics. Models are grouped by category (Arabic-Specialized, Multilingual, etc.).

2. **dialect_table.tex** - Dialect-specific performance breakdown showing how each model performs on individual Arabic dialects (Iraqi, Qatari, Tunisian, Moroccan, Omani, MSA, Algerian, Egyptian, etc.).

3. **reverse_table.tex** - Reverse translation (Arabic → English) performance across all models, grouped by category.

4. **roundtrip_table.tex** - Roundtrip translation (English → Arabic → English) results showing top model pairs and their performance.

### Figures

5. **figures.tex** - LaTeX code for including key visualization figures:
   - Forward translation: Combined overview plots for Arabic General and Dialect variants
   - Reverse translation: Combined performance plots
   - Roundtrip translation: Performance plots for different model combinations

### Integration

6. **results_section.tex** - Complete LaTeX section that includes all tables and figures in a structured format. This can be directly included in the main paper.

## Usage

To include these in your paper, add to your main LaTeX file:

```latex
\input{paper_tables/results_section.tex}
```

Or include individual components:

```latex
\input{paper_tables/forward_table.tex}
\input{paper_tables/dialect_table.tex}
\input{paper_tables/reverse_table.tex}
\input{paper_tables/roundtrip_table.tex}
\input{paper_tables/figures.tex}
```

## Regenerating

To regenerate these files with updated results, run:

```bash
python3 generate_paper_tables.py \
    --fixed-translations-dir fixed_translations \
    --visualizations-dir visualizations_fixed \
    --output-dir paper_tables
```

To include COMET scores (if available in the results):

```bash
python3 generate_paper_tables.py \
    --fixed-translations-dir fixed_translations \
    --visualizations-dir visualizations_fixed \
    --output-dir paper_tables \
    --include-comet
```

## Notes

- Tables use `booktabs` package for professional formatting
- Tables are automatically resized to fit page width
- Figures are set to 0.9\textwidth for readability
- Model names are automatically escaped for LaTeX compatibility
- Numbers are formatted to 2 decimal places


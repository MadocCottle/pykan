# Section 1 LaTeX Document

## Overview

This directory contains the complete LaTeX document for Section 1 of the honours thesis on Kolmogorov-Arnold Networks.

## Files

- **section.tex**: Main LaTeX source file (13 pages)
- **section.pdf**: Compiled PDF document (3.5 MB)

## Document Structure

### 1. Abstract
Comprehensive abstract covering all three experimental sections (1.1, 1.2, 1.3) and key findings.

### 2. Introduction
- Motivation for KANs vs traditional neural networks
- Mathematical foundation (Kolmogorov-Arnold representation theorem)
- Research questions and contributions

### 3. Methods
- **Neural Network Architectures**:
  - Multi-Layer Perceptron (MLP) - mathematical formulation with tanh, ReLU, SiLU activations
  - SIREN - sinusoidal activation networks
  - Kolmogorov-Arnold Networks (KAN) - B-spline basis formulation
  - KAN with pruning

- **Experimental Design**:
  - Section 1.1: Function approximation (9 tasks: sinusoids, piecewise, sawtooth, polynomial, high-freq)
  - Section 1.2: 1D Poisson PDE (3 forcing functions)
  - Section 1.3: 2D Poisson PDE (4 forcing functions)

- **Training Procedure**: L-BFGS optimizer, MSE loss
- **Hyperparameters**: KAN grid sizes (3-100), MLP depths (2-6), activation functions
- **Evaluation Metrics**: Train/Test/Dense MSE, training time

### 4. Results
- **Section 1.1 Results**: Heatmaps, learning curves, function fits
- **Section 1.2 Results**: 1D PDE solutions with visualizations
- **Section 1.3 Results**: 2D PDE solutions with cross-sections and surface plots

All figures are included from the analysis directory:
`../section1/analysis/section1_complete_analysis_20251021_143055/`

### 5. Discussion
- Key findings synthesis
- When to use KANs
- Limitations and future work

## Included Figures

The document includes:
- **Heatmaps**: Performance comparison across all models and datasets
- **Learning curves**: Convergence behavior over training epochs
- **Function fitting plots**: 1D and 2D visualizations
- **Cross-sections**: Detailed error analysis for 2D problems
- **Surface plots**: 3D visualizations of 2D PDE solutions

## Placeholder Text

The document contains placeholder text marked with:
```
\textbf{[PLACEHOLDER: Description of what analysis to add]}
```

These placeholders indicate where:
- Detailed analysis of results should be written
- Quantitative comparisons should be added
- Statistical findings should be discussed
- Tables with numerical data should be inserted

## References

The document cites key papers from `references.md`:
- Liu et al. (2024) - KAN: Kolmogorov-Arnold Networks
- Sitzmann et al. (2020) - SIREN
- Kingma & Ba (2015) - Adam optimizer
- Liu & Nocedal (1989) - L-BFGS
- Raissi et al. (2019) - Physics-informed neural networks

## Compilation

To compile the PDF:
```bash
cd /Users/main/Desktop/my_pykan/pykan/madoc/papers
pdflatex section.tex
pdflatex section.tex  # Run twice for cross-references
```

## Next Steps

To complete the document:
1. Fill in all PLACEHOLDER sections with actual analysis
2. Add quantitative summary tables (Template provided)
3. Complete the discussion section with specific findings
4. Review and refine mathematical notation
5. Add acknowledgments
6. Verify all figure references are correct

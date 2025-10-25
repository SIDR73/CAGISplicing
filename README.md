# CAGI7 Splicing Prediction Pipeline

This repository contains a streamlined Python implementation of the DrASNet methodology for predicting splicing variants in the CAGI7 challenge.

## Overview

The pipeline implements the DrASNet approach which:
1. Extracts comprehensive features from variant data
2. Integrates PPI network information for trans-regulation
3. Trains ensemble machine learning models
4. Generates predictions with confidence scores

## Quick Start

### 1. Extract Features (One-time setup)
```bash
python extract_features.py
```
This creates:
- `train_features.csv` - Training data with all features
- `test_features.csv` - Test data with all features  
- `feature_summary.csv` - Summary of all features

### 2. Run Machine Learning Pipeline
```bash
# Fast mode (smaller models, ~30 seconds)
python run_ml.py --fast

# Full mode (larger models, ~2-3 minutes)
python run_ml.py

# With DrASNet greedy prioritization
python run_ml.py --greedy

# Custom prediction threshold
python run_ml.py --threshold 0.2
```

## Files

### Core Pipeline
- `extract_features.py` - Feature extraction script (run once)
- `ml_pipeline.py` - Machine learning pipeline
- `run_ml.py` - Main entry point for ML pipeline
- `greedy_prioritization.py` - DrASNet greedy algorithm for driver mutation prioritization

### Utilities
- `requirements.txt` - Python dependencies

## Data Requirements

### Required Files
- `cagi7splicingsample.csv` - Training data with PSI profiles
- `cagi7splicingvariants.csv` - Test variants for prediction
- `DrASNet_data/input_data/network.txt` - PPI network data

### Optional Files
- `gencode.v47.annotation.gtf.gz` - Gene annotations (for better gene mapping)
- `train_gene_mapping.csv` - Pre-computed gene mappings
- `test_gene_mapping.csv` - Pre-computed gene mappings

## Features

The pipeline extracts comprehensive features including:

### Basic Variant Features
- **Position**: Genomic position with multiple encodings
- **Allele Type**: Transition/transversion classification
- **Oligo Type**: Periexonic vs deep intronic
- **Chromosome**: Autosome vs sex chromosome

### Enhanced Features
- **Position Encodings**: Multiple scales (mod 1000, 100, 10, log)
- **Allele Length**: Reference and alternative allele lengths
- **Length Difference**: Insertion/deletion size

### Splicing Features (Training Only)
- **PSI Measurements**: Reference and variant PSI values
- **Splicing Changes**: Delta PSI, magnitude, direction
- **Derived Features**: PSI ratios and change patterns

### Network Features
- **PPI Network**: Degree, betweenness, closeness centrality
- **Network Membership**: Whether gene is in the network

## Output

The pipeline generates:
- `ml_predictions.csv` - Final predictions with confidence scores
- `feature_importance.csv` - Analysis of feature importance
- `feature_summary.csv` - Summary of all extracted features

## Machine Learning

### Ensemble Approach
- **Random Forest**: 200 trees, depth 15, balanced parameters
- **Gradient Boosting**: 200 estimators, learning rate 0.1
- **Logistic Regression**: Balanced class weights, L2 regularization
- **Voting Classifier**: Soft voting for final predictions

### Performance Modes
- **Fast Mode**: Smaller models (50 estimators) for quick testing
- **Full Mode**: Larger models (200 estimators) for best performance
- **Configurable Threshold**: Default 0.1, adjustable via command line

## Methodology

### DrASNet Integration
1. **Feature Engineering**: Comprehensive variant and network features
2. **PPI Network**: Real protein-protein interaction network
3. **Gene Mapping**: Genomic coordinate to gene name mapping
4. **Ensemble Learning**: Multiple models for robust predictions

### Feature Categories
- **Basic Features**: ~8 (position, allele, oligo type)
- **Enhanced Features**: ~6 (position encodings, chromosome type)
- **Splicing Features**: ~9 (PSI measurements and derivatives)
- **Network Features**: ~4 (degree, betweenness, closeness, membership)
- **Total**: ~27 features

## Performance

### Fast Mode
- **Training Time**: ~30 seconds
- **Models**: 50 estimators each
- **Use Case**: Quick testing and development

### Full Mode  
- **Training Time**: ~2-3 minutes
- **Models**: 200 estimators each
- **Use Case**: Final predictions and submission

## Troubleshooting

### Common Issues
1. **Missing feature files**: Run `python extract_features.py` first
2. **Missing network file**: Pipeline will use simple features only
3. **Gene mapping errors**: Check Gencode file format

### Debugging
- Check `feature_summary.csv` for feature statistics
- Use `--fast` flag for quick testing
- Verify all required CSV files exist

## Citation

If you use this pipeline, please cite the original DrASNet paper and the CAGI7 challenge.

## License

This project is licensed under the MIT License.
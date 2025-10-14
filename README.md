# DrASNet-inspired Splicing Variant Prediction Pipeline

This pipeline implements a machine learning approach for predicting splicing variant effects, inspired by the DrASNet methodology for identifying driver mutations that cause alternative splicing changes.

## Overview

The pipeline combines:
1. **DrASNet methodology**: Network-based identification of driver mutations
2. **Feature engineering**: Comprehensive variant and network-based features
3. **Ensemble learning**: Multiple machine learning models for robust predictions
4. **Greedy algorithm**: Driver mutation prioritization

## Files

- `advanced_splicing_pipeline.py`: Main pipeline implementation
- `splicing_prediction_pipeline.py`: Basic pipeline version
- `run_pipeline.py`: Simple runner script
- `requirements.txt`: Python dependencies
- `cagi7splicingsample.csv`: Training data
- `cagi7splicingvariants.csv`: Test variants for prediction

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
python run_pipeline.py
```

### Advanced Usage
```python
from advanced_splicing_pipeline import AdvancedDrASNetPipeline

# Initialize pipeline
pipeline = AdvancedDrASNetPipeline()

# Load data
pipeline.load_data('cagi7splicingsample.csv', 'cagi7splicingvariants.csv')

# Run complete pipeline
pipeline.create_ppi_network()
pipeline.identify_personalized_as_events()
pipeline.identify_mutation_as_pairs()
pipeline.prioritize_driver_mutations()
pipeline.create_advanced_features()
pipeline.train_ensemble_models()

# Generate predictions
predictions = pipeline.generate_final_predictions()
```

## Methodology

### 1. Personalized AS Event Identification
- Identifies variants with significant splicing changes (|ΔPSI| > 0.1)
- Maps variants to genes and creates AS event profiles

### 2. Mutation-AS Pair Identification
- **Cis regulation**: Mutations in the same gene as the splicing event
- **Trans regulation**: Mutations in genes connected to the splicing gene via PPI network

### 3. Driver Mutation Prioritization
- Implements greedy algorithm to identify key driver mutations
- Selects mutations that control the most splicing events in the network

### 4. Feature Engineering
- **Variant features**: Position, allele type, oligo type
- **Network features**: Degree, betweenness, closeness centrality
- **Driver features**: Driver mutation status and scores
- **Splicing features**: PSI values, ΔPSI, PRES scores

### 5. Ensemble Learning
- Random Forest Classifier
- Gradient Boosting Classifier  
- Logistic Regression
- Voting ensemble with soft voting

## Output

The pipeline generates:
- `advanced_splicing_predictions.csv`: Final predictions with scores
- Console output with performance metrics and summary statistics

## Key Features

- **Network-based approach**: Leverages protein-protein interaction networks
- **Driver mutation prioritization**: Identifies functional vs. passenger mutations
- **Comprehensive feature set**: Combines multiple data types
- **Ensemble methodology**: Robust predictions from multiple models
- **Scalable design**: Handles large variant datasets

## Performance

The pipeline includes:
- Cross-validation for model evaluation
- Validation set performance metrics
- ROC-AUC scoring for binary classification
- Feature importance analysis

## Customization

The pipeline can be customized by:
- Modifying feature engineering functions
- Adding new machine learning models
- Incorporating additional network databases
- Adjusting hyperparameters

## References

Based on the DrASNet methodology:
- Sahni, N., et al. "Widespread macromolecular interaction perturbations in human genetic disorders." Cell 161.3 (2015): 647-660.
- DrASNet R implementation for cancer splicing analysis

## Contact

For questions or issues, please refer to the original DrASNet authors or create an issue in this repository.

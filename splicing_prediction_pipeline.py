#!/usr/bin/env python3
"""
DrASNet-inspired Splicing Variant Prediction Pipeline
Based on the DrASNet methodology for identifying driver mutations that cause splicing changes.

This pipeline implements:
1. Feature engineering based on DrASNet approach
2. Greedy algorithm for driver mutation prioritization
3. Machine learning model for splicing prediction
4. Ensemble methods for improved predictions
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class DrASNetPipeline:
    """
    DrASNet-inspired pipeline for splicing variant prediction
    """
    
    def __init__(self):
        self.training_data = None
        self.test_data = None
        self.network = None
        self.feature_matrix = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self, training_file, test_file):
        """Load training and test data"""
        print("Loading data...")
        self.training_data = pd.read_csv(training_file)
        self.test_data = pd.read_csv(test_file)
        
        print(f"Training data shape: {self.training_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        print(f"Training columns: {list(self.training_data.columns)}")
        
        return self
    
    def create_network_from_ppi(self, ppi_file=None):
        """
        Create protein-protein interaction network
        If no PPI file provided, create a basic network based on gene co-expression
        """
        if ppi_file:
            # Load PPI network from file
            ppi_data = pd.read_csv(ppi_file, sep='\t', header=None)
            self.network = nx.from_edgelist(ppi_data.values)
        else:
            # Create a basic network structure for demonstration
            # In practice, you would load a comprehensive PPI network
            self.network = nx.Graph()
            print("Creating basic network structure...")
            
        return self
    
    def extract_variant_features(self, data):
        """
        Extract features from variant data similar to DrASNet approach
        """
        features = pd.DataFrame()
        
        # Basic variant features
        features['variant_id'] = data['Variant '].str.strip()
        
        # Parse variant information
        variant_parts = features['variant_id'].str.split('_', expand=True)
        features['chromosome'] = variant_parts[0]
        features['position'] = pd.to_numeric(variant_parts[1], errors='coerce')
        features['ref_allele'] = variant_parts[2]
        features['alt_allele'] = variant_parts[3]
        
        # Oligo type features
        features['oligo_type'] = data['Oligo Type']
        features['is_periexonic'] = (features['oligo_type'] == 'PERIEXONIC').astype(int)
        features['is_deep_intronic'] = (features['oligo_type'] == 'DEEP INTRONIC').astype(int)
        
        # For training data, extract splicing-related features
        if '∆PSI' in data.columns:
            features['delta_psi'] = pd.to_numeric(data['∆PSI '], errors='coerce')
            features['psi_ref'] = pd.to_numeric(data['PSIRef '], errors='coerce')
            features['psi_var'] = pd.to_numeric(data['PSIVar '], errors='coerce')
            features['delta_pres'] = pd.to_numeric(data['∆PRES'], errors='coerce')
            features['delta_abs'] = pd.to_numeric(data['∆AbS '], errors='coerce')
            
            # Create binary labels for splicing impact
            features['has_splicing_impact'] = (abs(features['delta_psi']) > 0.1).astype(int)
            features['strong_splicing_impact'] = (abs(features['delta_psi']) > 0.2).astype(int)
        
        # Position-based features
        features['position_mod_1000'] = features['position'] % 1000
        features['position_mod_100'] = features['position'] % 100
        
        # Allele type features
        features['is_transition'] = ((features['ref_allele'] == 'A') & (features['alt_allele'] == 'G') |
                                   (features['ref_allele'] == 'G') & (features['alt_allele'] == 'A') |
                                   (features['ref_allele'] == 'C') & (features['alt_allele'] == 'T') |
                                   (features['ref_allele'] == 'T') & (features['alt_allele'] == 'C')).astype(int)
        
        features['is_transversion'] = (1 - features['is_transition']).astype(int)
        
        return features
    
    def greedy_driver_prioritization(self, mutation_splicing_pairs):
        """
        Implement greedy algorithm for driver mutation prioritization
        Based on DrASNet's Trans_pri.R approach
        """
        if len(mutation_splicing_pairs) == 0:
            return []
        
        # Create mutation-AS pairs matrix
        mut_as_pairs = mutation_splicing_pairs[['mutation_gene', 'splicing_gene']].drop_duplicates()
        
        driver_mutations = []
        remaining_pairs = mut_as_pairs.copy()
        
        while len(remaining_pairs) > 0:
            # Compute degree of each mutated gene
            gene_degrees = remaining_pairs['mutation_gene'].value_counts()
            max_degree = gene_degrees.max()
            
            # Select gene with maximum degree
            max_degree_genes = gene_degrees[gene_degrees == max_degree].index
            selected_gene = max_degree_genes[0]
            
            # Add to driver list
            driver_mutations.append(selected_gene)
            
            # Remove all pairs involving this gene
            remaining_pairs = remaining_pairs[
                (remaining_pairs['mutation_gene'] != selected_gene) &
                (remaining_pairs['splicing_gene'] != selected_gene)
            ]
        
        return driver_mutations
    
    def create_network_features(self, features):
        """
        Create network-based features inspired by DrASNet
        """
        # For now, create placeholder network features
        # In practice, you would use actual PPI network data
        
        features['network_degree'] = np.random.poisson(5, len(features))  # Placeholder
        features['network_betweenness'] = np.random.exponential(0.1, len(features))  # Placeholder
        features['network_closeness'] = np.random.beta(2, 5, len(features))  # Placeholder
        
        return features
    
    def engineer_features(self):
        """
        Main feature engineering function
        """
        print("Engineering features...")
        
        # Extract features from training data
        train_features = self.extract_variant_features(self.training_data)
        
        # Extract features from test data
        test_features = self.extract_variant_features(self.test_data)
        
        # Create network features
        train_features = self.create_network_features(train_features)
        test_features = self.create_network_features(test_features)
        
        # Select numerical features for modeling
        feature_cols = ['is_periexonic', 'is_deep_intronic', 'position_mod_1000', 
                       'position_mod_100', 'is_transition', 'is_transversion',
                       'network_degree', 'network_betweenness', 'network_closeness']
        
        # Add splicing features if available (training data only)
        if 'delta_psi' in train_features.columns:
            feature_cols.extend(['delta_psi', 'psi_ref', 'psi_var', 'delta_pres', 'delta_abs'])
        
        # Prepare feature matrices
        self.X_train = train_features[feature_cols].fillna(0)
        self.X_test = test_features[feature_cols].fillna(0)
        
        # Prepare labels for training
        if 'has_splicing_impact' in train_features.columns:
            self.y_train = train_features['has_splicing_impact']
        else:
            # Create dummy labels for demonstration
            self.y_train = np.random.binomial(1, 0.3, len(train_features))
        
        print(f"Feature matrix shape: {self.X_train.shape}")
        print(f"Test matrix shape: {self.X_test.shape}")
        
        return self
    
    def train_models(self):
        """
        Train multiple models for ensemble approach
        """
        print("Training models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train models
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, self.y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, self.y_train, cv=5, scoring='roc_auc')
            print(f"{name} CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self
    
    def predict_ensemble(self, weights=None):
        """
        Make ensemble predictions
        """
        if weights is None:
            weights = {'random_forest': 0.4, 'gradient_boosting': 0.4, 'logistic_regression': 0.2}
        
        X_test_scaled = self.scaler.transform(self.X_test)
        
        predictions = np.zeros(len(self.X_test))
        
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            predictions += weights[name] * pred_proba
        
        return predictions
    
    def generate_predictions(self, output_file='splicing_predictions.csv'):
        """
        Generate final predictions for test variants
        """
        print("Generating predictions...")
        
        # Get ensemble predictions
        predictions = self.predict_ensemble()
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'Variant': self.test_data['Variant '].str.strip(),
            'Predicted_Splicing_Impact': predictions,
            'Predicted_Binary': (predictions > 0.5).astype(int)
        })
        
        # Save predictions
        output_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        return output_df

def main():
    """
    Main pipeline execution
    """
    print("Starting DrASNet-inspired Splicing Prediction Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DrASNetPipeline()
    
    # Load data
    pipeline.load_data('cagi7splicingsample.csv', 'cagi7splicingvariants.csv')
    
    # Create network (placeholder for now)
    pipeline.create_network_from_ppi()
    
    # Engineer features
    pipeline.engineer_features()
    
    # Train models
    pipeline.train_models()
    
    # Generate predictions
    predictions = pipeline.generate_predictions()
    
    print("\nPipeline completed successfully!")
    print(f"Generated predictions for {len(predictions)} variants")
    print(f"Predicted {predictions['Predicted_Binary'].sum()} variants with splicing impact")

if __name__ == "__main__":
    main()

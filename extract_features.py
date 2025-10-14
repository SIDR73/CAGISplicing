#!/usr/bin/env python3
"""
Feature Extraction Script
Extracts and saves all features to CSV files for faster ML development.
"""

import pandas as pd
import numpy as np
import networkx as nx
import os
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor:
    """
    Extract features and save to CSV files
    """
    
    def __init__(self):
        self.training_data = None
        self.test_data = None
        self.network = None
        self.variant_gene_map = None
        
    def load_data(self, training_file, test_file):
        """Load training and test data"""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        self.training_data = pd.read_csv(training_file)
        self.test_data = pd.read_csv(test_file)
        
        print(f"✓ Training: {self.training_data.shape[0]} variants")
        print(f"✓ Test: {self.test_data.shape[0]} variants")
        return self
    
    def load_network(self):
        """Load PPI network and pre-compute centrality measures"""
        print("\n" + "=" * 60)
        print("LOADING NETWORK")
        print("=" * 60)
        
        network_file = 'DrASNet_data/input_data/network.txt'
        if os.path.exists(network_file):
            ppi_data = pd.read_csv(network_file, sep='\t', header=None, names=['gene1', 'gene2'])
            self.network = nx.from_edgelist(ppi_data.values)
            print(f"✓ Network: {self.network.number_of_nodes()} genes, {self.network.number_of_edges()} edges")
            
            # Pre-compute centrality measures (this is the optimization!)
            print("  Computing network centrality measures...")
            print("    - Degree centrality...", end=" ")
            self.degree_centrality = dict(self.network.degree())
            print("✓")
            
            print("    - Betweenness centrality...", end=" ")
            # Use sampling for faster betweenness calculation
            self.betweenness_centrality = nx.betweenness_centrality(self.network, k=min(100, self.network.number_of_nodes()))
            print("✓")
            
            print("    - Closeness centrality...", end=" ")
            # Use improved algorithm for faster closeness calculation
            self.closeness_centrality = nx.closeness_centrality(self.network, wf_improved=True)
            print("✓")
            
            print("✓ All network features pre-computed!")
        else:
            print("⚠ No network file, using simple features only")
            self.network = None
            self.degree_centrality = {}
            self.betweenness_centrality = {}
            self.closeness_centrality = {}
        return self
    
    def create_gene_mapping(self):
        """Create gene mapping"""
        print("\n" + "=" * 60)
        print("GENE MAPPING")
        print("=" * 60)
        
        if os.path.exists('train_gene_mapping.csv'):
            print("✓ Using existing Gencode mapping")
            train_mapping = pd.read_csv('train_gene_mapping.csv')
            test_mapping = pd.read_csv('test_gene_mapping.csv')
            
            self.variant_gene_map = {}
            for _, row in train_mapping.iterrows():
                self.variant_gene_map[row['variant_id']] = row['gene_name']
            for _, row in test_mapping.iterrows():
                self.variant_gene_map[row['variant_id']] = row['gene_name']
        else:
            print("✓ Creating simple mapping")
            self.variant_gene_map = {}
            for variant in pd.concat([self.training_data['Variant '], self.test_data['Variant ']]).unique():
                parts = variant.split('_')
                if len(parts) >= 4:
                    gene_id = f"GENE_{hash(variant) % 1000:03d}"
                    self.variant_gene_map[variant] = gene_id
        
        print(f"✓ Mapped {len(self.variant_gene_map)} variants")
        return self
    
    def extract_all_features(self):
        """Extract all features and save to CSV"""
        print("\n" + "=" * 60)
        print("EXTRACTING FEATURES")
        print("=" * 60)
        
        # Extract basic features
        print("  Extracting basic features...")
        train_features = self._extract_features(self.training_data, is_training=True)
        test_features = self._extract_features(self.test_data, is_training=False)
        print("  ✓ Basic features extracted")
        
        # Add network features (now super fast!)
        if self.network is not None:
            print("  Adding network features...")
            train_features = self._add_network_features(train_features)
            test_features = self._add_network_features(test_features)
            print("  ✓ Network features added")
        
        # Save to CSV
        print("\nSaving features to CSV files...")
        train_features.to_csv('train_features.csv', index=False)
        test_features.to_csv('test_features.csv', index=False)
        
        print(f"✓ Training features: {train_features.shape[1]} columns, {train_features.shape[0]} rows")
        print(f"✓ Test features: {test_features.shape[1]} columns, {test_features.shape[0]} rows")
        print(f"✓ Saved to: train_features.csv, test_features.csv")
        
        # Create feature summary
        self._create_feature_summary(train_features, test_features)
        
        return train_features, test_features
    
    def _extract_features(self, data, is_training=True):
        """Extract features from data using vectorized operations"""
        features = pd.DataFrame()
        
        # Basic variant info
        features['variant_id'] = data['Variant '].str.strip()
        features['gene_id'] = features['variant_id'].map(self.variant_gene_map)
        
        # Parse variant (vectorized)
        variant_parts = features['variant_id'].str.split('_', expand=True)
        features['chromosome'] = variant_parts[0]
        features['position'] = pd.to_numeric(variant_parts[1], errors='coerce')
        features['ref_allele'] = variant_parts[2]
        features['alt_allele'] = variant_parts[3]
        
        # Oligo type features (vectorized)
        features['oligo_type'] = data['Oligo Type']
        features['is_periexonic'] = (features['oligo_type'] == 'PERIEXONIC').astype(int)
        features['is_deep_intronic'] = (features['oligo_type'] == 'DEEP INTRONIC').astype(int)
        
        # Position features (vectorized)
        features['position_mod_1000'] = features['position'] % 1000
        features['position_mod_100'] = features['position'] % 100
        features['position_mod_10'] = features['position'] % 10
        features['position_log'] = np.log1p(features['position'])
        
        # Allele features (vectorized)
        features['is_transition'] = ((features['ref_allele'] == 'A') & (features['alt_allele'] == 'G') |
                                   (features['ref_allele'] == 'G') & (features['alt_allele'] == 'A') |
                                   (features['ref_allele'] == 'C') & (features['alt_allele'] == 'T') |
                                   (features['ref_allele'] == 'T') & (features['alt_allele'] == 'C')).astype(int)
        
        features['is_transversion'] = (1 - features['is_transition']).astype(int)
        
        # Chromosome features (vectorized)
        features['is_autosome'] = features['chromosome'].isin([str(i) for i in range(1, 23)]).astype(int)
        features['is_sex_chromosome'] = features['chromosome'].isin(['X', 'Y']).astype(int)
        
        # Allele length features (vectorized)
        features['ref_length'] = features['ref_allele'].str.len()
        features['alt_length'] = features['alt_allele'].str.len()
        features['length_diff'] = features['alt_length'] - features['ref_length']
        
        # Splicing features (training only)
        if is_training and '∆PSI ' in data.columns:
            features['delta_psi'] = pd.to_numeric(data['∆PSI '], errors='coerce').fillna(0)
            features['psi_ref'] = pd.to_numeric(data['PSIRef\xa0'], errors='coerce').fillna(0)
            features['psi_var'] = pd.to_numeric(data['PSIVar\xa0'], errors='coerce').fillna(0)
            features['delta_pres'] = pd.to_numeric(data['∆PRES'], errors='coerce').fillna(0)
            features['delta_abs'] = pd.to_numeric(data['∆AbS '], errors='coerce').fillna(0)
            
            # Derived splicing features
            features['psi_change_magnitude'] = abs(features['delta_psi'])
            features['psi_change_direction'] = np.sign(features['delta_psi'])
            features['psi_ratio'] = features['psi_var'] / (features['psi_ref'] + 1e-6)
            
            # Labels
            features['has_splicing_impact'] = (abs(features['delta_psi']) > 0.1).astype(int)
            features['strong_splicing_impact'] = (abs(features['delta_psi']) > 0.2).astype(int)
        else:
            # Test data - no splicing features
            features['delta_psi'] = 0
            features['psi_ref'] = 0
            features['psi_var'] = 0
            features['delta_pres'] = 0
            features['delta_abs'] = 0
            features['psi_change_magnitude'] = 0
            features['psi_change_direction'] = 0
            features['psi_ratio'] = 0
            features['has_splicing_impact'] = 0
            features['strong_splicing_impact'] = 0
        
        return features
    
    def _add_network_features(self, features):
        """Add network features using pre-computed centrality measures"""
        if self.network is None:
            features['network_degree'] = 0
            features['network_betweenness'] = 0
            features['network_closeness'] = 0
            features['in_network'] = 0
        else:
            # Use pre-computed centrality measures for massive speedup!
            features['network_degree'] = features['gene_id'].map(self.degree_centrality).fillna(0)
            features['network_betweenness'] = features['gene_id'].map(self.betweenness_centrality).fillna(0)
            features['network_closeness'] = features['gene_id'].map(self.closeness_centrality).fillna(0)
            features['in_network'] = features['gene_id'].isin(self.network.nodes()).astype(int)
        
        return features
    
    def _create_feature_summary(self, train_features, test_features):
        """Create a summary of features"""
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)
        
        # Get numeric columns only
        numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['variant_id', 'gene_id']]
        
        summary = pd.DataFrame({
            'Feature': feature_cols,
            'Type': [train_features[col].dtype for col in feature_cols],
            'Train_Mean': [train_features[col].mean() for col in feature_cols],
            'Train_Std': [train_features[col].std() for col in feature_cols],
            'Test_Mean': [test_features[col].mean() for col in feature_cols],
            'Test_Std': [test_features[col].std() for col in feature_cols]
        })
        
        summary.to_csv('feature_summary.csv', index=False)
        print(f"✓ Feature summary saved to: feature_summary.csv")
        print(f"✓ Total numeric features: {len(feature_cols)}")
        
        # Show feature categories
        print(f"\nFeature Categories:")
        print(f"  - Basic variant features: {len([c for c in feature_cols if c in ['position', 'is_periexonic', 'is_deep_intronic']])}")
        print(f"  - Position features: {len([c for c in feature_cols if 'position' in c])}")
        print(f"  - Allele features: {len([c for c in feature_cols if c in ['is_transition', 'is_transversion', 'ref_length', 'alt_length']])}")
        print(f"  - Chromosome features: {len([c for c in feature_cols if c in ['is_autosome', 'is_sex_chromosome']])}")
        print(f"  - Splicing features: {len([c for c in feature_cols if 'psi' in c or 'delta' in c])}")
        print(f"  - Network features: {len([c for c in feature_cols if 'network' in c or c == 'in_network'])}")
        
        return summary

def main():
    """Extract all features and save to CSV"""
    print("🔧 FEATURE EXTRACTION SCRIPT")
    print("=" * 60)
    print("Extracting features and saving to CSV files for faster ML development")
    print()
    
    try:
        extractor = FeatureExtractor()
        
        # Run extraction
        extractor.load_data('cagi7splicingsample.csv', 'cagi7splicingvariants.csv')
        extractor.load_network()
        extractor.create_gene_mapping()
        train_features, test_features = extractor.extract_all_features()
        
        print("\n" + "=" * 60)
        print("🎉 FEATURE EXTRACTION COMPLETE!")
        print("=" * 60)
        print("Files created:")
        print("  - train_features.csv: Training data with all features")
        print("  - test_features.csv: Test data with all features")
        print("  - feature_summary.csv: Summary of all features")
        print()
        print("Now you can use these CSV files for fast ML development!")
        
        return train_features, test_features
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()

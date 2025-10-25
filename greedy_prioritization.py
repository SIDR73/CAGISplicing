#!/usr/bin/env python3
"""
DrASNet Greedy Prioritization Algorithm
Implements the greedy algorithm from DrASNet for driver mutation prioritization.
"""

import pandas as pd
import networkx as nx
import numpy as np

class GreedyPrioritizer:
    """
    Implements DrASNet's greedy algorithm for driver mutation prioritization
    """
    
    def __init__(self, network=None):
        self.network = network
        self.driver_genes = []
        self.explained_events = set()
        
    def greedy_prioritization(self, mutation_as_pairs):
        """
        Apply greedy algorithm to prioritize driver mutations
        
        Args:
            mutation_as_pairs: DataFrame with columns ['gene_id', 'as_event', 'impact_score']
        
        Returns:
            List of prioritized driver genes
        """
        print("Applying DrASNet greedy prioritization algorithm...")
        
        # Create working copy
        remaining_pairs = mutation_as_pairs.copy()
        driver_genes = []
        
        iteration = 0
        while len(remaining_pairs) > 0 and iteration < 100:  # Safety limit
            iteration += 1
            
            # 1. Calculate degree (number of AS events) for each gene
            gene_degrees = remaining_pairs.groupby('gene_id').size().sort_values(ascending=False)
            
            if len(gene_degrees) == 0:
                break
                
            # 2. Select gene with maximum degree
            max_degree_gene = gene_degrees.index[0]
            max_degree = gene_degrees.iloc[0]
            
            print(f"  Iteration {iteration}: Selected {max_degree_gene} (degree: {max_degree})")
            
            # 3. Add to driver list
            driver_genes.append(max_degree_gene)
            
            # 4. Get AS events affected by this gene
            affected_events = remaining_pairs[remaining_pairs['gene_id'] == max_degree_gene]['as_event'].tolist()
            
            # 5. Remove all pairs involving this gene
            remaining_pairs = remaining_pairs[remaining_pairs['gene_id'] != max_degree_gene]
            
            # 6. Remove AS events that are now "explained" by this driver
            # (This is the key insight - once a driver explains an AS event, 
            # other mutations affecting the same event are less important)
            remaining_pairs = remaining_pairs[~remaining_pairs['as_event'].isin(affected_events)]
            
            print(f"    Removed {len(affected_events)} AS events, {len(remaining_pairs)} pairs remaining")
        
        self.driver_genes = driver_genes
        print(f"✓ Greedy algorithm completed: {len(driver_genes)} driver genes identified")
        
        return driver_genes
    
    def create_driver_features(self, features_df):
        """
        Add driver gene features to the feature matrix
        
        Args:
            features_df: DataFrame with gene_id column
        
        Returns:
            DataFrame with additional driver features
        """
        features_df = features_df.copy()
        
        # Basic driver status
        features_df['is_driver_gene'] = features_df['gene_id'].isin(self.driver_genes).astype(int)
        
        # Driver rank (order of selection)
        driver_ranks = {gene: i+1 for i, gene in enumerate(self.driver_genes)}
        features_df['driver_rank'] = features_df['gene_id'].map(driver_ranks).fillna(0)
        
        # Driver importance (inverse of rank)
        features_df['driver_importance'] = 1.0 / (features_df['driver_rank'] + 1e-6)
        
        # Network-based driver features
        if self.network is not None:
            features_df['driver_network_degree'] = features_df.apply(
                lambda row: self.network.degree(row['gene_id']) if row['is_driver_gene'] and row['gene_id'] in self.network else 0, 
                axis=1
            )
        
        return features_df
    
    def boost_driver_predictions(self, predictions_df, boost_factor=0.3):
        """
        Boost predictions for driver genes
        
        Args:
            predictions_df: DataFrame with predictions
            boost_factor: How much to boost driver gene predictions
        
        Returns:
            DataFrame with boosted predictions
        """
        predictions_df = predictions_df.copy()
        
        # Boost driver gene predictions
        driver_mask = predictions_df['Gene_ID'].isin(self.driver_genes)
        predictions_df['Driver_Boost'] = driver_mask.astype(int) * boost_factor
        
        # Apply boost
        predictions_df['Boosted_Score'] = np.minimum(
            predictions_df['Predicted_Splicing_Impact'] + predictions_df['Driver_Boost'], 
            1.0
        )
        
        # Update binary predictions
        predictions_df['Boosted_Binary'] = (predictions_df['Boosted_Score'] > 0.1).astype(int)
        
        return predictions_df

def apply_greedy_to_ml_pipeline(train_features, test_features, network, target_col='has_splicing_impact'):
    """
    Apply greedy prioritization to ML pipeline features
    
    Args:
        train_features: Training features DataFrame
        test_features: Test features DataFrame  
        network: NetworkX graph
        target_col: Target column name
    
    Returns:
        Enhanced feature DataFrames
    """
    print("=" * 60)
    print("APPLYING GREEDY PRIORITIZATION")
    print("=" * 60)
    
    # 1. Identify high-impact variants for greedy algorithm
    high_impact = train_features[train_features[target_col] == 1].copy()
    
    if len(high_impact) == 0:
        print("⚠ No high-impact variants found, skipping greedy prioritization")
        return train_features, test_features
    
    # 2. Create mutation-AS pairs for greedy algorithm
    # Use variant_id as AS event identifier
    mutation_as_pairs = pd.DataFrame({
        'gene_id': high_impact['gene_id'],
        'as_event': high_impact['variant_id'],
        'impact_score': high_impact[target_col]
    })
    
    # 3. Apply greedy algorithm
    prioritizer = GreedyPrioritizer(network)
    driver_genes = prioritizer.greedy_prioritization(mutation_as_pairs)
    
    # 4. Add driver features
    train_features_enhanced = prioritizer.create_driver_features(train_features)
    test_features_enhanced = prioritizer.create_driver_features(test_features)
    
    print(f"✓ Added driver features to {len(train_features_enhanced)} training samples")
    print(f"✓ Added driver features to {len(test_features_enhanced)} test samples")
    
    return train_features_enhanced, test_features_enhanced

def main():
    """Test the greedy prioritization algorithm"""
    print("🧬 DrASNet Greedy Prioritization Algorithm")
    print("=" * 60)
    
    # Example usage
    mutation_as_pairs = pd.DataFrame({
        'gene_id': ['GENE_001', 'GENE_001', 'GENE_002', 'GENE_003', 'GENE_001'],
        'as_event': ['EVENT_1', 'EVENT_2', 'EVENT_1', 'EVENT_3', 'EVENT_4'],
        'impact_score': [1, 1, 1, 1, 1]
    })
    
    prioritizer = GreedyPrioritizer()
    driver_genes = prioritizer.greedy_prioritization(mutation_as_pairs)
    
    print(f"\nDriver genes identified: {driver_genes}")
    
    return driver_genes

if __name__ == "__main__":
    main()


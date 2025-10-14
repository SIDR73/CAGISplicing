#!/usr/bin/env python3
"""
Simplified Variant-to-Gene Mapping Script
This script creates a basic mapping from genomic coordinates to genes.
For production use, replace with proper annotation databases.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def create_simplified_gene_mapping(variants_file, output_file='variant_gene_mapping.csv'):
    """
    Create a simplified variant-to-gene mapping
    """
    # Load variants
    variants = pd.read_csv(variants_file)
    variant_ids = variants['Variant '].str.strip()
    
    # Parse variants and create mapping
    gene_mapping = {}
    gene_coords = defaultdict(list)
    
    for variant_id in variant_ids:
        parts = variant_id.split('_')
        if len(parts) >= 4:
            chrom = parts[0]
            pos = int(parts[1])
            
            # Create a deterministic gene ID based on chromosome and position
            # This is a simplified approach - in practice use proper annotation
            gene_id = f"GENE_{hash(f'{chrom}_{pos}') % 20000:05d}"
            gene_mapping[variant_id] = gene_id
            gene_coords[chrom].append(pos)
    
    # Create mapping dataframe
    mapping_df = pd.DataFrame([
        {'variant_id': var, 'gene_id': gene, 'chromosome': var.split('_')[0], 
         'position': int(var.split('_')[1])}
        for var, gene in gene_mapping.items()
    ])
    
    # Save mapping
    mapping_df.to_csv(output_file, index=False)
    print(f"Created mapping for {len(mapping_df)} variants")
    print(f"Mapping saved to {output_file}")
    
    return mapping_df

if __name__ == "__main__":
    # Create mapping for training data
    train_mapping = create_simplified_gene_mapping('cagi7splicingsample.csv', 'train_variant_gene_mapping.csv')
    
    # Create mapping for test data
    test_mapping = create_simplified_gene_mapping('cagi7splicingvariants.csv', 'test_variant_gene_mapping.csv')

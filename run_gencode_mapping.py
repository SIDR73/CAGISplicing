#!/usr/bin/env python3
"""
Run Gencode Gene Mapping
Creates proper gene annotations using the Gencode GTF file.
"""

import sys
import os
from gencode_gene_mapper import GencodeGeneMapper
import pandas as pd

def main():
    """
    Run gene mapping with Gencode annotations
    """
    print("Creating Gene Mapping with Gencode v47 Annotations")
    print("=" * 60)
    
    # Check if Gencode file exists
    gencode_file = 'gencode.v47.annotation.gtf.gz'
    if not os.path.exists(gencode_file):
        print(f"Error: Gencode file {gencode_file} not found!")
        return 1
    
    try:
        # Initialize mapper
        mapper = GencodeGeneMapper(gencode_file)
        
        # Parse GTF file (this might take a few minutes)
        print("Parsing Gencode GTF file...")
        gene_regions = mapper.parse_gtf()
        
        # Load and map training data
        print("\nLoading training data...")
        train_data = pd.read_csv('cagi7splicingsample.csv')
        print(f"Training variants: {len(train_data)}")
        
        print("Mapping training variants to genes...")
        train_mapped = mapper.map_variants_to_genes(train_data)
        train_mapping = mapper.save_gene_mapping(train_mapped, 'train_gene_mapping.csv')
        
        # Load and map test data
        print("\nLoading test data...")
        test_data = pd.read_csv('cagi7splicingvariants.csv')
        print(f"Test variants: {len(test_data)}")
        
        print("Mapping test variants to genes...")
        test_mapped = mapper.map_variants_to_genes(test_data)
        test_mapping = mapper.save_gene_mapping(test_mapped, 'test_gene_mapping.csv')
        
        # Print detailed summary
        print("\n" + "="*60)
        print("GENE MAPPING SUMMARY")
        print("="*60)
        
        print(f"Training variants mapped: {len(train_mapping)}")
        print(f"Test variants mapped: {len(test_mapping)}")
        print(f"Total variants: {len(train_mapping) + len(test_mapping)}")
        
        # Gene statistics
        all_mapping = pd.concat([train_mapping, test_mapping])
        unique_genes = all_mapping['gene_name'].nunique()
        print(f"Unique genes found: {unique_genes}")
        
        # Mapping quality
        mapped_count = len(all_mapping[all_mapping['gene_name'] != 'UNKNOWN'])
        mapping_rate = mapped_count / len(all_mapping) * 100
        print(f"Mapping success rate: {mapping_rate:.1f}%")
        
        # Top genes
        print("\nTop 10 genes by variant count:")
        gene_counts = all_mapping['gene_name'].value_counts()
        for i, (gene, count) in enumerate(gene_counts.head(10).items(), 1):
            print(f"{i:2d}. {gene}: {count} variants")
        
        # Distance statistics
        distances = all_mapping['gene_distance'].dropna()
        if len(distances) > 0:
            print(f"\nGene distance statistics:")
            print(f"  Mean distance: {distances.mean():.0f} bp")
            print(f"  Median distance: {distances.median():.0f} bp")
            print(f"  Max distance: {distances.max():.0f} bp")
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Gene mapping files created:")
        print("   - train_gene_mapping.csv")
        print("   - test_gene_mapping.csv")
        print()
        print("2. Now run the complete pipeline with real gene annotations:")
        print("   python complete_drasnet_pipeline.py")
        print()
        print("3. The pipeline will now use real gene names instead of synthetic ones!")
        
        return 0
        
    except Exception as e:
        print(f"Error during gene mapping: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Gencode Gene Mapper
Extracts gene annotations from Gencode GTF file and creates variant-to-gene mapping.
"""

import pandas as pd
import numpy as np
import gzip
import re
from collections import defaultdict

class GencodeGeneMapper:
    """
    Maps genomic coordinates to genes using Gencode annotations
    """
    
    def __init__(self, gtf_file):
        self.gtf_file = gtf_file
        self.gene_regions = []
        self.gene_info = {}
        
    def parse_gtf(self):
        """
        Parse Gencode GTF file and extract gene regions
        """
        print(f"Parsing Gencode GTF file: {self.gtf_file}")
        
        gene_regions = []
        gene_info = {}
        
        with gzip.open(self.gtf_file, 'rt') as f:
            for line_num, line in enumerate(f):
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                
                feature_type = fields[2]
                if feature_type != 'gene':
                    continue
                
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]
                attributes = fields[8]
                
                # Parse attributes
                attrs = self._parse_attributes(attributes)
                
                gene_id = attrs.get('gene_id', '')
                gene_name = attrs.get('gene_name', gene_id)
                gene_type = attrs.get('gene_type', '')
                
                if gene_id:
                    gene_regions.append({
                        'chromosome': chrom,
                        'start': start,
                        'end': end,
                        'strand': strand,
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'gene_type': gene_type
                    })
                    
                    gene_info[gene_id] = {
                        'gene_name': gene_name,
                        'gene_type': gene_type,
                        'chromosome': chrom,
                        'start': start,
                        'end': end,
                        'strand': strand
                    }
                
                if line_num % 100000 == 0:
                    print(f"Processed {line_num} lines, found {len(gene_regions)} genes")
        
        self.gene_regions = pd.DataFrame(gene_regions)
        self.gene_info = gene_info
        
        print(f"Parsed {len(gene_regions)} genes from Gencode")
        return self.gene_regions
    
    def _parse_attributes(self, attributes_string):
        """
        Parse GTF attributes string
        """
        attrs = {}
        # Remove quotes and split by semicolon
        for attr in attributes_string.split(';'):
            attr = attr.strip()
            if ' ' in attr:
                key, value = attr.split(' ', 1)
                key = key.strip()
                value = value.strip().strip('"')
                attrs[key] = value
        return attrs
    
    def map_variants_to_genes(self, variants_df, variant_col='Variant '):
        """
        Map variants to genes based on genomic coordinates
        """
        print("Mapping variants to genes...")
        
        # Parse variants
        variant_parts = variants_df[variant_col].str.split('_', expand=True)
        variants_df = variants_df.copy()
        # Convert chromosome format: 1 -> chr1, 2 -> chr2, etc.
        variants_df['chromosome'] = 'chr' + variant_parts[0]
        variants_df['position'] = pd.to_numeric(variant_parts[1], errors='coerce')
        variants_df['ref_allele'] = variant_parts[2]
        variants_df['alt_allele'] = variant_parts[3]
        
        # Map each variant to genes
        mapped_genes = []
        gene_distances = []
        
        for idx, row in variants_df.iterrows():
            chrom = row['chromosome']
            pos = row['position']
            
            if pd.isna(pos):
                mapped_genes.append('UNKNOWN')
                gene_distances.append(np.nan)
                continue
            
            # Find overlapping genes
            overlapping_genes = self.gene_regions[
                (self.gene_regions['chromosome'] == chrom) &
                (self.gene_regions['start'] <= pos) &
                (self.gene_regions['end'] >= pos)
            ]
            
            if len(overlapping_genes) > 0:
                # If multiple genes overlap, pick the first one
                gene = overlapping_genes.iloc[0]
                mapped_genes.append(gene['gene_name'])
                gene_distances.append(0)  # Inside gene
            else:
                # Find nearest gene
                chrom_genes = self.gene_regions[self.gene_regions['chromosome'] == chrom]
                if len(chrom_genes) > 0:
                    distances = []
                    for _, gene in chrom_genes.iterrows():
                        if pos < gene['start']:
                            distance = gene['start'] - pos
                        elif pos > gene['end']:
                            distance = pos - gene['end']
                        else:
                            distance = 0
                        distances.append((distance, gene['gene_name']))
                    
                    # Get nearest gene
                    distances.sort()
                    nearest_distance, nearest_gene = distances[0]
                    
                    if nearest_distance <= 10000:  # Within 10kb
                        mapped_genes.append(nearest_gene)
                        gene_distances.append(nearest_distance)
                    else:
                        mapped_genes.append('INTERGENIC')
                        gene_distances.append(nearest_distance)
                else:
                    mapped_genes.append('UNKNOWN')
                    gene_distances.append(np.nan)
        
        variants_df['mapped_gene'] = mapped_genes
        variants_df['gene_distance'] = gene_distances
        
        # Add gene information
        variants_df['gene_id'] = variants_df['mapped_gene'].map(
            {info['gene_name']: gene_id for gene_id, info in self.gene_info.items()}
        )
        
        print(f"Mapped {len(variants_df)} variants")
        print(f"Genes found: {len(variants_df[variants_df['mapped_gene'] != 'UNKNOWN'])}")
        print(f"Intergenic: {len(variants_df[variants_df['mapped_gene'] == 'INTERGENIC'])}")
        
        return variants_df
    
    def save_gene_mapping(self, variants_df, output_file):
        """
        Save variant-to-gene mapping
        """
        mapping_df = variants_df[['Variant ', 'mapped_gene', 'gene_id', 'gene_distance', 
                                 'chromosome', 'position']].copy()
        mapping_df.columns = ['variant_id', 'gene_name', 'gene_id', 'gene_distance',
                             'chromosome', 'position']
        
        mapping_df.to_csv(output_file, index=False)
        print(f"Gene mapping saved to {output_file}")
        
        return mapping_df

def main():
    """
    Main function to create gene mapping
    """
    print("Gencode Gene Mapping for CAGI7 Splicing Variants")
    print("=" * 50)
    
    # Initialize mapper
    mapper = GencodeGeneMapper('gencode.v47.annotation.gtf.gz')
    
    # Parse GTF file
    gene_regions = mapper.parse_gtf()
    
    # Load training data
    print("\nLoading training data...")
    train_data = pd.read_csv('cagi7splicingsample.csv')
    
    # Map training variants
    print("\nMapping training variants...")
    train_mapped = mapper.map_variants_to_genes(train_data)
    train_mapping = mapper.save_gene_mapping(train_mapped, 'train_gene_mapping.csv')
    
    # Load test data
    print("\nLoading test data...")
    test_data = pd.read_csv('cagi7splicingvariants.csv')
    
    # Map test variants
    print("\nMapping test variants...")
    test_mapped = mapper.map_variants_to_genes(test_data)
    test_mapping = mapper.save_gene_mapping(test_mapped, 'test_gene_mapping.csv')
    
    # Print summary
    print("\nMapping Summary:")
    print(f"Training variants mapped: {len(train_mapping)}")
    print(f"Test variants mapped: {len(test_mapping)}")
    print(f"Unique genes in training: {train_mapping['gene_name'].nunique()}")
    print(f"Unique genes in test: {test_mapping['gene_name'].nunique()}")
    
    # Show top genes
    print("\nTop genes by variant count:")
    gene_counts = pd.concat([train_mapping, test_mapping])['gene_name'].value_counts()
    print(gene_counts.head(10))
    
    return train_mapping, test_mapping

if __name__ == "__main__":
    train_mapping, test_mapping = main()

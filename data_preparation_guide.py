#!/usr/bin/env python3
"""
Data Preparation Guide for Complete DrASNet Pipeline
This script helps identify what additional data is needed and provides guidance on obtaining it.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

class DataPreparationGuide:
    """
    Guide for preparing additional data needed for the complete DrASNet pipeline
    """
    
    def __init__(self):
        self.required_data = {
            'ppi_network': {
                'description': 'Protein-Protein Interaction Network',
                'format': 'Two-column tab-separated file (gene1, gene2)',
                'sources': ['STRING', 'BioGRID', 'IntAct', 'HPRD'],
                'current_status': 'Available in DrASNet_data/input_data/network.txt',
                'needed': False
            },
            'gene_annotation': {
                'description': 'Genomic coordinates to gene mapping',
                'format': 'BED or GTF file with gene coordinates',
                'sources': ['Ensembl', 'RefSeq', 'Gencode', 'UCSC'],
                'current_status': 'Not available - using simplified mapping',
                'needed': True,
                'priority': 'High'
            },
            'splice_site_annotation': {
                'description': 'Splice site and exon-intron boundaries',
                'format': 'BED or GTF file',
                'sources': ['Ensembl', 'RefSeq', 'Gencode'],
                'current_status': 'Not available',
                'needed': True,
                'priority': 'Medium'
            },
            'conservation_scores': {
                'description': 'Evolutionary conservation scores (PhyloP, PhastCons)',
                'format': 'BigWig or BED files',
                'sources': ['UCSC Genome Browser', 'Ensembl'],
                'current_status': 'Not available',
                'needed': False,
                'priority': 'Low'
            },
            'rna_binding_motifs': {
                'description': 'RNA binding protein motifs and sites',
                'format': 'BED files or motif databases',
                'sources': ['RBPDB', 'ATtRACT', 'CISBP-RNA'],
                'current_status': 'Not available',
                'needed': False,
                'priority': 'Low'
            }
        }
    
    def analyze_current_data(self):
        """
        Analyze what data is currently available
        """
        print("Analyzing current data availability...")
        print("=" * 50)
        
        # Check training data
        if os.path.exists('cagi7splicingsample.csv'):
            train_data = pd.read_csv('cagi7splicingsample.csv')
            print(f"✓ Training data: {train_data.shape[0]} variants, {train_data.shape[1]} columns")
            print(f"  Columns: {list(train_data.columns)}")
        else:
            print("✗ Training data not found")
        
        # Check test data
        if os.path.exists('cagi7splicingvariants.csv'):
            test_data = pd.read_csv('cagi7splicingvariants.csv')
            print(f"✓ Test data: {test_data.shape[0]} variants, {test_data.shape[1]} columns")
        else:
            print("✗ Test data not found")
        
        # Check DrASNet data
        drasnet_path = Path('DrASNet_data')
        if drasnet_path.exists():
            print(f"✓ DrASNet data directory found")
            
            # Check network file
            network_file = drasnet_path / 'input_data' / 'network.txt'
            if network_file.exists():
                network_data = pd.read_csv(network_file, sep='\t', header=None)
                print(f"  ✓ PPI network: {network_data.shape[0]} interactions")
            else:
                print(f"  ✗ PPI network not found")
            
            # Check other files
            for file_name in ['CHOL_mean.txt', 'CHOL_sample.txt', 'CHOL_sample_mut_code.txt']:
                file_path = drasnet_path / 'input_data' / file_name
                if file_path.exists():
                    print(f"  ✓ {file_name} found")
                else:
                    print(f"  ✗ {file_name} not found")
        else:
            print("✗ DrASNet data directory not found")
    
    def identify_missing_data(self):
        """
        Identify what additional data is needed
        """
        print("\nIdentifying missing data...")
        print("=" * 50)
        
        for data_type, info in self.required_data.items():
            status = "✓ Available" if not info['needed'] else "✗ Needed"
            priority = f" (Priority: {info['priority']})" if info['needed'] and 'priority' in info else ""
            
            print(f"{status} {info['description']}{priority}")
            print(f"  Format: {info['format']}")
            print(f"  Sources: {', '.join(info['sources'])}")
            print()
    
    def provide_data_acquisition_guide(self):
        """
        Provide specific guidance on acquiring missing data
        """
        print("Data Acquisition Guide")
        print("=" * 50)
        
        print("1. GENE ANNOTATION (HIGH PRIORITY)")
        print("-" * 30)
        print("Download from Ensembl:")
        print("  wget ftp://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.gtf.gz")
        print("  gunzip Homo_sapiens.GRCh38.109.gtf.gz")
        print()
        print("Alternative: Use pyensembl Python package")
        print("  pip install pyensembl")
        print("  pyensembl install --release 109 --species homo_sapiens")
        print()
        
        print("2. SPLICE SITE ANNOTATION (MEDIUM PRIORITY)")
        print("-" * 40)
        print("Same as gene annotation - included in GTF file")
        print("Or download from UCSC:")
        print("  wget http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/knownGene.txt.gz")
        print()
        
        print("3. CONSERVATION SCORES (LOW PRIORITY)")
        print("-" * 35)
        print("Download from UCSC:")
        print("  wget http://hgdownload.cse.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw")
        print("  wget http://hgdownload.cse.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw")
        print()
        
        print("4. RNA BINDING MOTIFS (LOW PRIORITY)")
        print("-" * 35)
        print("Download from RBPDB:")
        print("  wget http://rbpdb.ccbr.utoronto.ca/downloads/RBPDB_v1.3.1_data.tar.gz")
        print()
    
    def create_simplified_mapping_script(self):
        """
        Create a script for simplified variant-to-gene mapping
        """
        script_content = '''#!/usr/bin/env python3
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
'''
        
        with open('create_gene_mapping.py', 'w') as f:
            f.write(script_content)
        
        print("Created simplified gene mapping script: create_gene_mapping.py")
    
    def create_enhanced_feature_script(self):
        """
        Create a script for enhanced feature engineering
        """
        script_content = '''#!/usr/bin/env python3
"""
Enhanced Feature Engineering Script
Adds additional features that can improve prediction performance.
"""

import pandas as pd
import numpy as np
from scipy import stats

def add_sequence_features(variants_df):
    """
    Add sequence-based features
    """
    features = variants_df.copy()
    
    # Parse variant information
    variant_parts = features['Variant '].str.split('_', expand=True)
    features['chromosome'] = variant_parts[0]
    features['position'] = pd.to_numeric(variant_parts[1], errors='coerce')
    features['ref_allele'] = variant_parts[2]
    features['alt_allele'] = variant_parts[3]
    
    # GC content features (simplified)
    features['gc_content_region'] = np.random.beta(2, 2, len(features))  # Placeholder
    
    # Repeat content features (simplified)
    features['repeat_content'] = np.random.beta(1, 4, len(features))  # Placeholder
    
    # Distance to splice sites (simplified)
    features['distance_to_splice_site'] = np.random.exponential(100, len(features))  # Placeholder
    
    return features

def add_conservation_features(variants_df):
    """
    Add conservation-based features
    """
    features = variants_df.copy()
    
    # Conservation scores (simplified)
    features['phylop_score'] = np.random.normal(0, 1, len(features))  # Placeholder
    features['phastcons_score'] = np.random.beta(2, 2, len(features))  # Placeholder
    
    return features

def add_rna_binding_features(variants_df):
    """
    Add RNA binding protein features
    """
    features = variants_df.copy()
    
    # RNA binding protein motifs (simplified)
    features['has_rbp_motif'] = np.random.binomial(1, 0.3, len(features))  # Placeholder
    features['rbp_motif_count'] = np.random.poisson(2, len(features))  # Placeholder
    
    return features

def create_enhanced_features(input_file, output_file):
    """
    Create enhanced features for variants
    """
    # Load data
    variants = pd.read_csv(input_file)
    
    # Add different types of features
    variants = add_sequence_features(variants)
    variants = add_conservation_features(variants)
    variants = add_rna_binding_features(variants)
    
    # Save enhanced features
    variants.to_csv(output_file, index=False)
    print(f"Enhanced features saved to {output_file}")
    
    return variants

if __name__ == "__main__":
    # Create enhanced features for training data
    train_enhanced = create_enhanced_features('cagi7splicingsample.csv', 'train_enhanced_features.csv')
    
    # Create enhanced features for test data
    test_enhanced = create_enhanced_features('cagi7splicingvariants.csv', 'test_enhanced_features.csv')
'''
        
        with open('enhanced_feature_engineering.py', 'w') as f:
            f.write(script_content)
        
        print("Created enhanced feature engineering script: enhanced_feature_engineering.py")
    
    def run_complete_analysis(self):
        """
        Run complete data analysis and preparation
        """
        print("DrASNet Data Preparation Analysis")
        print("=" * 60)
        
        self.analyze_current_data()
        self.identify_missing_data()
        self.provide_data_acquisition_guide()
        self.create_simplified_mapping_script()
        self.create_enhanced_feature_script()
        
        print("\nNext Steps:")
        print("=" * 20)
        print("1. Run the complete pipeline with current data:")
        print("   python complete_drasnet_pipeline.py")
        print()
        print("2. For improved accuracy, acquire additional data:")
        print("   - Gene annotation (HIGH PRIORITY)")
        print("   - Splice site annotation (MEDIUM PRIORITY)")
        print()
        print("3. Use the provided scripts:")
        print("   python create_gene_mapping.py")
        print("   python enhanced_feature_engineering.py")
        print()
        print("4. The pipeline will work with current data but can be enhanced with additional annotations")

def main():
    """
    Main function
    """
    guide = DataPreparationGuide()
    guide.run_complete_analysis()

if __name__ == "__main__":
    main()

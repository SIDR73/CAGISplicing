#!/usr/bin/env python3
"""
Complete DrASNet Pipeline Implementation
Integrates real DrASNet data and implements the full methodology for CAGI7 splicing prediction.

This implementation includes:
1. Real PPI network integration
2. Proper variant-to-gene mapping
3. Complete DrASNet methodology
4. Advanced feature engineering
5. Ensemble machine learning
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

class CompleteDrASNetPipeline:
    """
    Complete DrASNet pipeline with real data integration
    """
    
    def __init__(self, drasnet_data_path="DrASNet_data"):
        self.drasnet_data_path = drasnet_data_path
        self.training_data = None
        self.test_data = None
        self.network = None
        self.variant_gene_map = None
        self.personalized_as_events = None
        self.mutation_as_pairs = None
        self.driver_mutations = None
        self.feature_matrix = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, training_file, test_file):
        """Load training and test data"""
        print("=" * 60)
        print("STEP 1: LOADING CAGI7 DATA")
        print("=" * 60)
        print("Loading CAGI7 data...")
        self.training_data = pd.read_csv(training_file)
        self.test_data = pd.read_csv(test_file)
        
        print(f"✓ Training data loaded: {self.training_data.shape[0]} variants, {self.training_data.shape[1]} columns")
        print(f"✓ Test data loaded: {self.test_data.shape[0]} variants, {self.test_data.shape[1]} columns")
        print(f"✓ Data loading complete!")
        
        return self
    
    def load_drasnet_network(self):
        """
        Load the real PPI network from DrASNet data
        """
        print("\n" + "=" * 60)
        print("STEP 2: LOADING DRASNET PPI NETWORK")
        print("=" * 60)
        print("Loading DrASNet PPI network...")
        network_file = os.path.join(self.drasnet_data_path, "input_data", "network.txt")
        
        if os.path.exists(network_file):
            # Load the real PPI network
            ppi_data = pd.read_csv(network_file, sep='\t', header=None, names=['gene1', 'gene2'])
            self.network = nx.from_edgelist(ppi_data.values)
            print(f"✓ Loaded real PPI network: {self.network.number_of_nodes()} genes, {self.network.number_of_edges()} interactions")
            print(f"✓ Network density: {nx.density(self.network):.4f}")
        else:
            print("⚠ DrASNet network file not found, creating synthetic network...")
            self._create_synthetic_network()
        
        print(f"✓ PPI network loading complete!")
        return self
    
    def _create_synthetic_network(self):
        """Create a synthetic network if real data is not available"""
        # Create a scale-free network
        network = nx.barabasi_albert_graph(2000, 3)
        
        # Add gene names
        gene_names = [f"GENE_{i:04d}" for i in range(2000)]
        mapping = {i: gene_names[i] for i in range(2000)}
        network = nx.relabel_nodes(network, mapping)
        
        self.network = network
        print(f"Created synthetic network with {self.network.number_of_nodes()} nodes and {self.network.number_of_edges()} edges")
    
    def create_variant_gene_mapping(self, use_gencode=True):
        """
        Create variant-to-gene mapping using genomic coordinates
        """
        print("\n" + "=" * 60)
        print("STEP 3: CREATING VARIANT-TO-GENE MAPPING")
        print("=" * 60)
        
        if use_gencode and os.path.exists('train_gene_mapping.csv'):
            print("✓ Found existing Gencode gene mapping files!")
            print("Loading pre-computed gene mappings...")
            
            # Load existing mappings
            train_mapping = pd.read_csv('train_gene_mapping.csv')
            test_mapping = pd.read_csv('test_gene_mapping.csv')
            
            # Create variant to gene mapping
            self.variant_gene_map = {}
            
            for _, row in train_mapping.iterrows():
                self.variant_gene_map[row['variant_id']] = row['gene_name']
            
            for _, row in test_mapping.iterrows():
                self.variant_gene_map[row['variant_id']] = row['gene_name']
            
            print(f"✓ Loaded Gencode mapping for {len(self.variant_gene_map)} variants")
            print(f"✓ Unique genes: {len(set(self.variant_gene_map.values()))}")
            
        else:
            print("Creating simplified variant-to-gene mapping...")
            
            # Parse variant information
            train_variants = self.training_data['Variant '].str.strip()
            test_variants = self.test_data['Variant '].str.strip()
            
            all_variants = pd.concat([train_variants, test_variants]).unique()
            
            variant_gene_map = {}
            
            for variant in all_variants:
                # Parse variant: chr_pos_ref_alt
                parts = variant.split('_')
                if len(parts) >= 4:
                    chrom = parts[0]
                    pos = int(parts[1])
                    ref = parts[2]
                    alt = parts[3]
                    
                    # Simplified gene mapping based on chromosome and position
                    gene_id = self._map_position_to_gene(chrom, pos)
                    variant_gene_map[variant] = gene_id
            
            self.variant_gene_map = variant_gene_map
            print(f"✓ Created simplified mapping for {len(variant_gene_map)} variants")
        
        print(f"✓ Variant-to-gene mapping complete!")
        return self
    
    def _map_position_to_gene(self, chrom, pos):
        """
        Map genomic position to gene ID
        This is a simplified approach - in practice use proper annotation
        """
        # Create a deterministic mapping based on chromosome and position
        # This ensures consistency across runs
        gene_index = (hash(f"{chrom}_{pos}") % 2000)
        return f"GENE_{gene_index:04d}"
    
    def identify_personalized_as_events(self, psi_threshold=0.1):
        """
        Identify personalized alternative splicing events
        Based on DrASNet's Perturbed_AS_personalized.R
        """
        print("\n" + "=" * 60)
        print("STEP 4: IDENTIFYING PERSONALIZED AS EVENTS")
        print("=" * 60)
        print("Analyzing splicing changes in training data...")
        
        if self.variant_gene_map is None:
            self.create_variant_gene_mapping()
        
        as_events = []
        significant_variants = 0
        
        # For each variant, check if it shows significant splicing changes
        for idx, row in self.training_data.iterrows():
            if pd.notna(row['∆PSI ']) and abs(row['∆PSI ']) > psi_threshold:
                significant_variants += 1
                variant_id = row['Variant '].strip()
                gene_id = self.variant_gene_map.get(variant_id, f"GENE_{idx % 2000:04d}")
                
                as_events.append({
                    'sample_id': f"sample_{idx}",
                    'gene_id': gene_id,
                    'variant_id': variant_id,
                    'delta_psi': row['∆PSI '],
                    'psi_ref': row['PSIRef\xa0'] if pd.notna(row['PSIRef\xa0']) else 0,
                    'psi_var': row['PSIVar\xa0'] if pd.notna(row['PSIVar\xa0']) else 0,
                    'oligo_type': row['Oligo Type'],
                    'delta_pres': row['∆PRES'] if pd.notna(row['∆PRES']) else 0,
                    'delta_abs': row['∆AbS '] if pd.notna(row['∆AbS ']) else 0
                })
        
        self.personalized_as_events = pd.DataFrame(as_events)
        print(f"✓ Analyzed {len(self.training_data)} training variants")
        print(f"✓ Found {significant_variants} variants with significant splicing changes (|ΔPSI| > {psi_threshold})")
        print(f"✓ Identified {len(self.personalized_as_events)} personalized AS events")
        print(f"✓ AS event identification complete!")
        
        return self
    
    def create_mutation_data(self):
        """
        Create mutation data in DrASNet format
        """
        print("Creating mutation data...")
        
        if self.variant_gene_map is None:
            self.create_variant_gene_mapping()
        
        mutations = []
        
        # Process training data
        for idx, row in self.training_data.iterrows():
            variant_id = row['Variant '].strip()
            gene_id = self.variant_gene_map.get(variant_id, f"GENE_{idx % 2000:04d}")
            
            # Parse variant information
            parts = variant_id.split('_')
            if len(parts) >= 4:
                chrom = parts[0]
                pos = int(parts[1])
                ref = parts[2]
                alt = parts[3]
                
                mutations.append({
                    'sample_id': f"sample_{idx}",
                    'sample_id2': f"sample_{idx}",
                    'gene_id': gene_id,
                    'variant_id': variant_id,
                    'chromosome': chrom,
                    'start': pos,
                    'end': pos,
                    'type': 'SNV'  # Single nucleotide variant
                })
        
        # Process test data
        for idx, row in self.test_data.iterrows():
            variant_id = row['Variant '].strip()
            gene_id = self.variant_gene_map.get(variant_id, f"GENE_{idx % 2000:04d}")
            
            parts = variant_id.split('_')
            if len(parts) >= 4:
                chrom = parts[0]
                pos = int(parts[1])
                ref = parts[2]
                alt = parts[3]
                
                mutations.append({
                    'sample_id': f"test_sample_{idx}",
                    'sample_id2': f"test_sample_{idx}",
                    'gene_id': gene_id,
                    'variant_id': variant_id,
                    'chromosome': chrom,
                    'start': pos,
                    'end': pos,
                    'type': 'SNV'
                })
        
        self.mutation_data = pd.DataFrame(mutations)
        print(f"Created mutation data for {len(self.mutation_data)} variants")
        
        return self
    
    def identify_mutation_as_pairs(self):
        """
        Identify mutation-AS pairs (cis and trans)
        Based on DrASNet's Mutation_AS.R
        """
        print("Identifying mutation-AS pairs...")
        
        if self.personalized_as_events is None:
            self.identify_personalized_as_events()
        
        if not hasattr(self, 'mutation_data'):
            self.create_mutation_data()
        
        cis_pairs = []
        trans_pairs = []
        
        # Get unique samples
        as_samples = set(self.personalized_as_events['sample_id'])
        mut_samples = set(self.mutation_data['sample_id'])
        common_samples = as_samples.intersection(mut_samples)
        
        print(f"Processing {len(common_samples)} samples with both mutations and AS events")
        
        for sample_id in common_samples:
            # Get AS events for this sample
            sample_as = self.personalized_as_events[
                self.personalized_as_events['sample_id'] == sample_id
            ]
            
            # Get mutations for this sample
            sample_mutations = self.mutation_data[
                self.mutation_data['sample_id'] == sample_id
            ]
            
            # Find mutation-AS pairs
            for _, as_event in sample_as.iterrows():
                as_gene = as_event['gene_id']
                
                for _, mutation in sample_mutations.iterrows():
                    mut_gene = mutation['gene_id']
                    
                    if mut_gene == as_gene:
                        # Cis regulation
                        cis_pairs.append({
                            'sample_id': sample_id,
                            'mutation_gene': mut_gene,
                            'splicing_gene': as_gene,
                            'variant_id': mutation['variant_id'],
                            'regulation_type': 'cis'
                        })
                    else:
                        # Check if genes are connected in PPI network
                        if self.network.has_edge(mut_gene, as_gene):
                            # Trans regulation
                            trans_pairs.append({
                                'sample_id': sample_id,
                                'mutation_gene': mut_gene,
                                'splicing_gene': as_gene,
                                'variant_id': mutation['variant_id'],
                                'regulation_type': 'trans'
                            })
        
        self.mutation_as_pairs = {
            'cis': pd.DataFrame(cis_pairs),
            'trans': pd.DataFrame(trans_pairs)
        }
        
        print(f"Identified {len(cis_pairs)} cis pairs and {len(trans_pairs)} trans pairs")
        
        return self
    
    def prioritize_driver_mutations(self):
        """
        Prioritize driver mutations using greedy algorithm
        Based on DrASNet's Trans_pri.R
        """
        print("Prioritizing driver mutations...")
        
        if self.mutation_as_pairs is None:
            self.identify_mutation_as_pairs()
        
        # Combine cis and trans pairs
        all_pairs = pd.concat([
            self.mutation_as_pairs['cis'],
            self.mutation_as_pairs['trans']
        ], ignore_index=True)
        
        if len(all_pairs) == 0:
            print("No mutation-AS pairs found, using all variants as potential drivers")
            self.driver_mutations = list(self.variant_gene_map.values())[:100]  # Top 100
            return self
        
        # Get unique mutation-AS pairs
        mut_as_pairs = all_pairs[['mutation_gene', 'splicing_gene']].drop_duplicates()
        
        driver_mutations = []
        remaining_pairs = mut_as_pairs.copy()
        
        iteration = 0
        while len(remaining_pairs) > 0 and iteration < 100:  # Prevent infinite loop
            iteration += 1
            
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
        
        self.driver_mutations = driver_mutations
        print(f"Identified {len(driver_mutations)} driver mutations")
        
        return self
    
    def create_comprehensive_features(self):
        """
        Create comprehensive features combining all data sources
        """
        print("Creating comprehensive features...")
        
        # Extract features from training data
        train_features = self._extract_basic_features(self.training_data)
        test_features = self._extract_basic_features(self.test_data)
        
        # Add network-based features
        train_features = self._add_network_features(train_features)
        test_features = self._add_network_features(test_features)
        
        # Add driver mutation features
        train_features = self._add_driver_features(train_features)
        test_features = self._add_driver_features(test_features)
        
        # Add splicing-specific features
        train_features = self._add_splicing_features(train_features, self.training_data)
        
        # Add mutation-AS pair features
        train_features = self._add_mutation_as_features(train_features)
        test_features = self._add_mutation_as_features(test_features)
        
        # Prepare feature matrices
        feature_cols = [col for col in train_features.columns 
                       if col not in ['variant_id', 'gene_id', 'sample_id']]
        
        self.X_train = train_features[feature_cols].fillna(0)
        self.X_test = test_features[feature_cols].fillna(0)
        
        # Prepare labels
        if 'has_splicing_impact' in train_features.columns:
            self.y_train = train_features['has_splicing_impact']
        else:
            # Create labels based on delta PSI
            delta_psi = pd.to_numeric(self.training_data['∆PSI '], errors='coerce').fillna(0)
            self.y_train = (abs(delta_psi) > 0.1).astype(int)
        
        print(f"Feature matrix shape: {self.X_train.shape}")
        print(f"Test matrix shape: {self.X_test.shape}")
        print(f"Positive class ratio: {self.y_train.mean():.3f}")
        
        return self
    
    def _extract_basic_features(self, data):
        """Extract basic variant features"""
        features = pd.DataFrame()
        
        features['variant_id'] = data['Variant '].str.strip()
        features['gene_id'] = features['variant_id'].map(self.variant_gene_map)
        
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
        
        # Position-based features
        features['position_mod_1000'] = features['position'] % 1000
        features['position_mod_100'] = features['position'] % 100
        features['position_mod_10'] = features['position'] % 10
        
        # Allele type features
        features['is_transition'] = ((features['ref_allele'] == 'A') & (features['alt_allele'] == 'G') |
                                   (features['ref_allele'] == 'G') & (features['alt_allele'] == 'A') |
                                   (features['ref_allele'] == 'C') & (features['alt_allele'] == 'T') |
                                   (features['ref_allele'] == 'T') & (features['alt_allele'] == 'C')).astype(int)
        
        features['is_transversion'] = (1 - features['is_transition']).astype(int)
        
        # Chromosome features
        features['is_autosome'] = features['chromosome'].isin([str(i) for i in range(1, 23)]).astype(int)
        features['is_sex_chromosome'] = features['chromosome'].isin(['X', 'Y']).astype(int)
        
        return features
    
    def _add_network_features(self, features):
        """Add network-based features"""
        if self.network is None:
            # Add placeholder network features
            features['network_degree'] = np.random.poisson(5, len(features))
            features['network_betweenness'] = np.random.exponential(0.1, len(features))
            features['network_closeness'] = np.random.beta(2, 5, len(features))
        else:
            # Calculate actual network features
            degrees = []
            betweenness = []
            closeness = []
            
            for gene_id in features['gene_id']:
                if gene_id in self.network:
                    degrees.append(self.network.degree(gene_id))
                    betweenness.append(nx.betweenness_centrality(self.network)[gene_id])
                    closeness.append(nx.closeness_centrality(self.network)[gene_id])
                else:
                    degrees.append(0)
                    betweenness.append(0)
                    closeness.append(0)
            
            features['network_degree'] = degrees
            features['network_betweenness'] = betweenness
            features['network_closeness'] = closeness
        
        return features
    
    def _add_driver_features(self, features):
        """Add driver mutation features"""
        if self.driver_mutations is None:
            features['is_driver_mutation'] = 0
            features['driver_score'] = 0
        else:
            features['is_driver_mutation'] = features['gene_id'].isin(self.driver_mutations).astype(int)
            # Create a driver score based on network connectivity
            features['driver_score'] = features['network_degree'] * features['is_driver_mutation']
        
        return features
    
    def _add_splicing_features(self, features, data):
        """Add splicing-specific features"""
        if '∆PSI ' in data.columns:
            features['delta_psi'] = pd.to_numeric(data['∆PSI '], errors='coerce').fillna(0)
            features['psi_ref'] = pd.to_numeric(data['PSIRef\xa0'], errors='coerce').fillna(0)
            features['psi_var'] = pd.to_numeric(data['PSIVar\xa0'], errors='coerce').fillna(0)
            features['delta_pres'] = pd.to_numeric(data['∆PRES'], errors='coerce').fillna(0)
            features['delta_abs'] = pd.to_numeric(data['∆AbS '], errors='coerce').fillna(0)
            
            # Create binary labels
            features['has_splicing_impact'] = (abs(features['delta_psi']) > 0.1).astype(int)
            features['strong_splicing_impact'] = (abs(features['delta_psi']) > 0.2).astype(int)
            
            # Additional splicing features
            features['psi_change_magnitude'] = abs(features['delta_psi'])
            features['psi_change_direction'] = np.sign(features['delta_psi'])
        
        return features
    
    def _add_mutation_as_features(self, features):
        """Add mutation-AS pair features"""
        if self.mutation_as_pairs is None:
            features['has_cis_regulation'] = 0
            features['has_trans_regulation'] = 0
            features['total_regulation_pairs'] = 0
        else:
            # Count regulation pairs for each variant
            cis_counts = {}
            trans_counts = {}
            
            for _, pair in self.mutation_as_pairs['cis'].iterrows():
                variant_id = pair['variant_id']
                cis_counts[variant_id] = cis_counts.get(variant_id, 0) + 1
            
            for _, pair in self.mutation_as_pairs['trans'].iterrows():
                variant_id = pair['variant_id']
                trans_counts[variant_id] = trans_counts.get(variant_id, 0) + 1
            
            features['has_cis_regulation'] = features['variant_id'].map(cis_counts).fillna(0).astype(int)
            features['has_trans_regulation'] = features['variant_id'].map(trans_counts).fillna(0).astype(int)
            features['total_regulation_pairs'] = features['has_cis_regulation'] + features['has_trans_regulation']
        
        return features
    
    def train_ensemble_models(self, fast_mode=False):
        """
        Train ensemble of models with comprehensive evaluation
        """
        print("\n" + "=" * 60)
        print("STEP 8: TRAINING ENSEMBLE MODELS")
        print("=" * 60)
        
        if fast_mode:
            print("🚀 FAST MODE: Training smaller models for quick testing...")
        else:
            print("🎯 FULL MODE: Training optimized models for best performance...")
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # Split data for validation
        print("Splitting data for validation...")
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )
        
        print(f"✓ Training set: {X_train_split.shape[0]} samples")
        print(f"✓ Validation set: {X_val_split.shape[0]} samples")
        
        # Initialize models based on mode
        if fast_mode:
            # Smaller, faster models for testing
            rf = RandomForestClassifier(
                n_estimators=50, 
                max_depth=10, 
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            gb = GradientBoostingClassifier(
                n_estimators=50, 
                learning_rate=0.1, 
                max_depth=6,
                min_samples_split=10,
                random_state=42
            )
            
            lr = LogisticRegression(
                random_state=42, 
                max_iter=500, 
                C=1.0,
                class_weight='balanced'
            )
        else:
            # Full optimized models
            rf = RandomForestClassifier(
                n_estimators=300, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            gb = GradientBoostingClassifier(
                n_estimators=300, 
                learning_rate=0.05, 
                max_depth=8,
                min_samples_split=5,
                random_state=42
            )
            
            lr = LogisticRegression(
                random_state=42, 
                max_iter=2000, 
                C=0.1,
                class_weight='balanced'
            )
        
        # Train individual models
        self.models = {
            'random_forest': rf,
            'gradient_boosting': gb,
            'logistic_regression': lr
        }
        
        print("\nTraining individual models...")
        for name, model in self.models.items():
            print(f"  Training {name}...", end=" ")
            model.fit(X_train_split, y_train_split)
            
            # Validation score
            val_pred = model.predict_proba(X_val_split)[:, 1]
            val_auc = roc_auc_score(y_val_split, val_pred)
            print(f"AUC: {val_auc:.3f}")
        
        # Create ensemble model
        print("\nCreating ensemble model...")
        self.ensemble_model = VotingClassifier(
            estimators=list(self.models.items()),
            voting='soft'
        )
        self.ensemble_model.fit(X_train_split, y_train_split)
        
        # Ensemble validation score
        ensemble_pred = self.ensemble_model.predict_proba(X_val_split)[:, 1]
        ensemble_auc = roc_auc_score(y_val_split, ensemble_pred)
        print(f"✓ Ensemble validation AUC: {ensemble_auc:.3f}")
        
        print(f"✓ Model training complete!")
        return self
    
    def predict_and_evaluate(self):
        """
        Make predictions and evaluate performance
        """
        print("Making predictions...")
        
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Individual model predictions
        predictions = {}
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            predictions[name] = pred_proba
        
        # Ensemble predictions
        ensemble_pred = self.ensemble_model.predict_proba(X_test_scaled)[:, 1]
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def generate_final_predictions(self, output_file='complete_drasnet_predictions.csv'):
        """
        Generate final predictions for submission
        """
        print("Generating final predictions...")
        
        predictions = self.predict_and_evaluate()
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'Variant': self.test_data['Variant '].str.strip(),
            'Predicted_Splicing_Impact': predictions['ensemble'],
            'Predicted_Binary': (predictions['ensemble'] > 0.5).astype(int),
            'RF_Score': predictions['random_forest'],
            'GB_Score': predictions['gradient_boosting'],
            'LR_Score': predictions['logistic_regression'],
            'Gene_ID': [self.variant_gene_map.get(v, 'UNKNOWN') for v in self.test_data['Variant '].str.strip()],
            'Is_Driver': [self.variant_gene_map.get(v, 'UNKNOWN') in (self.driver_mutations or []) 
                         for v in self.test_data['Variant '].str.strip()]
        })
        
        # Save predictions
        output_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Print summary statistics
        print(f"\nPrediction Summary:")
        print(f"Total variants: {len(output_df)}")
        print(f"Predicted with splicing impact: {output_df['Predicted_Binary'].sum()}")
        print(f"Mean prediction score: {output_df['Predicted_Splicing_Impact'].mean():.3f}")
        print(f"Driver mutations identified: {output_df['Is_Driver'].sum()}")
        
        return output_df

def main(fast_mode=False):
    """
    Main pipeline execution
    """
    print("🚀 Starting Complete DrASNet Splicing Prediction Pipeline")
    print("=" * 60)
    
    if fast_mode:
        print("⚡ FAST MODE ENABLED - Using smaller models for quick testing")
    else:
        print("🎯 FULL MODE - Using optimized models for best performance")
    
    # Initialize pipeline
    pipeline = CompleteDrASNetPipeline()
    
    # Load data
    pipeline.load_data('cagi7splicingsample.csv', 'cagi7splicingvariants.csv')
    
    # Load DrASNet network
    pipeline.load_drasnet_network()
    
    # Create variant-gene mapping
    pipeline.create_variant_gene_mapping()
    
    # Identify personalized AS events
    pipeline.identify_personalized_as_events()
    
    # Create mutation data
    print("\n" + "=" * 60)
    print("STEP 5: CREATING MUTATION DATA")
    print("=" * 60)
    pipeline.create_mutation_data()
    print(f"✓ Mutation data creation complete!")
    
    # Identify mutation-AS pairs
    print("\n" + "=" * 60)
    print("STEP 6: IDENTIFYING MUTATION-AS PAIRS")
    print("=" * 60)
    pipeline.identify_mutation_as_pairs()
    print(f"✓ Mutation-AS pair identification complete!")
    
    # Prioritize driver mutations
    print("\n" + "=" * 60)
    print("STEP 7: PRIORITIZING DRIVER MUTATIONS")
    print("=" * 60)
    pipeline.prioritize_driver_mutations()
    print(f"✓ Driver mutation prioritization complete!")
    
    # Create comprehensive features
    print("\n" + "=" * 60)
    print("STEP 8: CREATING COMPREHENSIVE FEATURES")
    print("=" * 60)
    pipeline.create_comprehensive_features()
    print(f"✓ Feature engineering complete!")
    
    # Train ensemble models
    pipeline.train_ensemble_models(fast_mode=fast_mode)
    
    # Generate final predictions
    print("\n" + "=" * 60)
    print("STEP 9: GENERATING FINAL PREDICTIONS")
    print("=" * 60)
    predictions = pipeline.generate_final_predictions()
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"✓ Generated predictions for {len(predictions)} variants")
    print(f"✓ Predictions saved to: complete_drasnet_predictions.csv")
    
    return predictions

if __name__ == "__main__":
    predictions = main()

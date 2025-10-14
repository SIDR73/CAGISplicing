#!/usr/bin/env python3
"""
Complete DrASNet Pipeline Runner
Runs the full pipeline with real DrASNet data integration.
"""

import sys
import os
from complete_drasnet_pipeline import CompleteDrASNetPipeline

def main():
    """
    Run the complete DrASNet pipeline
    """
    print("Complete DrASNet Splicing Prediction Pipeline")
    print("=" * 50)
    
    # Check if data files exist
    training_file = 'cagi7splicingsample.csv'
    test_file = 'cagi7splicingvariants.csv'
    drasnet_data_path = 'DrASNet_data'
    
    if not os.path.exists(training_file):
        print(f"Error: Training file {training_file} not found!")
        return 1
    
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found!")
        return 1
    
    if not os.path.exists(drasnet_data_path):
        print(f"Warning: DrASNet data directory {drasnet_data_path} not found!")
        print("Pipeline will use synthetic network data.")
    
    try:
        # Initialize and run pipeline
        pipeline = CompleteDrASNetPipeline(drasnet_data_path=drasnet_data_path)
        
        print("Step 1: Loading CAGI7 data...")
        pipeline.load_data(training_file, test_file)
        
        print("Step 2: Loading DrASNet network...")
        pipeline.load_drasnet_network()
        
        print("Step 3: Creating variant-gene mapping...")
        pipeline.create_variant_gene_mapping()
        
        print("Step 4: Identifying personalized AS events...")
        pipeline.identify_personalized_as_events()
        
        print("Step 5: Creating mutation data...")
        pipeline.create_mutation_data()
        
        print("Step 6: Identifying mutation-AS pairs...")
        pipeline.identify_mutation_as_pairs()
        
        print("Step 7: Prioritizing driver mutations...")
        pipeline.prioritize_driver_mutations()
        
        print("Step 8: Creating comprehensive features...")
        pipeline.create_comprehensive_features()
        
        print("Step 9: Training ensemble models...")
        pipeline.train_ensemble_models()
        
        print("Step 10: Generating final predictions...")
        predictions = pipeline.generate_final_predictions()
        
        print("\nPipeline completed successfully!")
        print(f"Generated predictions for {len(predictions)} variants")
        print(f"Predictions saved to: complete_drasnet_predictions.csv")
        
        return 0
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

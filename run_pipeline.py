#!/usr/bin/env python3
"""
Simple runner script for the splicing prediction pipeline
"""

import sys
import os
from advanced_splicing_pipeline import AdvancedDrASNetPipeline

def main():
    """
    Run the splicing prediction pipeline
    """
    print("CAGI7 Splicing Variant Prediction Pipeline")
    print("=" * 50)
    
    # Check if data files exist
    training_file = 'cagi7splicingsample.csv'
    test_file = 'cagi7splicingvariants.csv'
    
    if not os.path.exists(training_file):
        print(f"Error: Training file {training_file} not found!")
        return 1
    
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found!")
        return 1
    
    try:
        # Initialize and run pipeline
        pipeline = AdvancedDrASNetPipeline()
        
        # Load data
        pipeline.load_data(training_file, test_file)
        
        # Create network
        pipeline.create_ppi_network()
        
        # Identify personalized AS events
        pipeline.identify_personalized_as_events()
        
        # Identify mutation-AS pairs
        pipeline.identify_mutation_as_pairs()
        
        # Prioritize driver mutations
        pipeline.prioritize_driver_mutations()
        
        # Create advanced features
        pipeline.create_advanced_features()
        
        # Train ensemble models
        pipeline.train_ensemble_models()
        
        # Generate final predictions
        predictions = pipeline.generate_final_predictions()
        
        print("\nPipeline completed successfully!")
        print(f"Generated predictions for {len(predictions)} variants")
        
        return 0
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

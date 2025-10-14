#!/usr/bin/env python3
"""
Simple runner script for the ML pipeline
"""

import os
import sys

def main():
    """Run the complete ML pipeline"""
    print("🚀 CAGI7 Splicing Prediction Pipeline")
    print("=" * 50)
    
    # Check if features exist
    if not os.path.exists('train_features.csv') or not os.path.exists('test_features.csv'):
        print("❌ Feature files not found!")
        print("Please run: python extract_features.py")
        print("This will create train_features.csv and test_features.csv")
        return
    
    # Import and run ML pipeline
    from ml_pipeline import MLPipeline
    
    # Parse arguments
    fast_mode = '--fast' in sys.argv
    threshold = 0.1
    
    if '--threshold' in sys.argv:
        idx = sys.argv.index('--threshold')
        if idx + 1 < len(sys.argv):
            threshold = float(sys.argv[idx + 1])
    
    print(f"Fast mode: {fast_mode}")
    print(f"Threshold: {threshold}")
    print()
    
    # Run ML pipeline with proper feature selection (no data leakage)
    pipeline = MLPipeline()
    
    # Load features
    pipeline.load_features()
    pipeline.prepare_data()
    
    # Remove ground truth splicing features to prevent data leakage
    print("Removing ground truth splicing features to prevent data leakage...")
    splicing_features = ['delta_psi', 'psi_ref', 'psi_var', 'delta_pres', 'delta_abs', 
                        'psi_change_magnitude', 'psi_change_direction', 'psi_ratio']
    
    # Filter out splicing features from feature columns
    original_features = len(pipeline.feature_cols)
    pipeline.feature_cols = [col for col in pipeline.feature_cols if col not in splicing_features]
    removed_features = original_features - len(pipeline.feature_cols)
    
    print(f"✓ Removed {removed_features} ground truth features")
    print(f"✓ Using {len(pipeline.feature_cols)} predictive features")
    
    # Re-prepare data with filtered features
    pipeline.X_train = pipeline.train_features[pipeline.feature_cols].fillna(0)
    pipeline.X_test = pipeline.test_features[pipeline.feature_cols].fillna(0)
    
    # Train models
    pipeline.train_models(fast_mode=fast_mode)
    predictions = pipeline.generate_predictions(threshold=threshold)
    feature_importance = pipeline.analyze_features()
    cv_results = pipeline.analyze_cv_results()
    
    print("\n" + "=" * 60)
    print("🎉 ML PIPELINE COMPLETE!")
    print("=" * 60)
    print("Files created:")
    print("  - ml_predictions.csv: Final predictions")
    print("  - feature_importance.csv: Feature importance analysis")
    print("  - cv_results.csv: Cross-validation results")
    print()
    print("Ready for submission!")

if __name__ == "__main__":
    main()

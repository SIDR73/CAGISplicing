#!/usr/bin/env python3
"""
ML Pipeline using pre-extracted features
Fast machine learning with pre-computed features.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    """
    Machine learning pipeline using pre-extracted features
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def load_features(self, train_file='train_features.csv', test_file='test_features.csv'):
        """Load pre-extracted features"""
        print("=" * 60)
        print("LOADING PRE-EXTRACTED FEATURES")
        print("=" * 60)
        
        self.train_features = pd.read_csv(train_file)
        self.test_features = pd.read_csv(test_file)
        
        print(f"✓ Training features: {self.train_features.shape}")
        print(f"✓ Test features: {self.test_features.shape}")
        
        return self
    
    def prepare_data(self, target_col='has_splicing_impact'):
        """Prepare data for ML"""
        print("\n" + "=" * 60)
        print("PREPARING DATA FOR ML")
        print("=" * 60)
        
        # Get numeric columns only
        numeric_cols = self.train_features.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = [col for col in numeric_cols if col not in ['variant_id', 'gene_id', target_col, 'strong_splicing_impact']]
        
        # Prepare matrices
        self.X_train = self.train_features[self.feature_cols].fillna(0)
        self.X_test = self.test_features[self.feature_cols].fillna(0)
        self.y_train = self.train_features[target_col]
        
        print(f"✓ Features: {len(self.feature_cols)} features")
        print(f"✓ Training: {self.X_train.shape[0]} samples")
        print(f"✓ Test: {self.X_test.shape[0]} samples")
        print(f"✓ Positive class: {self.y_train.mean():.3f}")
        
        return self
    
    def train_models(self, fast_mode=False):
        """Train ensemble models with cross-validation"""
        print("\n" + "=" * 60)
        print("TRAINING MODELS WITH CROSS-VALIDATION")
        print("=" * 60)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # Model parameters
        if fast_mode:
            print("🚀 Fast mode: Using smaller models")
            rf_params = {'n_estimators': 50, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
            gb_params = {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42}
        else:
            print("🔥 Full mode: Using larger models")
            rf_params = {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}
            gb_params = {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 8, 'min_samples_split': 5, 'random_state': 42}
        
        # Models
        rf = RandomForestClassifier(**rf_params)
        gb = GradientBoostingClassifier(**gb_params)
        lr = LogisticRegression(random_state=42, max_iter=1000, C=0.1, class_weight='balanced')
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train individual models with cross-validation
        self.models = {
            'random_forest': rf,
            'gradient_boosting': gb,
            'logistic_regression': lr
        }
        
        print("Cross-validation results:")
        cv_scores = {}
        for name, model in self.models.items():
            print(f"  {name}...", end=" ")
            # Cross-validation scores
            scores = cross_val_score(model, X_train_scaled, self.y_train, cv=cv, scoring='roc_auc')
            cv_scores[name] = scores
            print(f"CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")
        
        # Train final models on full training set
        print("\nTraining final models on full training set...")
        for name, model in self.models.items():
            model.fit(X_train_scaled, self.y_train)
        
        # Create ensemble
        print("Creating ensemble model...")
        self.ensemble_model = VotingClassifier(
            estimators=list(self.models.items()),
            voting='soft'
        )
        self.ensemble_model.fit(X_train_scaled, self.y_train)
        
        # Cross-validation for ensemble
        ensemble_scores = cross_val_score(self.ensemble_model, X_train_scaled, self.y_train, cv=cv, scoring='roc_auc')
        print(f"✓ Ensemble CV AUC: {ensemble_scores.mean():.3f} ± {ensemble_scores.std():.3f}")
        
        # Store CV results for analysis
        self.cv_scores = cv_scores
        self.ensemble_cv_scores = ensemble_scores
        
        return self
    
    def generate_predictions(self, threshold=0.1):
        """Generate predictions"""
        print("\n" + "=" * 60)
        print("GENERATING PREDICTIONS")
        print("=" * 60)
        
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Individual model predictions
        predictions = {}
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            predictions[name] = pred_proba
        
        # Ensemble predictions
        ensemble_pred = self.ensemble_model.predict_proba(X_test_scaled)[:, 1]
        predictions['ensemble'] = ensemble_pred
        
        # Create output
        output_df = pd.DataFrame({
            'Variant': self.test_features['variant_id'],
            'Predicted_Splicing_Impact': predictions['ensemble'],
            'Predicted_Binary': (predictions['ensemble'] > threshold).astype(int),
            'RF_Score': predictions['random_forest'],
            'GB_Score': predictions['gradient_boosting'],
            'LR_Score': predictions['logistic_regression'],
            'Gene_ID': self.test_features['gene_id']
        })
        
        # Save
        output_df.to_csv('ml_predictions.csv', index=False)
        
        print(f"✓ Generated predictions for {len(output_df)} variants")
        print(f"✓ Predicted {output_df['Predicted_Binary'].sum()} with splicing impact (threshold: {threshold})")
        print(f"✓ Mean score: {output_df['Predicted_Splicing_Impact'].mean():.3f}")
        print(f"✓ Saved to: ml_predictions.csv")
        
        return output_df
    
    def analyze_features(self):
        """Analyze feature importance"""
        print("\n" + "=" * 60)
        print("FEATURE ANALYSIS")
        print("=" * 60)
        
        # Random Forest feature importance
        rf_importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': self.models['random_forest'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 10 most important features:")
        for i, (_, row) in enumerate(rf_importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['Feature']:<25} {row['Importance']:.4f}")
        
        # Save feature importance
        rf_importance.to_csv('feature_importance.csv', index=False)
        print(f"\n✓ Feature importance saved to: feature_importance.csv")
        
        return rf_importance
    
    def analyze_cv_results(self):
        """Analyze cross-validation results in detail"""
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION ANALYSIS")
        print("=" * 60)
        
        # Individual model CV results
        print("Individual Model Performance:")
        for name, scores in self.cv_scores.items():
            print(f"  {name:<20}: {scores.mean():.3f} ± {scores.std():.3f} (range: {scores.min():.3f}-{scores.max():.3f})")
        
        # Ensemble CV results
        ensemble_scores = self.ensemble_cv_scores
        print(f"\nEnsemble Performance:")
        print(f"  Mean CV AUC: {ensemble_scores.mean():.3f} ± {ensemble_scores.std():.3f}")
        print(f"  Range: {ensemble_scores.min():.3f} - {ensemble_scores.max():.3f}")
        
        # Performance interpretation
        mean_auc = ensemble_scores.mean()
        std_auc = ensemble_scores.std()
        
        print(f"\nPerformance Interpretation:")
        if mean_auc > 0.9:
            print("  ✅ Excellent performance (>0.9 AUC)")
        elif mean_auc > 0.8:
            print("  ✅ Good performance (>0.8 AUC)")
        elif mean_auc > 0.7:
            print("  ⚠️  Moderate performance (>0.7 AUC)")
        else:
            print("  ❌ Poor performance (<0.7 AUC)")
        
        if std_auc < 0.05:
            print("  ✅ Low variance - stable performance")
        elif std_auc < 0.1:
            print("  ⚠️  Moderate variance - some instability")
        else:
            print("  ❌ High variance - unstable performance")
        
        # Save CV results
        cv_results = pd.DataFrame({
            'Model': list(self.cv_scores.keys()) + ['ensemble'],
            'Mean_AUC': [scores.mean() for scores in self.cv_scores.values()] + [ensemble_scores.mean()],
            'Std_AUC': [scores.std() for scores in self.cv_scores.values()] + [ensemble_scores.std()],
            'Min_AUC': [scores.min() for scores in self.cv_scores.values()] + [ensemble_scores.min()],
            'Max_AUC': [scores.max() for scores in self.cv_scores.values()] + [ensemble_scores.max()]
        })
        cv_results.to_csv('cv_results.csv', index=False)
        print(f"\n✓ CV results saved to: cv_results.csv")
        
        return cv_results

def main(fast_mode=False, threshold=0.1):
    """Run the ML pipeline"""
    print("🤖 ML PIPELINE")
    print("=" * 60)
    print("Using pre-extracted features for fast ML development")
    print()
    
    try:
        pipeline = MLPipeline()
        
        # Run pipeline
        pipeline.load_features()
        pipeline.prepare_data()
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
        
        return predictions, feature_importance, cv_results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    fast_mode = '--fast' in sys.argv
    threshold = 0.1
    
    if '--threshold' in sys.argv:
        idx = sys.argv.index('--threshold')
        if idx + 1 < len(sys.argv):
            threshold = float(sys.argv[idx + 1])
    
    print(f"Fast mode: {fast_mode}")
    print(f"Threshold: {threshold}")
    print()
    
    main(fast_mode=fast_mode, threshold=threshold)

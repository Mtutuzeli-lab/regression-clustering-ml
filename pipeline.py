"""
COMPLETE ML PIPELINE - eCommerce Customer Spending Prediction

This script demonstrates the end-to-end machine learning pipeline:
1. Data Ingestion - Load and split data
2. Data Transformation - Preprocess and scale features
3. Model Training - Train and select best model
4. Prediction - Make predictions on new data

All artifacts are saved and can be reused for production deployment.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logger


def main():
    """Run the complete ML pipeline"""
    
    print("\n" + "=" * 80)
    print("ECOMMERCE CUSTOMER SPENDING PREDICTION - ML PIPELINE")
    print("=" * 80)
    
    try:
        # ============================================================
        # STEP 1: DATA INGESTION
        # ============================================================
        print("\n[STEP 1] DATA INGESTION")
        print("-" * 80)
        
        ingestion = DataIngestion()
        train_df, test_df = ingestion.initiate_data_ingestion(source='local')
        
        print(f"\nData ingestion completed:")
        print(f"  Train set: {train_df.shape[0]} samples")
        print(f"  Test set: {test_df.shape[0]} samples")
        print(f"  Artifacts created: train.csv, test.csv, raw_data.csv")
        
        # ============================================================
        # STEP 2: DATA TRANSFORMATION
        # ============================================================
        print("\n[STEP 2] DATA TRANSFORMATION")
        print("-" * 80)
        
        transformation = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor_path = transformation.initiate_data_transformation()
        
        print(f"\nData transformation completed:")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Preprocessor saved: {preprocessor_path}")
        
        # ============================================================
        # STEP 3: MODEL TRAINING
        # ============================================================
        print("\n[STEP 3] MODEL TRAINING")
        print("-" * 80)
        
        trainer = ModelTrainer()
        best_model, model_path, report_df = trainer.initiate_model_training(
            X_train, y_train, X_test, y_test
        )
        
        print(f"\nModel training completed:")
        print(f"  Best model: {report_df.iloc[0]['Model']}")
        print(f"  Test R²: {report_df.iloc[0]['Test_R2']:.4f}")
        print(f"  Test RMSE: ${report_df.iloc[0]['Test_RMSE']:.2f}")
        print(f"  Test MAE: ${report_df.iloc[0]['Test_MAE']:.2f}")
        print(f"  Model saved: {model_path}")
        
        # ============================================================
        # STEP 4: MODEL EVALUATION
        # ============================================================
        print("\n[STEP 4] MODEL EVALUATION")
        print("-" * 80)
        
        # Show top 5 models
        print("\nTop 5 Models Performance:")
        for i in range(min(5, len(report_df))):
            row = report_df.iloc[i]
            print(f"\n  {i+1}. {row['Model']}")
            print(f"     Test R²: {row['Test_R2']:.4f}")
            print(f"     Test RMSE: ${row['Test_RMSE']:.2f}")
            print(f"     Test MAE: ${row['Test_MAE']:.2f}")
        
        # ============================================================
        # STEP 5: SAMPLE PREDICTIONS
        # ============================================================
        print("\n[STEP 5] SAMPLE PREDICTIONS")
        print("-" * 80)
        
        # Make predictions on test set
        predictions = best_model.predict(X_test[:5])
        
        print("\nSample predictions (first 5 customers):")
        print(f"\n{'Actual':<15} {'Predicted':<15} {'Difference':<15}")
        print("-" * 45)
        for i in range(5):
            diff = predictions[i] - y_test[i]
            print(f"${y_test[i]:<14.2f} ${predictions[i]:<14.2f} ${diff:<14.2f}")
        
        # ============================================================
        # PIPELINE SUMMARY
        # ============================================================
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        print("\nArtifacts Created:")
        print("  artifacts/")
        print("    - raw_data.csv          (Original data)")
        print("    - train.csv             (Training set)")
        print("    - test.csv              (Test set)")
        print("    - preprocessor.pkl      (Feature scaler)")
        print("    - model.pkl             (Trained model)")
        print("    - model_report.csv      (Model comparison)")
        
        print("\nProduction Deployment:")
        print("  1. Load preprocessor: joblib.load('artifacts/preprocessor.pkl')")
        print("  2. Load model: joblib.load('artifacts/model.pkl')")
        print("  3. Transform new data: preprocessor.transform(new_data)")
        print("  4. Predict: model.predict(transformed_data)")
        
        print("\n" + "=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"\nERROR: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

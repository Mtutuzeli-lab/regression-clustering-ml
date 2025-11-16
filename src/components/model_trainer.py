import os
import sys
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exception import CustomException
from logger import logger


@dataclass
class ModelTrainerConfig:
    """
    Configuration for model trainer component
    """
    # Model artifact paths
    trained_model_path: str = os.path.join('artifacts', 'model.pkl')
    model_report_path: str = os.path.join('artifacts', 'model_report.csv')
    
    # Performance threshold
    expected_r2_score: float = 0.6  # Minimum acceptable R² score
    
    # MLflow settings
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "Customer_Spending_Prediction"


class ModelTrainer:
    """
    Handles model training and evaluation:
    1. Train multiple regression models
    2. Evaluate and compare performance
    3. Select best model
    4. Save best model as artifact
    5. Generate performance report
    
    Models trained:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - ElasticNet
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - AdaBoost
    - Support Vector Regression (SVR)
    - K-Nearest Neighbors
    """
    
    def __init__(self, config: Optional[ModelTrainerConfig] = None):
        """
        Initialize ModelTrainer component
        
        Args:
            config: ModelTrainerConfig object with paths and settings
        """
        self.config = config if config else ModelTrainerConfig()
        logger.info("ModelTrainer component initialized")
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)
        logger.info(f"MLflow tracking enabled: {self.config.mlflow_experiment_name}")
    
    def get_models(self) -> Dict:
        """
        Get dictionary of models to train
        
        Returns:
            Dict: Dictionary of model names and instances
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor()
        }
        
        logger.info(f"Initialized {len(models)} models for training")
        return models
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dict: Dictionary of metrics
        """
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            return {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise CustomException(e, sys)
    
    def train_and_evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Train and evaluate all models with MLflow tracking
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            pd.DataFrame: Model comparison report
        """
        try:
            models = self.get_models()
            results = []
            
            logger.info("=" * 80)
            logger.info("MODEL TRAINING AND EVALUATION STARTED (with MLflow tracking)")
            logger.info("=" * 80)
            
            # Start parent MLflow run
            with mlflow.start_run(run_name=f"Training_Pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Log dataset info
                mlflow.log_param("train_samples", X_train.shape[0])
                mlflow.log_param("test_samples", X_test.shape[0])
                mlflow.log_param("n_features", X_train.shape[1])
                
                for model_name, model in models.items():
                    try:
                        logger.info(f"\nTraining {model_name}...")
                        
                        # Start child run for each model
                        with mlflow.start_run(run_name=model_name, nested=True):
                            
                            # Log model type
                            mlflow.log_param("model_type", model_name)
                            mlflow.log_param("model_class", model.__class__.__name__)
                            
                            # Train model
                            model.fit(X_train, y_train)
                            
                            # Predictions
                            y_train_pred = model.predict(X_train)
                            y_test_pred = model.predict(X_test)
                            
                            # Evaluate on training set
                            train_metrics = self.evaluate_model(y_train, y_train_pred)
                            
                            # Evaluate on test set
                            test_metrics = self.evaluate_model(y_test, y_test_pred)
                            
                            # Log metrics to MLflow
                            mlflow.log_metrics({
                                "train_rmse": train_metrics['RMSE'],
                                "train_mae": train_metrics['MAE'],
                                "train_r2": train_metrics['R2'],
                                "test_rmse": test_metrics['RMSE'],
                                "test_mae": test_metrics['MAE'],
                                "test_r2": test_metrics['R2']
                            })
                            
                            # Log model to MLflow
                            mlflow.sklearn.log_model(model, "model")
                            
                            # Store results
                            results.append({
                                'Model': model_name,
                                'Train_RMSE': train_metrics['RMSE'],
                                'Train_MAE': train_metrics['MAE'],
                                'Train_R2': train_metrics['R2'],
                                'Test_RMSE': test_metrics['RMSE'],
                                'Test_MAE': test_metrics['MAE'],
                                'Test_R2': test_metrics['R2']
                            })
                            
                            logger.info(f"{model_name} - Test R²: {test_metrics['R2']:.4f}, "
                                      f"Test RMSE: ${test_metrics['RMSE']:.2f}")
                            logger.info(f"✓ Logged to MLflow: {model_name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to train {model_name}: {str(e)}")
                        continue
                
                # Create DataFrame and sort by Test R2
                report_df = pd.DataFrame(results)
                report_df = report_df.sort_values('Test_R2', ascending=False)
                
                # Log best model info
                best_model_name = report_df.iloc[0]['Model']
                best_r2 = report_df.iloc[0]['Test_R2']
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_test_r2", best_r2)
                
                logger.info("=" * 80)
                logger.info("MODEL TRAINING COMPLETED")
                logger.info(f"All experiments logged to MLflow experiment: {self.config.mlflow_experiment_name}")
                logger.info("=" * 80)
            
            return report_df
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)
    
    def save_model(self, model, file_path: Optional[str] = None) -> str:
        """
        Save trained model as artifact
        
        Args:
            model: Trained model object
            file_path: Path to save model. If None, uses config default
            
        Returns:
            str: Path where model was saved
        """
        try:
            path = file_path if file_path else self.config.trained_model_path
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            joblib.dump(model, path)
            
            logger.info(f"Model saved to: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise CustomException(e, sys)
    
    def load_model(self, file_path: Optional[str] = None):
        """
        Load saved model artifact
        
        Args:
            file_path: Path to saved model. If None, uses config default
            
        Returns:
            Loaded model object
        """
        try:
            path = file_path if file_path else self.config.trained_model_path
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found at: {path}")
            
            model = joblib.load(path)
            logger.info(f"Model loaded from: {path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise CustomException(e, sys)
    
    def save_report(self, report_df: pd.DataFrame, file_path: Optional[str] = None) -> str:
        """
        Save model comparison report
        
        Args:
            report_df: Model comparison dataframe
            file_path: Path to save report. If None, uses config default
            
        Returns:
            str: Path where report was saved
        """
        try:
            path = file_path if file_path else self.config.model_report_path
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save report
            report_df.to_csv(path, index=False)
            
            logger.info(f"Model report saved to: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_model_training(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray) -> Tuple[object, str, pd.DataFrame]:
        """
        Main method to initiate model training pipeline
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Tuple: (best_model, model_path, report_df)
        """
        try:
            logger.info("=" * 80)
            logger.info("MODEL TRAINING PIPELINE STARTED")
            logger.info("=" * 80)
            
            # Train and evaluate all models
            report_df = self.train_and_evaluate_models(X_train, y_train, X_test, y_test)
            
            # Get best model
            best_model_name = report_df.iloc[0]['Model']
            best_r2_score = report_df.iloc[0]['Test_R2']
            
            logger.info(f"\nBest Model: {best_model_name}")
            logger.info(f"Best Test R²: {best_r2_score:.4f}")
            
            # Check if best model meets threshold
            if best_r2_score < self.config.expected_r2_score:
                logger.warning(f"Best model R² ({best_r2_score:.4f}) is below threshold "
                             f"({self.config.expected_r2_score})")
            
            # Train best model on full dataset
            logger.info(f"\nRetraining best model ({best_model_name}) on training data...")
            models = self.get_models()
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            
            # Save best model
            model_path = self.save_model(best_model)
            
            # Save report
            report_path = self.save_report(report_df)
            
            # Final evaluation on test set
            y_test_pred = best_model.predict(X_test)
            final_metrics = self.evaluate_model(y_test, y_test_pred)
            
            logger.info("=" * 80)
            logger.info("MODEL TRAINING PIPELINE COMPLETED")
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"Test RMSE: ${final_metrics['RMSE']:.2f}")
            logger.info(f"Test MAE: ${final_metrics['MAE']:.2f}")
            logger.info(f"Test R²: {final_metrics['R2']:.4f}")
            logger.info(f"Model saved: {model_path}")
            logger.info(f"Report saved: {report_path}")
            logger.info("=" * 80)
            
            return best_model, model_path, report_df
            
        except Exception as e:
            logger.error(f"Model training pipeline failed: {str(e)}")
            raise CustomException(e, sys)


# Example usage and testing
if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print("TESTING MODEL TRAINER COMPONENT")
        print("=" * 80)
        
        # Import data transformation to get preprocessed data
        from data_transformation import DataTransformation
        
        print("\n[STEP 1] Loading and transforming data...")
        transformation = DataTransformation()
        X_train, X_test, y_train, y_test, _ = transformation.initiate_data_transformation()
        
        print(f"Data loaded:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        
        # Initialize model trainer
        print("\n[STEP 2] Training models...")
        model_trainer = ModelTrainer()
        
        # Train models
        best_model, model_path, report_df = model_trainer.initiate_model_training(
            X_train, y_train, X_test, y_test
        )
        
        print(f"\n" + "=" * 80)
        print("MODEL COMPARISON REPORT")
        print("=" * 80)
        print(report_df.to_string(index=False))
        
        # Test loading saved model
        print("\n[STEP 3] Testing saved model...")
        loaded_model = model_trainer.load_model()
        
        # Make predictions with loaded model
        sample_predictions = loaded_model.predict(X_test[:5])
        print(f"\nSample predictions from loaded model:")
        print(f"Predicted: {sample_predictions}")
        print(f"Actual:    {y_test[:5]}")
        
        print("\n" + "=" * 80)
        print("MODEL TRAINER TESTING COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

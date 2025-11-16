import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import logging
from datetime import datetime

def setup_logging(log_file='logs/pipeline.log'):
    """
    Setup logging configuration for the entire project
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def save_object(file_path, obj):
    """
    Save object as pickle file
    
    Args:
        file_path: Path where object will be saved
        obj: Object to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Object saved successfully at {file_path}")
        
    except Exception as e:
        logging.error(f"Error saving object: {str(e)}")
        raise

def load_object(file_path):
    """
    Load object from pickle file
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded object
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
            
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
        
    except Exception as e:
        logging.error(f"Error loading object: {str(e)}")
        raise

def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids=None):
    """
    Train and evaluate multiple models with optional hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of model names and model objects
        param_grids: Dictionary of parameter grids for GridSearchCV (optional)
        
    Returns:
        Dictionary with model performance metrics
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")
            
            # Hyperparameter tuning if param_grids provided
            if param_grids and model_name in param_grids:
                grid_search = GridSearchCV(
                    model, 
                    param_grids[model_name], 
                    cv=3, 
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                logging.info(f"Best params for {model_name}: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = {
                'model': model,
                'train_r2': train_score,
                'test_r2': test_score,
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'mae': mean_absolute_error(y_test, y_test_pred)
            }
            
            logging.info(f"{model_name} - Test R²: {test_score:.4f}")
        
        return report
        
    except Exception as e:
        logging.error(f"Error evaluating models: {str(e)}")
        raise

def get_model_performance_summary(model, X_test, y_test, model_name="Model"):
    """
    Get comprehensive performance metrics for a model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        Dictionary of performance metrics
    """
    try:
        y_pred = model.predict(X_test)
        
        metrics = {
            'model_name': model_name,
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error getting model performance: {str(e)}")
        raise

def log_metrics_to_csv(metrics_dict, file_path='artifacts/metrics_log.csv'):
    """
    Log model metrics to CSV for Power BI monitoring
    
    Args:
        metrics_dict: Dictionary containing metrics
        file_path: Path to CSV file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df = pd.DataFrame([metrics_dict])
        
        # Append to existing file or create new
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, mode='w', header=True, index=False)
            
        logging.info(f"Metrics logged to {file_path}")
        
    except Exception as e:
        logging.error(f"Error logging metrics: {str(e)}")
        raise

def validate_data_quality(df, required_columns=None):
    """
    Validate data quality and completeness
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names (optional)
        
    Returns:
        Dictionary with validation results
    """
    try:
        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            validation_report['missing_required_columns'] = list(missing_cols)
        
        # Calculate missing percentage
        total_cells = len(df) * len(df.columns)
        total_missing = df.isnull().sum().sum()
        validation_report['missing_percentage'] = (total_missing / total_cells) * 100
        
        logging.info(f"Data validation complete: {len(df)} rows, {df.isnull().sum().sum()} missing values")
        
        return validation_report
        
    except Exception as e:
        logging.error(f"Error validating data: {str(e)}")
        raise

def check_missing_values(df):
    """
    Detailed missing value analysis
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with missing value statistics
    """
    try:
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum().values,
            'Missing_Percentage': (df.isnull().sum().values / len(df)) * 100
        })
        
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        
        return missing_data
        
    except Exception as e:
        logging.error(f"Error checking missing values: {str(e)}")
        raise

def create_directory_if_not_exists(directory_path):
    """
    Create directory if it doesn't exist
    
    Args:
        directory_path: Path to directory
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        logging.info(f"Directory ensured: {directory_path}")
        
    except Exception as e:
        logging.error(f"Error creating directory: {str(e)}")
        raise

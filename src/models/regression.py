"""
Regression models module for customer spending prediction
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle
import os

from ..exception import CustomException
from ..logger import logging

class RegressionModelTrainer:
    """
    Class to train and evaluate regression models for customer spending prediction
    """
    
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'elastic_net': ElasticNet(),
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'svr': SVR()
        }
        
        self.model_params = {
            'ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'lasso': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'elastic_net': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
            'random_forest': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
            'gradient_boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
            'svr': {'C': [1, 10, 100], 'gamma': ['scale', 'auto']}
        }
        
        self.trained_models = {}
        self.results = {}
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train all regression models and evaluate performance
        """
        try:
            logging.info("Starting regression model training")
            
            for name, model in self.models.items():
                logging.info(f"Training {name}")
                
                # Perform hyperparameter tuning if parameters exist
                if name in self.model_params:
                    grid_search = GridSearchCV(
                        model, 
                        self.model_params[name], 
                        cv=5, 
                        scoring='r2',
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    logging.info(f"{name} best parameters: {grid_search.best_params_}")
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = best_model.predict(X_train)
                test_pred = best_model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                # Store results
                self.trained_models[name] = best_model
                self.results[name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': np.sqrt(train_mse),
                    'test_rmse': np.sqrt(test_mse)
                }
                
                logging.info(f"{name} - Test R2: {test_r2:.4f}, Test RMSE: {np.sqrt(test_mse):.4f}")
            
            logging.info("Regression model training completed")
            return self.results
            
        except Exception as e:
            logging.error("Error in regression model training")
            raise CustomException(e, sys)
    
    def get_best_model(self, metric='test_r2'):
        """
        Get the best performing model based on specified metric
        """
        if not self.results:
            raise ValueError("No models trained yet. Please run train_models first.")
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x][metric])
        return best_model_name, self.trained_models[best_model_name]
    
    def save_models(self, save_dir='../../models/regression'):
        """
        Save all trained models
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            for name, model in self.trained_models.items():
                model_path = os.path.join(save_dir, f"{name}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logging.info(f"Saved {name} model to {model_path}")
            
            # Save results
            results_path = os.path.join(save_dir, "model_results.pkl")
            with open(results_path, 'wb') as f:
                pickle.dump(self.results, f)
            logging.info(f"Saved results to {results_path}")
            
        except Exception as e:
            logging.error("Error saving regression models")
            raise CustomException(e, sys)
"""
Deep Learning models module for customer spending prediction using Artificial Neural Networks
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json

from ..exception import CustomException
from ..logger import logging

class DeepLearningModelTrainer:
    """
    Class to train and evaluate Deep Learning models for customer spending prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.history = {}
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def create_simple_ann(self, input_dim, hidden_layers=[64, 32], dropout_rate=0.2, 
                         activation='relu', output_activation='linear', learning_rate=0.001):
        """
        Create a simple feedforward neural network for regression
        """
        try:
            model = keras.Sequential([
                layers.Dense(hidden_layers[0], activation=activation, input_shape=(input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(dropout_rate)
            ])
            
            # Add additional hidden layers
            for units in hidden_layers[1:]:
                model.add(layers.Dense(units, activation=activation))
                model.add(layers.BatchNormalization())
                model.add(layers.Dropout(dropout_rate))
            
            # Output layer
            model.add(layers.Dense(1, activation=output_activation))
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_absolute_error', 'mean_squared_error']
            )
            
            logging.info(f"Created Simple ANN with architecture: {hidden_layers}")
            return model
            
        except Exception as e:
            logging.error("Error creating simple ANN model")
            raise CustomException(e, sys)
    
    def create_deep_ann(self, input_dim, hidden_layers=[128, 64, 32, 16], dropout_rate=0.3,
                       activation='relu', output_activation='linear', learning_rate=0.001):
        """
        Create a deeper neural network for more complex pattern learning
        """
        try:
            model = keras.Sequential()
            
            # Input layer
            model.add(layers.Dense(hidden_layers[0], activation=activation, input_shape=(input_dim,)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
            
            # Hidden layers with progressive size reduction
            for i, units in enumerate(hidden_layers[1:], 1):
                model.add(layers.Dense(units, activation=activation))
                model.add(layers.BatchNormalization())
                # Increase dropout for deeper layers
                model.add(layers.Dropout(min(dropout_rate + 0.1 * i, 0.5)))
            
            # Output layer
            model.add(layers.Dense(1, activation=output_activation))
            
            # Compile with different optimizer for deeper networks
            optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
            model.compile(
                optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_absolute_error']
            )
            
            logging.info(f"Created Deep ANN with architecture: {hidden_layers}")
            return model
            
        except Exception as e:
            logging.error("Error creating deep ANN model")
            raise CustomException(e, sys)
    
    def create_wide_and_deep(self, input_dim, wide_features=None, deep_layers=[64, 32], 
                           dropout_rate=0.2, learning_rate=0.001):
        """
        Create a Wide & Deep model combining linear and deep learning
        """
        try:
            # Input layer
            inputs = layers.Input(shape=(input_dim,))
            
            # Wide component (linear)
            if wide_features is not None:
                wide_input = layers.Lambda(lambda x: tf.gather(x, wide_features, axis=1))(inputs)
                wide = layers.Dense(1, activation='linear')(wide_input)
            else:
                wide = layers.Dense(1, activation='linear')(inputs)
            
            # Deep component
            deep = layers.Dense(deep_layers[0], activation='relu')(inputs)
            deep = layers.BatchNormalization()(deep)
            deep = layers.Dropout(dropout_rate)(deep)
            
            for units in deep_layers[1:]:
                deep = layers.Dense(units, activation='relu')(deep)
                deep = layers.BatchNormalization()(deep)
                deep = layers.Dropout(dropout_rate)(deep)
            
            deep = layers.Dense(1, activation='linear')(deep)
            
            # Combine wide and deep
            combined = layers.Add()([wide, deep])
            
            # Create model
            model = keras.Model(inputs=inputs, outputs=combined)
            
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mean_absolute_error']
            )
            
            logging.info(f"Created Wide & Deep model with deep layers: {deep_layers}")
            return model
            
        except Exception as e:
            logging.error("Error creating Wide & Deep model")
            raise CustomException(e, sys)
    
    def prepare_data_for_dl(self, X_train, X_test, y_train, y_test, scale_features=True, scale_target=False):
        """
        Prepare data specifically for deep learning models
        """
        try:
            logging.info("Preparing data for deep learning")
            
            if scale_features:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                
                # Convert back to DataFrame to maintain feature names
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            else:
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
            
            # Optional target scaling for better convergence
            if scale_target:
                self.target_scaler = StandardScaler()
                y_train_scaled = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                y_test_scaled = self.target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
            else:
                y_train_scaled = y_train.copy()
                y_test_scaled = y_test.copy()
            
            logging.info(f"Data prepared - Features: {X_train_scaled.shape}, Target scaling: {scale_target}")
            return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
            
        except Exception as e:
            logging.error("Error in data preparation for deep learning")
            raise CustomException(e, sys)
    
    def train_dl_models(self, X_train, X_test, y_train, y_test, epochs=100, batch_size=32, 
                       validation_split=0.2, early_stopping_patience=15):
        """
        Train multiple deep learning models and compare performance
        """
        try:
            logging.info("Starting deep learning model training")
            
            # Prepare data
            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self.prepare_data_for_dl(
                X_train, X_test, y_train, y_test
            )
            
            input_dim = X_train_scaled.shape[1]
            
            # Define model configurations
            model_configs = {
                'simple_ann': {
                    'model_func': self.create_simple_ann,
                    'params': {'input_dim': input_dim, 'hidden_layers': [64, 32], 'dropout_rate': 0.2}
                },
                'deep_ann': {
                    'model_func': self.create_deep_ann,
                    'params': {'input_dim': input_dim, 'hidden_layers': [128, 64, 32, 16], 'dropout_rate': 0.3}
                },
                'wide_and_deep': {
                    'model_func': self.create_wide_and_deep,
                    'params': {'input_dim': input_dim, 'deep_layers': [64, 32], 'dropout_rate': 0.2}
                }
            }
            
            # Training callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
            
            callback_list = [early_stopping, reduce_lr]
            
            # Train each model
            for name, config in model_configs.items():
                logging.info(f"Training {name}")
                
                # Create model
                model = config['model_func'](**config['params'])
                
                # Train model
                history = model.fit(
                    X_train_scaled, y_train_scaled,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callback_list,
                    verbose=0
                )
                
                # Make predictions
                train_pred = model.predict(X_train_scaled, verbose=0).flatten()
                test_pred = model.predict(X_test_scaled, verbose=0).flatten()
                
                # Calculate metrics
                train_r2 = r2_score(y_train_scaled, train_pred)
                test_r2 = r2_score(y_test_scaled, test_pred)
                train_mse = mean_squared_error(y_train_scaled, train_pred)
                test_mse = mean_squared_error(y_test_scaled, test_pred)
                train_mae = mean_absolute_error(y_train_scaled, train_pred)
                test_mae = mean_absolute_error(y_test_scaled, test_pred)
                
                # Store results
                self.models[name] = model
                self.history[name] = history.history
                self.results[name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': np.sqrt(train_mse),
                    'test_rmse': np.sqrt(test_mse),
                    'epochs_trained': len(history.history['loss']),
                    'best_val_loss': min(history.history['val_loss'])
                }
                
                logging.info(f"{name} - Test R2: {test_r2:.4f}, Test RMSE: {np.sqrt(test_mse):.4f}, Epochs: {len(history.history['loss'])}")
            
            logging.info("Deep learning model training completed")
            return self.results
            
        except Exception as e:
            logging.error("Error in deep learning model training")
            raise CustomException(e, sys)
    
    def plot_training_history(self, model_name=None, save_path=None):
        """
        Plot training history for model analysis
        """
        try:
            models_to_plot = [model_name] if model_name else list(self.history.keys())
            
            fig, axes = plt.subplots(2, len(models_to_plot), figsize=(6*len(models_to_plot), 10))
            if len(models_to_plot) == 1:
                axes = axes.reshape(-1, 1)
            
            for i, name in enumerate(models_to_plot):
                history = self.history[name]
                
                # Loss plot
                axes[0, i].plot(history['loss'], label='Training Loss', alpha=0.8)
                axes[0, i].plot(history['val_loss'], label='Validation Loss', alpha=0.8)
                axes[0, i].set_title(f'{name} - Loss')
                axes[0, i].set_xlabel('Epoch')
                axes[0, i].set_ylabel('Mean Squared Error')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)
                
                # MAE plot
                axes[1, i].plot(history['mean_absolute_error'], label='Training MAE', alpha=0.8)
                axes[1, i].plot(history['val_mean_absolute_error'], label='Validation MAE', alpha=0.8)
                axes[1, i].set_title(f'{name} - Mean Absolute Error')
                axes[1, i].set_xlabel('Epoch')
                axes[1, i].set_ylabel('Mean Absolute Error')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Training history plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logging.error("Error plotting training history")
            raise CustomException(e, sys)
    
    def get_best_dl_model(self, metric='test_r2'):
        """
        Get the best performing deep learning model
        """
        if not self.results:
            raise ValueError("No models trained yet. Please run train_dl_models first.")
        
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x][metric])
        return best_model_name, self.models[best_model_name]
    
    def save_dl_models(self, save_dir='../../models/deep_learning'):
        """
        Save all trained deep learning models
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            for name, model in self.models.items():
                model_path = os.path.join(save_dir, f"{name}_model.h5")
                model.save(model_path)
                logging.info(f"Saved {name} model to {model_path}")
            
            # Save scaler
            scaler_path = os.path.join(save_dir, "feature_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save results and history
            results_path = os.path.join(save_dir, "dl_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'results': self.results,
                    'history': self.history
                }, f, indent=2)
            
            logging.info(f"Saved deep learning results to {results_path}")
            
        except Exception as e:
            logging.error("Error saving deep learning models")
            raise CustomException(e, sys)
    
    def compare_with_traditional_ml(self, traditional_results):
        """
        Compare deep learning results with traditional ML models
        """
        try:
            comparison_data = []
            
            # Add traditional ML results
            for model_name, metrics in traditional_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Type': 'Traditional ML',
                    'Test R2': metrics['test_r2'],
                    'Test RMSE': metrics['test_rmse'],
                    'Test MAE': metrics['test_mae']
                })
            
            # Add deep learning results
            for model_name, metrics in self.results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Type': 'Deep Learning',
                    'Test R2': metrics['test_r2'],
                    'Test RMSE': metrics['test_rmse'],
                    'Test MAE': metrics['test_mae']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Test R2', ascending=False)
            
            logging.info("Model comparison completed")
            return comparison_df
            
        except Exception as e:
            logging.error("Error in model comparison")
            raise CustomException(e, sys)
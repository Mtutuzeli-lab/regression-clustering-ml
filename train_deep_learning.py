"""
Deep Learning Model Training with TensorBoard

This script trains neural network models (Simple ANN, Deep ANN, Wide & Deep)
with TensorBoard integration for real-time monitoring and visualization.

TensorBoard shows:
- Training & validation loss curves
- Accuracy metrics over epochs
- Model architecture graphs
- Learning rate changes
- Weight distributions

Run TensorBoard: tensorboard --logdir=logs/tensorboard
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

sys.path.append('src')
from src.logger import logger


class DeepLearningTrainer:
    """
    Train deep learning models with TensorBoard tracking
    """
    
    def __init__(self):
        self.log_dir = os.path.join('logs', 'tensorboard', datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.model_dir = 'models/deep_learning'
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info("Deep Learning Trainer initialized")
        logger.info(f"TensorBoard logs: {self.log_dir}")
    
    def load_and_prepare_data(self):
        """Load and prepare data for deep learning"""
        logger.info("Loading data...")
        
        # Load data
        df = pd.read_csv('artifacts/raw_data.csv')
        
        # Drop non-numerical columns
        X = df.drop(columns=['Email', 'Address', 'Avatar', 'Yearly Amount Spent'])
        y = df['Yearly Amount Spent']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Data loaded: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test")
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler
    
    def create_simple_ann(self, input_dim):
        """Create Simple ANN architecture"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=input_dim, name='hidden_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu', name='hidden_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1, name='output')
        ], name='Simple_ANN')
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        return model
    
    def create_deep_ann(self, input_dim):
        """Create Deep ANN architecture"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim, name='hidden_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', name='hidden_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu', name='hidden_3'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu', name='hidden_4'),
            layers.BatchNormalization(),
            layers.Dense(1, name='output')
        ], name='Deep_ANN')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        return model
    
    def create_wide_deep(self, input_dim):
        """Create Wide & Deep architecture"""
        # Input
        input_layer = layers.Input(shape=(input_dim,), name='input')
        
        # Wide part (linear)
        wide = layers.Dense(1, name='wide')(input_layer)
        
        # Deep part
        deep = layers.Dense(64, activation='relu', name='deep_1')(input_layer)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.2)(deep)
        deep = layers.Dense(32, activation='relu', name='deep_2')(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.2)(deep)
        deep = layers.Dense(16, activation='relu', name='deep_3')(deep)
        
        # Combine wide and deep
        combined = layers.concatenate([wide, deep], name='wide_deep_concat')
        output = layers.Dense(1, name='output')(combined)
        
        model = keras.Model(inputs=input_layer, outputs=output, name='Wide_Deep')
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        return model
    
    def get_callbacks(self, model_name):
        """Get training callbacks including TensorBoard"""
        
        # TensorBoard callback
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=os.path.join(self.log_dir, model_name),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0
        )
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning rate reduction
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
        
        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            os.path.join(self.model_dir, f'{model_name}_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        return [tensorboard_callback, early_stopping, reduce_lr, checkpoint]
    
    def train_model(self, model, model_name, X_train, y_train, X_test, y_test):
        """Train a model with TensorBoard tracking"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*80}")
        
        # Get callbacks
        model_callbacks = self.get_callbacks(model_name)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=model_callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_mae, test_rmse = model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Test Loss (MSE): {test_loss:.2f}")
        logger.info(f"  Test MAE: ${test_mae:.2f}")
        logger.info(f"  Test RMSE: ${test_rmse:.2f}")
        
        return history, test_loss, test_mae, test_rmse
    
    def train_all_models(self):
        """Train all deep learning models"""
        
        logger.info("="*80)
        logger.info("DEEP LEARNING TRAINING WITH TENSORBOARD")
        logger.info("="*80)
        
        # Load data
        X_train, X_test, y_train, y_test, scaler = self.load_and_prepare_data()
        
        input_dim = X_train.shape[1]
        
        # Results storage
        results = []
        
        # Train Simple ANN
        simple_ann = self.create_simple_ann(input_dim)
        logger.info(f"\nSimple ANN Architecture:")
        simple_ann.summary(print_fn=logger.info)
        
        hist1, loss1, mae1, rmse1 = self.train_model(
            simple_ann, 'Simple_ANN', X_train, y_train, X_test, y_test
        )
        results.append({
            'Model': 'Simple_ANN',
            'Test_Loss': loss1,
            'Test_MAE': mae1,
            'Test_RMSE': rmse1
        })
        
        # Train Deep ANN
        deep_ann = self.create_deep_ann(input_dim)
        logger.info(f"\nDeep ANN Architecture:")
        deep_ann.summary(print_fn=logger.info)
        
        hist2, loss2, mae2, rmse2 = self.train_model(
            deep_ann, 'Deep_ANN', X_train, y_train, X_test, y_test
        )
        results.append({
            'Model': 'Deep_ANN',
            'Test_Loss': loss2,
            'Test_MAE': mae2,
            'Test_RMSE': rmse2
        })
        
        # Train Wide & Deep
        wide_deep = self.create_wide_deep(input_dim)
        logger.info(f"\nWide & Deep Architecture:")
        wide_deep.summary(print_fn=logger.info)
        
        hist3, loss3, mae3, rmse3 = self.train_model(
            wide_deep, 'Wide_Deep', X_train, y_train, X_test, y_test
        )
        results.append({
            'Model': 'Wide_Deep',
            'Test_Loss': loss3,
            'Test_MAE': mae3,
            'Test_RMSE': rmse3
        })
        
        # Save comparison report
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test_RMSE')
        results_df.to_csv(os.path.join(self.model_dir, 'model_comparison.csv'), index=False)
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80)
        logger.info("\nModel Comparison:")
        logger.info(f"\n{results_df.to_string()}")
        logger.info(f"\nBest Model: {results_df.iloc[0]['Model']}")
        logger.info(f"Best Test RMSE: ${results_df.iloc[0]['Test_RMSE']:.2f}")
        
        logger.info(f"\n{'='*80}")
        logger.info("VIEW TENSORBOARD:")
        logger.info(f"Run: tensorboard --logdir={self.log_dir}")
        logger.info("Then open: http://localhost:6006")
        logger.info(f"{'='*80}")
        
        return results_df


def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("DEEP LEARNING MODEL TRAINING WITH TENSORBOARD")
    print("="*80)
    
    try:
        trainer = DeepLearningTrainer()
        results = trainer.train_all_models()
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nTo view training progress in TensorBoard:")
        print(f"  tensorboard --logdir=logs/tensorboard")
        print("  Then open: http://localhost:6006")
        print("\nModels saved in: models/deep_learning/")
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

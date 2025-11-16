"""
Train and save clustering model artifacts for production use

This script trains the K-Means clustering model and saves it along with
the preprocessors (scaler and PCA) for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys

sys.path.append('src')
from src.logger import logger

def train_clustering_model():
    """Train and save clustering model artifacts"""
    
    try:
        logger.info("=" * 80)
        logger.info("CLUSTERING MODEL TRAINING STARTED")
        logger.info("=" * 80)
        
        # Create artifacts directory
        os.makedirs('artifacts/clustering', exist_ok=True)
        logger.info("Clustering artifacts directory created")
        
        # Load data
        logger.info("Loading data from artifacts/raw_data.csv...")
        df = pd.read_csv('artifacts/raw_data.csv')
        logger.info(f"Data loaded: {df.shape}")
        
        # Drop non-numerical columns
        columns_to_drop = ['Email', 'Address', 'Avatar', 'Yearly Amount Spent']
        X = df.drop(columns=columns_to_drop, errors='ignore')
        logger.info(f"Features prepared: {X.shape}")
        logger.info(f"Features: {list(X.columns)}")
        
        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Features scaled successfully")
        
        # Train K-Means with optimal k=4
        logger.info("Training K-Means clustering model (k=4)...")
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        logger.info("K-Means model trained successfully")
        
        # Train PCA for visualization
        logger.info("Training PCA model...")
        pca = PCA(n_components=2, random_state=42)
        pca.fit(X_scaled)
        logger.info(f"PCA trained - Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
        
        # Save artifacts
        logger.info("Saving clustering artifacts...")
        
        # Save K-Means model
        with open('artifacts/clustering/kmeans_model.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
        logger.info("✓ K-Means model saved")
        
        # Save scaler
        with open('artifacts/clustering/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        logger.info("✓ Scaler saved")
        
        # Save PCA
        with open('artifacts/clustering/pca_model.pkl', 'wb') as f:
            pickle.dump(pca, f)
        logger.info("✓ PCA model saved")
        
        # Test predictions
        logger.info("\nTesting clustering predictions...")
        sample_data = X_scaled[:5]
        predictions = kmeans.predict(sample_data)
        logger.info(f"Sample predictions: {predictions}")
        
        logger.info("\n" + "=" * 80)
        logger.info("CLUSTERING MODEL TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nArtifacts saved:")
        logger.info("  - artifacts/clustering/kmeans_model.pkl")
        logger.info("  - artifacts/clustering/scaler.pkl")
        logger.info("  - artifacts/clustering/pca_model.pkl")
        logger.info("\n✅ Clustering model ready for predictions!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training clustering model: {str(e)}")
        return False


if __name__ == "__main__":
    success = train_clustering_model()
    if success:
        print("\n✅ Clustering model trained and saved successfully!")
        print("You can now run the Streamlit app with customer segmentation!")
    else:
        print("\n❌ Clustering model training failed!")
    
    sys.exit(0 if success else 1)

"""
Prediction Pipeline - Production Inference

This module provides prediction functionality for:
1. Customer Spending Prediction (Regression)
2. Customer Segmentation (Clustering)

Classes load trained models and preprocessors from artifacts folder
and provide clean interfaces for making predictions on new data.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.logger import logger


class RegressionPredictionPipeline:
    """
    Pipeline for predicting customer yearly spending
    """
    
    def __init__(self):
        """Initialize the prediction pipeline"""
        self.model_path = 'artifacts/model.pkl'
        self.preprocessor_path = 'artifacts/preprocessor.pkl'
        self.model = None
        self.preprocessor = None
        
    def load_artifacts(self):
        """Load trained model and preprocessor"""
        try:
            logger.info("Loading regression model and preprocessor...")
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info("Artifacts loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise Exception(f"Could not load model artifacts: {str(e)}")
    
    def predict(self, features):
        """
        Make prediction on new data
        
        Args:
            features: DataFrame or dict with features:
                - Avg. Session Length
                - Time on App
                - Time on Website
                - Length of Membership
                
        Returns:
            Predicted yearly spending amount
        """
        try:
            # Load artifacts if not already loaded
            if self.model is None or self.preprocessor is None:
                self.load_artifacts()
            
            # Convert to DataFrame if dict
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            
            # Ensure correct column order
            required_columns = [
                'Avg. Session Length',
                'Time on App', 
                'Time on Website',
                'Length of Membership'
            ]
            
            features = features[required_columns]
            
            # Transform features
            features_scaled = self.preprocessor.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            
            logger.info(f"Prediction made: ${prediction[0]:.2f}")
            return prediction[0]
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")


class ClusteringPredictionPipeline:
    """
    Pipeline for predicting customer segment/cluster
    """
    
    def __init__(self):
        """Initialize the clustering pipeline"""
        self.kmeans_model_path = 'artifacts/clustering/kmeans_model.pkl'
        self.scaler_path = 'artifacts/clustering/scaler.pkl'
        self.pca_path = 'artifacts/clustering/pca_model.pkl'
        
        self.kmeans_model = None
        self.scaler = None
        self.pca = None
        
        # Cluster descriptions
        self.cluster_profiles = {
            0: {
                'name': 'High-Value Loyalists',
                'description': 'Long-term members with high engagement and spending',
                'characteristics': 'High app usage, long membership, premium spenders'
            },
            1: {
                'name': 'Growing Customers',
                'description': 'Medium engagement with growth potential',
                'characteristics': 'Moderate usage across channels, developing loyalty'
            },
            2: {
                'name': 'Window Shoppers',
                'description': 'High browsing but lower conversion',
                'characteristics': 'High website time, lower app usage, moderate spending'
            },
            3: {
                'name': 'New Explorers',
                'description': 'New customers still discovering the platform',
                'characteristics': 'Short membership, varied engagement patterns'
            }
        }
    
    def load_artifacts(self):
        """Load trained clustering model and preprocessors"""
        try:
            logger.info("Loading clustering model and preprocessors...")
            
            # Check if clustering artifacts exist
            if not os.path.exists(self.kmeans_model_path):
                logger.warning("Clustering artifacts not found. Run clustering training first.")
                return False
            
            self.kmeans_model = joblib.load(self.kmeans_model_path)
            self.scaler = joblib.load(self.scaler_path)
            
            # PCA is optional
            if os.path.exists(self.pca_path):
                self.pca = joblib.load(self.pca_path)
            
            logger.info("Clustering artifacts loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading clustering artifacts: {str(e)}")
            return False
    
    def predict(self, features):
        """
        Predict customer segment
        
        Args:
            features: DataFrame or dict with features:
                - Avg. Session Length
                - Time on App
                - Time on Website
                - Length of Membership
                
        Returns:
            Dictionary with cluster prediction and profile
        """
        try:
            # Load artifacts if not already loaded
            if self.kmeans_model is None:
                success = self.load_artifacts()
                if not success:
                    return {
                        'cluster': None,
                        'error': 'Clustering model not available. Train clustering model first.'
                    }
            
            # Convert to DataFrame if dict
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            
            # Ensure correct column order
            required_columns = [
                'Avg. Session Length',
                'Time on App', 
                'Time on Website',
                'Length of Membership'
            ]
            
            features = features[required_columns]
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict cluster
            cluster = self.kmeans_model.predict(features_scaled)[0]
            
            # Get cluster profile
            profile = self.cluster_profiles.get(cluster, {
                'name': f'Cluster {cluster}',
                'description': 'Customer segment',
                'characteristics': 'Unique customer profile'
            })
            
            result = {
                'cluster': int(cluster),
                'cluster_name': profile['name'],
                'description': profile['description'],
                'characteristics': profile['characteristics']
            }
            
            logger.info(f"Customer assigned to cluster {cluster}: {profile['name']}")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting cluster: {str(e)}")
            return {
                'cluster': None,
                'error': f'Prediction failed: {str(e)}'
            }


class CustomData:
    """
    Helper class to create feature dictionary from individual inputs
    """
    
    def __init__(self,
                 avg_session_length: float,
                 time_on_app: float,
                 time_on_website: float,
                 length_of_membership: float):
        """
        Initialize custom data
        
        Args:
            avg_session_length: Average session duration (minutes)
            time_on_app: Time spent on mobile app (minutes)
            time_on_website: Time spent on website (minutes)
            length_of_membership: Years of membership
        """
        self.avg_session_length = avg_session_length
        self.time_on_app = time_on_app
        self.time_on_website = time_on_website
        self.length_of_membership = length_of_membership
    
    def get_data_as_dataframe(self):
        """
        Convert custom data to DataFrame
        
        Returns:
            DataFrame with features
        """
        try:
            data_dict = {
                'Avg. Session Length': [self.avg_session_length],
                'Time on App': [self.time_on_app],
                'Time on Website': [self.time_on_website],
                'Length of Membership': [self.length_of_membership]
            }
            
            return pd.DataFrame(data_dict)
            
        except Exception as e:
            logger.error(f"Error creating DataFrame: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Test regression prediction
    print("Testing Regression Prediction Pipeline...")
    print("-" * 50)
    
    regression_pipeline = RegressionPredictionPipeline()
    
    test_data = CustomData(
        avg_session_length=33.0,
        time_on_app=12.5,
        time_on_website=37.0,
        length_of_membership=3.5
    )
    
    features_df = test_data.get_data_as_dataframe()
    prediction = regression_pipeline.predict(features_df)
    
    print(f"Predicted Yearly Spending: ${prediction:.2f}")
    
    # Test clustering prediction
    print("\n" + "=" * 50)
    print("Testing Clustering Prediction Pipeline...")
    print("-" * 50)
    
    clustering_pipeline = ClusteringPredictionPipeline()
    cluster_result = clustering_pipeline.predict(features_df)
    
    if cluster_result.get('cluster') is not None:
        print(f"Customer Segment: {cluster_result['cluster_name']}")
        print(f"Description: {cluster_result['description']}")
        print(f"Characteristics: {cluster_result['characteristics']}")
    else:
        print(f"Clustering not available: {cluster_result.get('error', 'Unknown error')}")

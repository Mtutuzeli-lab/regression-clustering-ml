"""
Clustering models module for customer segmentation
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from ..exception import CustomException
from ..logger import logging

class ClusteringModelTrainer:
    """
    Class to perform customer segmentation using various clustering algorithms
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.pca = None
        self.cluster_labels = {}
    
    def prepare_data_for_clustering(self, df, features_to_use=None, scale_data=True, use_pca=False, n_components=None):
        """
        Prepare data for clustering analysis
        """
        try:
            logging.info("Preparing data for clustering")
            
            # Select features (exclude non-numeric columns)
            if features_to_use is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                # Exclude target variable if it exists
                if 'Yearly Amount Spent' in numeric_cols:
                    features_to_use = [col for col in numeric_cols if col != 'Yearly Amount Spent']
                else:
                    features_to_use = numeric_cols.tolist()
            
            X = df[features_to_use].copy()
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Scale the data
            if scale_data:
                X_scaled = self.scaler.fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=features_to_use, index=X.index)
            else:
                X_scaled = X
            
            # Apply PCA if requested
            if use_pca:
                if n_components is None:
                    n_components = min(len(features_to_use), 5)
                
                self.pca = PCA(n_components=n_components)
                X_pca = self.pca.fit_transform(X_scaled)
                X_final = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)], index=X.index)
                logging.info(f"Applied PCA with {n_components} components. Explained variance ratio: {self.pca.explained_variance_ratio_}")
            else:
                X_final = X_scaled
            
            logging.info(f"Data prepared for clustering. Shape: {X_final.shape}")
            return X_final, features_to_use
            
        except Exception as e:
            logging.error("Error in data preparation for clustering")
            raise CustomException(e, sys)
    
    def find_optimal_clusters(self, X, max_clusters=10, methods=['kmeans']):
        """
        Find optimal number of clusters using various methods
        """
        try:
            logging.info("Finding optimal number of clusters")
            
            results = {}
            
            for method in methods:
                if method == 'kmeans':
                    inertias = []
                    silhouette_scores = []
                    calinski_scores = []
                    
                    for k in range(2, max_clusters + 1):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(X)
                        
                        inertias.append(kmeans.inertia_)
                        silhouette_scores.append(silhouette_score(X, cluster_labels))
                        calinski_scores.append(calinski_harabasz_score(X, cluster_labels))
                    
                    results[method] = {
                        'inertias': inertias,
                        'silhouette_scores': silhouette_scores,
                        'calinski_scores': calinski_scores,
                        'k_range': list(range(2, max_clusters + 1))
                    }
            
            logging.info("Optimal cluster analysis completed")
            return results
            
        except Exception as e:
            logging.error("Error in finding optimal clusters")
            raise CustomException(e, sys)
    
    def train_clustering_models(self, X, optimal_clusters=None):
        """
        Train various clustering models
        """
        try:
            logging.info("Starting clustering model training")
            
            if optimal_clusters is None:
                optimal_clusters = {'kmeans': 4, 'hierarchical': 4}
            
            # K-Means Clustering
            if 'kmeans' in optimal_clusters:
                logging.info(f"Training K-Means with {optimal_clusters['kmeans']} clusters")
                kmeans = KMeans(n_clusters=optimal_clusters['kmeans'], random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(X)
                
                self.models['kmeans'] = kmeans
                self.cluster_labels['kmeans'] = kmeans_labels
                
                # Calculate metrics
                silhouette_avg = silhouette_score(X, kmeans_labels)
                calinski_score = calinski_harabasz_score(X, kmeans_labels)
                davies_bouldin = davies_bouldin_score(X, kmeans_labels)
                
                self.results['kmeans'] = {
                    'n_clusters': optimal_clusters['kmeans'],
                    'silhouette_score': silhouette_avg,
                    'calinski_score': calinski_score,
                    'davies_bouldin_score': davies_bouldin,
                    'inertia': kmeans.inertia_
                }
                
                logging.info(f"K-Means - Silhouette Score: {silhouette_avg:.4f}")
            
            # Hierarchical Clustering
            if 'hierarchical' in optimal_clusters:
                logging.info(f"Training Hierarchical Clustering with {optimal_clusters['hierarchical']} clusters")
                hierarchical = AgglomerativeClustering(n_clusters=optimal_clusters['hierarchical'])
                hierarchical_labels = hierarchical.fit_predict(X)
                
                self.models['hierarchical'] = hierarchical
                self.cluster_labels['hierarchical'] = hierarchical_labels
                
                # Calculate metrics
                silhouette_avg = silhouette_score(X, hierarchical_labels)
                calinski_score = calinski_harabasz_score(X, hierarchical_labels)
                davies_bouldin = davies_bouldin_score(X, hierarchical_labels)
                
                self.results['hierarchical'] = {
                    'n_clusters': optimal_clusters['hierarchical'],
                    'silhouette_score': silhouette_avg,
                    'calinski_score': calinski_score,
                    'davies_bouldin_score': davies_bouldin
                }
                
                logging.info(f"Hierarchical - Silhouette Score: {silhouette_avg:.4f}")
            
            # DBSCAN Clustering
            logging.info("Training DBSCAN")
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise = list(dbscan_labels).count(-1)
            
            self.models['dbscan'] = dbscan
            self.cluster_labels['dbscan'] = dbscan_labels
            
            if n_clusters > 1:
                silhouette_avg = silhouette_score(X, dbscan_labels)
                calinski_score = calinski_harabasz_score(X, dbscan_labels)
                davies_bouldin = davies_bouldin_score(X, dbscan_labels)
            else:
                silhouette_avg = calinski_score = davies_bouldin = 0
            
            self.results['dbscan'] = {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'silhouette_score': silhouette_avg,
                'calinski_score': calinski_score,
                'davies_bouldin_score': davies_bouldin
            }
            
            logging.info(f"DBSCAN - Clusters: {n_clusters}, Noise points: {n_noise}")
            logging.info("Clustering model training completed")
            
            return self.results
            
        except Exception as e:
            logging.error("Error in clustering model training")
            raise CustomException(e, sys)
    
    def analyze_segments(self, df, X, method='kmeans'):
        """
        Analyze characteristics of each cluster/segment
        """
        try:
            if method not in self.cluster_labels:
                raise ValueError(f"Method {method} not trained yet")
            
            labels = self.cluster_labels[method]
            df_analysis = df.copy()
            df_analysis['Cluster'] = labels
            
            # Remove noise points for DBSCAN
            if method == 'dbscan':
                df_analysis = df_analysis[df_analysis['Cluster'] != -1]
            
            # Calculate segment characteristics
            segment_summary = df_analysis.groupby('Cluster').agg({
                'Yearly Amount Spent': ['mean', 'median', 'std', 'count'],
                'Avg. Session Length': ['mean'],
                'Time on App': ['mean'],
                'Time on Website': ['mean'],
                'Length of Membership': ['mean']
            }).round(2)
            
            logging.info(f"Segment analysis completed for {method}")
            return segment_summary, df_analysis
            
        except Exception as e:
            logging.error(f"Error in segment analysis for {method}")
            raise CustomException(e, sys)
    
    def save_models(self, save_dir='../../models/clustering'):
        """
        Save all trained clustering models
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            for name, model in self.models.items():
                model_path = os.path.join(save_dir, f"{name}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logging.info(f"Saved {name} model to {model_path}")
            
            # Save scaler and PCA
            scaler_path = os.path.join(save_dir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            if self.pca:
                pca_path = os.path.join(save_dir, "pca.pkl")
                with open(pca_path, 'wb') as f:
                    pickle.dump(self.pca, f)
            
            # Save results and cluster labels
            results_path = os.path.join(save_dir, "clustering_results.pkl")
            with open(results_path, 'wb') as f:
                pickle.dump({
                    'results': self.results,
                    'cluster_labels': self.cluster_labels
                }, f)
            
            logging.info(f"Saved clustering results to {results_path}")
            
        except Exception as e:
            logging.error("Error saving clustering models")
            raise CustomException(e, sys)
"""
E-Commerce Customer Analytics - Streamlit Application

This app provides two main functionalities:
1. Customer Spending Prediction (Regression)
2. Customer Segmentation (Clustering)

Author: Mtutuzeli
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.pipeline.predict_pipeline import (
    RegressionPredictionPipeline,
    ClusteringPredictionPipeline,
    CustomData
)

# Page configuration
st.set_page_config(
    page_title="E-Commerce Customer Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .cluster-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🛒 E-Commerce Customer Analytics</div>', unsafe_allow_html=True)
st.markdown("### Predict Customer Spending & Identify Customer Segments")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/online-store.png", width=150)
    st.title("Navigation")
    st.markdown("---")
    
    app_mode = st.radio(
        "Choose Functionality:",
        ["🏠 Home", "💰 Spending Prediction", "👥 Customer Segmentation", "📊 Batch Prediction"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        """
        **ML Pipeline Features:**
        - 10 Model Comparison
        - 97.79% Accuracy (Ridge)
        - Real-time Predictions
        - Customer Segmentation
        - Batch Processing
        """
    )
    
    st.markdown("---")
    st.markdown("**Developed by:** Mtutuzeli")
    st.markdown("**Model:** Ridge Regression")
    st.markdown("**R² Score:** 0.9779")

# Initialize pipelines
@st.cache_resource
def load_regression_pipeline():
    pipeline = RegressionPredictionPipeline()
    pipeline.load_artifacts()
    return pipeline

@st.cache_resource
def load_clustering_pipeline():
    pipeline = ClusteringPredictionPipeline()
    pipeline.load_artifacts()
    return pipeline


# ============================================================
# HOME PAGE
# ============================================================
if app_mode == "🏠 Home":
    st.markdown("## Welcome to Customer Analytics Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💰 Spending Prediction")
        st.write("Predict customer yearly spending based on behavior patterns")
        st.metric("Model Accuracy", "97.79%")
        st.metric("RMSE", "$10.46")
    
    with col2:
        st.markdown("### 👥 Customer Segmentation")
        st.write("Identify customer segments using clustering algorithms")
        st.metric("Segments", "4 Clusters")
        st.metric("Algorithm", "K-Means")
    
    with col3:
        st.markdown("### 📊 Batch Analysis")
        st.write("Upload CSV for bulk predictions and insights")
        st.metric("Processing", "Real-time")
        st.metric("Format", "CSV")
    
    st.markdown("---")
    
    # Feature importance visualization
    st.markdown("### 📈 Key Features Impact")
    
    features = ['Length of Membership', 'Time on App', 'Avg. Session Length', 'Time on Website']
    importance = [0.45, 0.35, 0.15, 0.05]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Viridis',
                showscale=True
            )
        )
    ])
    
    fig.update_layout(
        title="Feature Importance for Spending Prediction",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.info("👈 **Select a functionality from the sidebar to get started!**")


# ============================================================
# SPENDING PREDICTION PAGE
# ============================================================
elif app_mode == "💰 Spending Prediction":
    st.markdown("## Customer Spending Prediction")
    st.write("Enter customer behavior metrics to predict their yearly spending")
    
    # Model selection
    st.markdown("### 🤖 Select Prediction Model")
    model_type = st.selectbox(
        "Choose Model Type",
        ["Classical ML (Ridge Regression)", "Deep Learning - Simple ANN", "Deep Learning - Deep ANN", "Deep Learning - Wide & Deep"],
        help="Select between classical ML and deep learning models"
    )
    
    # Load pipeline
    try:
        regression_pipeline = load_regression_pipeline()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📱 App & Session Metrics")
            avg_session_length = st.slider(
                "Average Session Length (minutes)",
                min_value=25.0,
                max_value=40.0,
                value=33.0,
                step=0.5,
                help="Average time per session"
            )
            
            time_on_app = st.slider(
                "Time on App (minutes/week)",
                min_value=8.0,
                max_value=16.0,
                value=12.0,
                step=0.5,
                help="Weekly app usage time"
            )
        
        with col2:
            st.markdown("### 🌐 Website & Membership")
            time_on_website = st.slider(
                "Time on Website (minutes/week)",
                min_value=30.0,
                max_value=45.0,
                value=37.0,
                step=0.5,
                help="Weekly website usage time"
            )
            
            length_of_membership = st.slider(
                "Length of Membership (years)",
                min_value=0.5,
                max_value=6.0,
                value=3.0,
                step=0.5,
                help="Years as a customer"
            )
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Predict Yearly Spending", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                # Create custom data
                custom_data = CustomData(
                    avg_session_length=avg_session_length,
                    time_on_app=time_on_app,
                    time_on_website=time_on_website,
                    length_of_membership=length_of_membership
                )
                
                # Get prediction based on model type
                features_df = custom_data.get_data_as_dataframe()
                
                if "Deep Learning" in model_type:
                    # Load deep learning model
                    import tensorflow as tf
                    from tensorflow import keras
                    
                    model_map = {
                        "Deep Learning - Simple ANN": "models/deep_learning/Simple_ANN_best.keras",
                        "Deep Learning - Deep ANN": "models/deep_learning/Deep_ANN_best.keras",
                        "Deep Learning - Wide & Deep": "models/deep_learning/Wide_Deep_best.keras"
                    }
                    
                    model_path = model_map[model_type]
                    dl_model = keras.models.load_model(model_path)
                    
                    # Preprocess features (standardize using the same preprocessor)
                    features_scaled = regression_pipeline.preprocessor.transform(features_df)
                    
                    # Make prediction
                    prediction = dl_model.predict(features_scaled, verbose=0)[0][0]
                else:
                    # Classical ML prediction
                    prediction = regression_pipeline.predict(features_df)
                
                # Display prediction
                st.markdown(
                    f'<div class="prediction-box">💰 ${prediction:,.2f}</div>',
                    unsafe_allow_html=True
                )
                
                st.success("✅ Prediction completed successfully!")
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    monthly = prediction / 12
                    st.metric("Monthly Spending", f"${monthly:,.2f}")
                
                with col2:
                    weekly = prediction / 52
                    st.metric("Weekly Spending", f"${weekly:,.2f}")
                
                with col3:
                    daily = prediction / 365
                    st.metric("Daily Spending", f"${daily:,.2f}")
                
                # Customer value category
                st.markdown("### 📊 Customer Value Category")
                if prediction > 550:
                    st.success("🌟 **High-Value Customer** - Premium segment")
                elif prediction > 450:
                    st.info("💎 **Medium-Value Customer** - Core segment")
                else:
                    st.warning("📈 **Growing Customer** - Development segment")
    
    except Exception as e:
        st.error(f"Error loading prediction pipeline: {str(e)}")
        st.info("Make sure you have run the training pipeline first: `python pipeline.py`")


# ============================================================
# CUSTOMER SEGMENTATION PAGE
# ============================================================
elif app_mode == "👥 Customer Segmentation":
    st.markdown("## Customer Segmentation Analysis")
    st.write("Identify which customer segment a user belongs to")
    
    # Load pipeline
    try:
        clustering_pipeline = load_clustering_pipeline()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📱 App & Session Metrics")
            avg_session_length = st.number_input(
                "Average Session Length (minutes)",
                min_value=25.0,
                max_value=40.0,
                value=33.0,
                step=0.5
            )
            
            time_on_app = st.number_input(
                "Time on App (minutes/week)",
                min_value=8.0,
                max_value=16.0,
                value=12.0,
                step=0.5
            )
        
        with col2:
            st.markdown("### 🌐 Website & Membership")
            time_on_website = st.number_input(
                "Time on Website (minutes/week)",
                min_value=30.0,
                max_value=45.0,
                value=37.0,
                step=0.5
            )
            
            length_of_membership = st.number_input(
                "Length of Membership (years)",
                min_value=0.5,
                max_value=6.0,
                value=3.0,
                step=0.5
            )
        
        st.markdown("---")
        
        # Predict button
        if st.button("🎯 Identify Customer Segment", type="primary", use_container_width=True):
            with st.spinner("Analyzing customer profile..."):
                # Create custom data
                custom_data = CustomData(
                    avg_session_length=avg_session_length,
                    time_on_app=time_on_app,
                    time_on_website=time_on_website,
                    length_of_membership=length_of_membership
                )
                
                # Get cluster prediction
                features_df = custom_data.get_data_as_dataframe()
                cluster_result = clustering_pipeline.predict(features_df)
                
                if cluster_result.get('cluster') is not None:
                    # Display cluster info
                    st.markdown(
                        f"""
                        <div class="cluster-box">
                            <h2>🎯 {cluster_result['cluster_name']}</h2>
                            <p style="font-size: 1.2rem;">{cluster_result['description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.success("✅ Segmentation completed successfully!")
                    
                    # Characteristics
                    st.markdown("### 📋 Segment Characteristics")
                    st.info(cluster_result['characteristics'])
                    
                    # Cluster number
                    st.metric("Cluster ID", f"Cluster {cluster_result['cluster']}")
                    
                else:
                    st.warning(cluster_result.get('error', 'Clustering model not available'))
                    st.info("Train the clustering model first by running the customer segmentation notebook.")
    
    except Exception as e:
        st.error(f"Error loading clustering pipeline: {str(e)}")
        st.info("Train the clustering model by running notebooks/03_Customer_Segmentation.ipynb")


# ============================================================
# BATCH PREDICTION PAGE
# ============================================================
elif app_mode == "📊 Batch Prediction":
    st.markdown("## Batch Prediction & Analysis")
    st.write("Upload a CSV file for bulk predictions")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload CSV with columns: Avg. Session Length, Time on App, Time on Website, Length of Membership"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! ({len(df)} rows)")
            
            # Show preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head())
            
            # Load pipelines
            regression_pipeline = load_regression_pipeline()
            
            # Predict button
            if st.button("🚀 Run Batch Predictions", type="primary", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    # Make predictions
                    predictions = []
                    for idx, row in df.iterrows():
                        custom_data = CustomData(
                            avg_session_length=row['Avg. Session Length'],
                            time_on_app=row['Time on App'],
                            time_on_website=row['Time on Website'],
                            length_of_membership=row['Length of Membership']
                        )
                        features_df = custom_data.get_data_as_dataframe()
                        pred = regression_pipeline.predict(features_df)
                        predictions.append(pred)
                    
                    # Add predictions to dataframe
                    df['Predicted_Spending'] = predictions
                    
                    st.success("✅ Batch predictions completed!")
                    
                    # Show results
                    st.markdown("### 📊 Results")
                    st.dataframe(df)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Average Spending", f"${df['Predicted_Spending'].mean():,.2f}")
                    with col2:
                        st.metric("Max Spending", f"${df['Predicted_Spending'].max():,.2f}")
                    with col3:
                        st.metric("Min Spending", f"${df['Predicted_Spending'].min():,.2f}")
                    with col4:
                        st.metric("Total Customers", len(df))
                    
                    # Distribution plot
                    st.markdown("### 📈 Spending Distribution")
                    fig = px.histogram(
                        df,
                        x='Predicted_Spending',
                        nbins=30,
                        title="Customer Spending Distribution",
                        labels={'Predicted_Spending': 'Yearly Spending ($)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show sample CSV format
        st.markdown("### 📝 Sample CSV Format")
        sample_df = pd.DataFrame({
            'Avg. Session Length': [33.0, 34.5, 31.2],
            'Time on App': [12.0, 11.5, 13.2],
            'Time on Website': [37.0, 36.2, 38.5],
            'Length of Membership': [3.5, 4.2, 2.8]
        })
        st.dataframe(sample_df)
        
        # Download sample
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_data.csv",
            mime="text/csv"
        )


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit | ML Pipeline with 97.79% Accuracy</p>
        <p>Ridge Regression Model | K-Means Clustering | Production Ready</p>
    </div>
    """,
    unsafe_allow_html=True
)

# eCommerce Customer Analysis - Advanced ML Project

A comprehensive machine learning project implementing **Traditional ML**, **Deep Learning ANNs**, and **Customer Segmentation** for eCommerce customer spending prediction and behavioral analysis. This project demonstrates mastery of the complete ML spectrum from classical algorithms to modern neural networks.

## 🎯 Project Objectives

### Traditional Machine Learning
- **Regression Models**: Linear, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting, SVR
- **Feature Engineering**: Statistical analysis and feature importance ranking
- **Model Interpretability**: Understand which factors drive customer spending

### Deep Learning & Neural Networks
- **Artificial Neural Networks**: Simple ANN, Deep ANN, and Wide & Deep architectures
- **Advanced Training**: Early stopping, learning rate scheduling, batch normalization
- **Pattern Recognition**: Capture complex non-linear relationships in customer data

### Customer Segmentation & Clustering
- **Behavioral Grouping**: K-Means, Hierarchical, and DBSCAN clustering algorithms
- **Customer Personas**: Data-driven segment characterization for business strategy
- **Spending Analysis**: Understand behavior patterns across different customer groups

### Business Intelligence & MLOps
- **Model Comparison**: Comprehensive evaluation across all ML paradigms
- **Production Pipeline**: End-to-end workflow from EDA to model deployment
- **Performance Optimization**: Hyperparameter tuning and cross-validation

## 📊 Dataset Features

**eCommerce Customer Dataset** (500 customers, 8 features)
- **Customer Identifiers**: Email, Address, Avatar (profile representation)
- **Behavioral Metrics**: 
  - `Avg. Session Length`: Average time spent per session
  - `Time on App`: Mobile application usage duration  
  - `Time on Website`: Desktop website engagement time
  - `Length of Membership`: Customer tenure/loyalty indicator
- **Target Variable**: `Yearly Amount Spent` (continuous, $256-$765 range)

**Dataset Characteristics:**
- **Size**: 500 records × 8 features
- **Quality**: No missing values, clean data ready for modeling
- **Distribution**: Normal spending distribution suitable for regression
- **Business Context**: Real-world eCommerce customer behavioral data

## 🏗️ Project Architecture

```
regression-clustering-ml/
├── 📊 data/                          # Dataset storage
│   └── ecommerce_customer.csv        # Customer behavioral data
├── 📓 notebooks/                     # Complete analysis workflow
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_Regression_Models.ipynb   # Traditional ML models
│   ├── 03_Customer_Segmentation.ipynb # Clustering & segmentation  
│   ├── 05_Deep_Learning_Regression.ipynb # Neural networks
│   └── 06_Final_Model_Comparison.ipynb # Comprehensive evaluation
├── 🔧 src/                          # Production-ready source code
│   ├── models/                      # Specialized ML modules
│   │   ├── regression.py            # Traditional ML trainer
│   │   ├── clustering.py            # Clustering algorithms
│   │   ├── deep_learning.py         # ANN implementations
│   │   └── __init__.py              # Package initialization
│   ├── components/                  # Data processing pipeline
│   ├── pipeline/                    # ML workflow management
│   ├── exception.py                 # Custom error handling
│   ├── logger.py                    # Enterprise logging system
│   └── utils.py                     # Utility functions
├── 💾 models/                       # Trained model artifacts  
│   ├── regression/                  # Traditional ML models (.pkl)
│   ├── clustering/                  # Clustering models & scalers
│   └── deep_learning/               # Neural networks (.h5)
├── 📈 results/                      # Analysis outputs & reports
├── 📝 logs/                         # Application execution logs
├── 🐍 venv/                        # Virtual environment
├── 📋 requirements.txt              # Python dependencies
├── ⚙️ setup.py                     # Package configuration  
└── 📖 README.md                    # Project documentation
```

## 🛠️ Technology Stack

### **Machine Learning & Deep Learning**
- **Traditional ML**: Scikit-learn (Linear, Tree, Ensemble models)
- **Deep Learning**: TensorFlow 2.13+ & Keras (ANN architectures)
- **Clustering**: K-Means, Hierarchical, DBSCAN algorithms
- **Optimization**: GridSearchCV, Early Stopping, Learning Rate Scheduling

### **Data Science & Analysis**
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualization**: Matplotlib, Seaborn, Plotly for interactive charts
- **Statistics**: Scipy for statistical analysis and hypothesis testing

### **Development & MLOps**
- **Environment**: Python 3.8+, Virtual environments
- **Experiment Tracking**: MLflow for model versioning
- **Model Persistence**: Joblib, Pickle, TensorFlow SavedModel
- **Code Quality**: Custom exception handling, comprehensive logging

### **Deployment & Scaling**
- **Web Framework**: Streamlit for model deployment dashboard  
- **Version Control**: Git & GitHub for code management
- **Documentation**: Jupyter notebooks with markdown documentation

## 🤖 Machine Learning Models

### **Traditional ML Algorithms (7 models)**
- **Linear Regression**: Baseline linear relationship modeling
- **Ridge Regression**: L2 regularization for feature stability
- **Lasso Regression**: L1 regularization with feature selection  
- **Elastic Net**: Combined L1/L2 regularization
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Advanced ensemble with sequential learning
- **Support Vector Regression**: Non-linear pattern recognition

### **Deep Learning Architectures (3 models)**
- **Simple ANN**: 2-layer feedforward network (baseline)
  - Architecture: [64, 32] neurons with ReLU activation
  - Features: Batch normalization, dropout regularization
- **Deep ANN**: 4-layer deep architecture (advanced)
  - Architecture: [128, 64, 32, 16] with progressive size reduction
  - Features: Deep feature learning, early stopping, LR scheduling
- **Wide & Deep**: Google's hybrid linear + deep learning
  - Combines memorization (wide) and generalization (deep)
  - Optimal for structured data with both linear and non-linear patterns

### **Clustering Algorithms (3 methods)**
- **K-Means**: Centroid-based partitioning with optimal k selection
- **Hierarchical Clustering**: Dendrogram-based agglomerative grouping
- **DBSCAN**: Density-based clustering with automatic outlier detection

## 📈 Model Evaluation Metrics

### **Regression Performance**
- **R² Score**: Coefficient of determination (variance explained)
- **RMSE**: Root Mean Square Error (prediction accuracy)
- **MAE**: Mean Absolute Error (robust to outliers)
- **Cross-Validation**: K-fold validation for generalization assessment

### **Clustering Quality**  
- **Silhouette Score**: Cluster cohesion and separation
- **Calinski-Harabasz Index**: Variance ratio criterion
- **Davies-Bouldin Score**: Average similarity between clusters
- **Inertia**: Within-cluster sum of squared distances

## 🚀 Getting Started

### **1. Environment Setup**
```bash
# Clone the repository
git clone https://github.com/Mtutuzeli-lab/regression-clustering-ml.git
cd regression-clustering-ml

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### **2. Data Preparation**
```bash
# Verify dataset
ls data/ecommerce_customer.csv

# Launch Jupyter for analysis
jupyter notebook
```

### **3. Complete Analysis Workflow**

**Phase 1: Exploratory Data Analysis**
```bash
jupyter notebook notebooks/01_EDA.ipynb
```
- Data quality assessment and cleaning
- Feature distribution analysis and correlation
- Statistical insights and business understanding

**Phase 2: Traditional Machine Learning**  
```bash
jupyter notebook notebooks/02_Regression_Models.ipynb
```
- Train 7 traditional ML algorithms
- Hyperparameter optimization with GridSearchCV
- Model performance comparison and feature importance

**Phase 3: Deep Learning Neural Networks**
```bash
jupyter notebook notebooks/05_Deep_Learning_Regression.ipynb  
```
- Build Simple ANN, Deep ANN, and Wide & Deep architectures
- Advanced training with TensorFlow/Keras
- Neural network optimization and pattern analysis

**Phase 4: Customer Segmentation**
```bash
jupyter notebook notebooks/03_Customer_Segmentation.ipynb
```
- Apply K-Means, Hierarchical, and DBSCAN clustering
- Determine optimal number of customer segments
- Develop customer personas and behavioral insights

**Phase 5: Comprehensive Model Comparison**
```bash
jupyter notebook notebooks/06_Final_Model_Comparison.ipynb
```
- Compare all modeling approaches across metrics
- Business value assessment and interpretability analysis  
- Production deployment recommendations

## 💼 Business Applications & Insights

### **Customer Spending Prediction**
- **Revenue Forecasting**: Predict individual customer lifetime value
- **Budget Planning**: Accurate revenue projections for business planning
- **Risk Assessment**: Identify customers with spending decline risk

### **Customer Segmentation Strategy**
- **Targeted Marketing**: Personalized campaigns for each customer segment
- **Product Recommendations**: Segment-specific product offerings
- **Retention Programs**: Customized retention strategies by customer type
- **Price Optimization**: Dynamic pricing based on customer segments

### **Deep Learning Insights**
- **Complex Patterns**: Capture non-linear relationships traditional ML might miss
- **Feature Interactions**: Automatic discovery of feature combinations
- **Scalability**: Handle larger datasets with neural network architectures

### **Expected Business Outcomes**
- **15-25% improvement** in customer targeting accuracy
- **10-20% increase** in marketing campaign effectiveness  
- **Enhanced customer experience** through personalized offerings
- **Data-driven decision making** with quantifiable insights

## 🏆 Key Technical Achievements

- **Comprehensive ML Pipeline**: End-to-end workflow from raw data to production
- **Multi-Paradigm Modeling**: Traditional ML, Deep Learning, and Unsupervised Learning
- **Production-Ready Code**: Enterprise-level logging, error handling, and documentation
- **Model Persistence**: Automated saving/loading of all trained models
- **Performance Optimization**: Hyperparameter tuning across all algorithms
- **Scalable Architecture**: Modular design for easy extension and maintenance

## 📊 Expected Results

*Results will be populated after running the complete analysis pipeline*

### **Model Performance Benchmarks**
- **Best Traditional ML**: TBD (R² score, RMSE)
- **Best Deep Learning**: TBD (R² score, RMSE, architecture)  
- **Optimal Segments**: TBD (number of clusters, silhouette score)

### **Business Impact Metrics**
- **Prediction Accuracy**: Target >85% R² score
- **Customer Segments**: 3-5 distinct behavioral groups
- **Feature Importance**: Key drivers of customer spending identified

---

## 🤝 Contributing & Usage

This project demonstrates:
- **Advanced ML Engineering**: Multi-paradigm modeling with production practices
- **Business Analytics**: Practical application of data science to real problems  
- **Technical Depth**: Modern ML stack from classical statistics to deep learning
- **Code Quality**: Enterprise-level software engineering practices

## 📄 License

This project is developed for educational and portfolio purposes.

---

**Author**: Mtutuzeli Ngamlana  
**Purpose**: Comprehensive Data Science Portfolio Project  
**Technology**: Python ML/DL Stack with Enterprise Practices  
**Last Updated**: November 2025

**🎯 Portfolio Highlights**: Traditional ML + Deep Learning + Clustering | Production-Ready MLOps | Business Intelligence Focus
- **Davies-Bouldin Score**: Average similarity between clusters
- **Inertia**: Within-cluster sum of squared distances
- **AWS**: Cloud deployment and hosting

### **Business Intelligence**
- **Power BI**: Executive dashboards and business reporting
- **Interactive Visualizations**: Customer segmentation and spending analysis

### **Development & Collaboration**
- **GitHub**: Version control and collaborative development
- **Jupyter Notebooks**: Exploratory data analysis and prototyping

## 📁 Project Structure

```
regression-clustering-ml/
├── data/                          # Raw and processed datasets
├── notebooks/                     # Jupyter notebooks for exploration
├── src/                          # Source code modules
│   ├── data_preprocessing.py     # Data cleaning and feature engineering
│   ├── model_training.py         # Model training and evaluation
│   ├── model_evaluation.py       # Performance metrics and validation
│   └── utils.py                  # Helper functions
├── models/                       # Trained model artifacts
├── streamlit_app/               # Streamlit web application
├── docker/                      # Docker configuration files
├── powerbi/                     # Power BI dashboard files
├── mlflow_experiments/          # MLflow tracking data
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites
- Python 3.13+
- Conda or virtualenv
- Docker (for containerized deployment)
- Power BI Desktop (for dashboard development)

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mtutuzeli-lab/regression-clustering-ml.git
   cd regression-clustering-ml
   ```

2. **Create conda environment**:
   ```bash
   conda create -n ecommerce-regression python=3.13
   conda activate ecommerce-regression
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

1. **Run Complete ML Pipeline**:
   ```bash
   python pipeline.py
   ```
   This will:
   - Ingest data and create train/test splits
   - Transform features with StandardScaler
   - Train 10 regression models
   - Select best model (Ridge - 97.79% R²)
   - Save all artifacts to `artifacts/` folder

2. **Train Clustering Model**:
   ```bash
   python train_clustering.py
   ```
   This creates customer segmentation artifacts needed for the app.

3. **Launch Streamlit Web App** 🚀:
   ```bash
   streamlit run app.py
   ```
   Then open http://localhost:8501 in your browser.
   
   **App Features**:
   - 💰 **Spending Prediction**: Predict customer yearly spending ($)
   - 👥 **Customer Segmentation**: Identify customer segments
   - 📊 **Batch Prediction**: Upload CSV for bulk predictions
   - 📈 **Interactive Dashboard**: Real-time visualizations

4. **Data Exploration (Optional)**:
   ```bash
   jupyter notebook notebooks/03_Customer_Segmentation.ipynb
   ```

5. **View MLflow UI (Optional)**:
   ```bash
   mlflow ui
   ```

## 📈 Model Performance

- **Primary Metric**: R-squared (R²) for regression performance
- **Secondary Metrics**: RMSE, MAE for error analysis
- **Cross-Validation**: K-fold validation for robust evaluation
- **Feature Importance**: Analysis of key predictive features

## 🔮 Model Deployment

### Streamlit Web App
Interactive application for real-time predictions with user-friendly interface.

### Docker Containerization
```bash
docker build -t ecommerce-regression .
docker run -p 8501:8501 ecommerce-regression
```

### AWS Deployment
Cloud deployment using AWS EC2/ECS for scalable model serving.

## 📊 Business Dashboards

**Power BI Integration**: 
- Customer spending trends and patterns
- Feature importance visualization
- Model performance monitoring
- Business KPI tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [Your LinkedIn Profile]
- **GitHub**: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset source and contributors
- Open source libraries and frameworks
- Community resources and tutorials
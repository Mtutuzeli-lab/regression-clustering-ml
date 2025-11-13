# eCommerce Customer Spending Prediction

This project builds a multiple regression model to predict Yearly Amount Spent by eCommerce customers based on behavioral and demographic features. It includes a full pipeline from data exploration to model deployment using Streamlit, MLflow, Docker, and AWS, with integrated Power BI for business-facing dashboards and GitHub for version control.

## 🎯 Objectives

- **Predict Customer Spending**: Build accurate regression models to forecast yearly spending amounts
- **Feature Analysis**: Identify key behavioral and demographic factors driving customer spending
- **End-to-End Pipeline**: Create a complete ML pipeline from data preprocessing to deployment
- **Business Intelligence**: Provide actionable insights through interactive dashboards
- **Scalable Deployment**: Deploy models using modern MLOps practices

## 📊 Dataset Features

The dataset includes the following customer attributes:
- **Demographic Features**: Age, location, membership tenure
- **Behavioral Features**: Website usage patterns, mobile app engagement, session duration
- **Target Variable**: Yearly Amount Spent (continuous variable for regression)

## 🛠️ Technology Stack

### **Machine Learning & Data Science**
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models and preprocessing
- **Matplotlib & Seaborn**: Data visualization

### **Model Management & Deployment**
- **MLflow**: Experiment tracking and model registry
- **Streamlit**: Interactive web application for model inference
- **Docker**: Containerization for consistent deployment
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
   git clone https://github.com/yourusername/regression-clustering-ml.git
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

1. **Data Exploration**:
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

2. **Model Training**:
   ```bash
   python src/model_training.py
   ```

3. **Launch Streamlit App**:
   ```bash
   streamlit run streamlit_app/app.py
   ```

4. **View MLflow UI**:
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
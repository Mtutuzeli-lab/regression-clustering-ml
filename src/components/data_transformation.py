import os
import sys
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Add parent directory to path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exception import CustomException
from logger import logger


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation component
    """
    # Artifacts paths
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
    # Feature configuration
    target_column: str = 'Yearly Amount Spent'
    drop_columns: list = None  # Columns to drop (Email, Address, Avatar)
    
    def __post_init__(self):
        """Initialize drop_columns after dataclass creation"""
        if self.drop_columns is None:
            self.drop_columns = ['Email', 'Address', 'Avatar']


class DataTransformation:
    """
    Handles data transformation and preprocessing:
    1. Load train/test data from artifacts
    2. Drop unnecessary columns
    3. Separate features and target
    4. Scale numerical features
    5. Save preprocessor for future use
    
    Features:
    - StandardScaler for numerical features
    - Column dropping for non-predictive features
    - Preprocessor saved as artifact
    - Comprehensive logging
    """
    
    def __init__(self, config: Optional[DataTransformationConfig] = None):
        """
        Initialize DataTransformation component
        
        Args:
            config: DataTransformationConfig object with paths and settings
        """
        self.config = config if config else DataTransformationConfig()
        logger.info("DataTransformation component initialized")
    
    def load_data(self, train_path: Optional[str] = None, 
                  test_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test data from artifacts
        
        Args:
            train_path: Path to training data. If None, uses config default
            test_path: Path to test data. If None, uses config default
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        try:
            train_data_path = train_path if train_path else self.config.train_data_path
            test_data_path = test_path if test_path else self.config.test_data_path
            
            logger.info(f"Loading training data from: {train_data_path}")
            train_df = pd.read_csv(train_data_path)
            
            logger.info(f"Loading test data from: {test_data_path}")
            test_df = pd.read_csv(test_data_path)
            
            logger.info(f"Train data loaded: {train_df.shape}")
            logger.info(f"Test data loaded: {test_df.shape}")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise CustomException(e, sys)
    
    def get_preprocessor(self, numerical_features: list) -> ColumnTransformer:
        """
        Create preprocessing pipeline for numerical features
        
        Args:
            numerical_features: List of numerical column names
            
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        try:
            logger.info(f"Creating preprocessor for {len(numerical_features)} numerical features")
            
            # Create pipeline for numerical features
            numerical_pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )
            
            # Combine into ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_features)
                ]
            )
            
            logger.info("Preprocessor created successfully")
            return preprocessor
            
        except Exception as e:
            logger.error(f"Error creating preprocessor: {str(e)}")
            raise CustomException(e, sys)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Prepare features by dropping unnecessary columns and separating target
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple: (features_df, target_series, numerical_columns)
        """
        try:
            logger.info("Preparing features...")
            
            # Drop columns that are not useful for prediction
            logger.info(f"Dropping columns: {self.config.drop_columns}")
            df_processed = df.drop(columns=self.config.drop_columns, errors='ignore')
            
            # Separate features and target
            if self.config.target_column not in df_processed.columns:
                raise ValueError(f"Target column '{self.config.target_column}' not found in data")
            
            X = df_processed.drop(columns=[self.config.target_column])
            y = df_processed[self.config.target_column]
            
            # Get numerical columns
            numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            logger.info(f"Features prepared - Shape: {X.shape}")
            logger.info(f"Target prepared - Shape: {y.shape}")
            logger.info(f"Numerical features: {numerical_columns}")
            
            return X, y, numerical_columns
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise CustomException(e, sys)
    
    def save_preprocessor(self, preprocessor: ColumnTransformer) -> str:
        """
        Save preprocessor as artifact
        
        Args:
            preprocessor: Fitted preprocessor object
            
        Returns:
            str: Path where preprocessor was saved
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
            
            # Save preprocessor
            joblib.dump(preprocessor, self.config.preprocessor_path)
            
            logger.info(f"Preprocessor saved to: {self.config.preprocessor_path}")
            return self.config.preprocessor_path
            
        except Exception as e:
            logger.error(f"Error saving preprocessor: {str(e)}")
            raise CustomException(e, sys)
    
    def load_preprocessor(self, preprocessor_path: Optional[str] = None) -> ColumnTransformer:
        """
        Load saved preprocessor artifact
        
        Args:
            preprocessor_path: Path to saved preprocessor. If None, uses config default
            
        Returns:
            ColumnTransformer: Loaded preprocessor
        """
        try:
            path = preprocessor_path if preprocessor_path else self.config.preprocessor_path
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Preprocessor not found at: {path}")
            
            preprocessor = joblib.load(path)
            logger.info(f"Preprocessor loaded from: {path}")
            
            return preprocessor
            
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Main method to initiate data transformation pipeline
        
        Returns:
            Tuple: (X_train_scaled, X_test_scaled, y_train, y_test, preprocessor_path)
        """
        try:
            logger.info("=" * 80)
            logger.info("DATA TRANSFORMATION STARTED")
            logger.info("=" * 80)
            
            # Step 1: Load train and test data from artifacts
            train_df, test_df = self.load_data()
            
            # Step 2: Prepare features and target for train set
            X_train, y_train, numerical_columns = self.prepare_features(train_df)
            
            # Step 3: Prepare features and target for test set
            X_test, y_test, _ = self.prepare_features(test_df)
            
            # Step 4: Create preprocessor
            preprocessor = self.get_preprocessor(numerical_columns)
            
            # Step 5: Fit preprocessor on training data and transform both sets
            logger.info("Fitting preprocessor on training data...")
            X_train_scaled = preprocessor.fit_transform(X_train)
            
            logger.info("Transforming test data...")
            X_test_scaled = preprocessor.transform(X_test)
            
            logger.info(f"Training data transformed: {X_train_scaled.shape}")
            logger.info(f"Test data transformed: {X_test_scaled.shape}")
            
            # Step 6: Save preprocessor as artifact
            preprocessor_path = self.save_preprocessor(preprocessor)
            
            # Step 7: Convert target to numpy arrays
            y_train = y_train.values
            y_test = y_test.values
            
            logger.info("=" * 80)
            logger.info("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
            logger.info(f"Preprocessor saved: {preprocessor_path}")
            logger.info(f"X_train shape: {X_train_scaled.shape}")
            logger.info(f"X_test shape: {X_test_scaled.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"y_test shape: {y_test.shape}")
            logger.info("=" * 80)
            
            return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor_path
            
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise CustomException(e, sys)


# Example usage and testing
if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print("TESTING DATA TRANSFORMATION COMPONENT")
        print("=" * 80)
        
        # Initialize data transformation
        data_transformation = DataTransformation()
        
        # Run transformation pipeline
        print("\n[TEST] Running data transformation pipeline...")
        X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation()
        
        print(f"\nTransformation successful!")
        print(f"\nResults:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        print(f"\nPreprocessor saved at: {preprocessor_path}")
        
        # Test loading preprocessor
        print("\n[TEST] Loading saved preprocessor...")
        loaded_preprocessor = data_transformation.load_preprocessor()
        print("Preprocessor loaded successfully!")
        
        # Show sample of transformed data
        print(f"\nSample of transformed training data (first 3 samples):")
        print(X_train[:3])
        
        print(f"\nSample of target values (first 5):")
        print(y_train[:5])
        
        # Show feature statistics after scaling
        print(f"\nFeature statistics after scaling:")
        print(f"  Mean: {np.mean(X_train, axis=0).round(4)}")
        print(f"  Std: {np.std(X_train, axis=0).round(4)}")
        print(f"  (Should be close to 0 and 1 respectively)")
        
        print("\n" + "=" * 80)
        print("DATA TRANSFORMATION TESTING COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

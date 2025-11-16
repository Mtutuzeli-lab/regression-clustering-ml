import os
import sys
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from pathlib import Path

# Add parent directory to path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exception import CustomException
from logger import logger


@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion component
    """
    # Data source paths
    raw_data_path: str = os.path.join('data', 'ecommerce_customer.csv')
    
    # Artifacts folder - stores pipeline outputs
    artifacts_dir: str = 'artifacts'
    raw_data_artifact: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    
    # Train-test split parameters
    test_size: float = 0.2
    random_state: int = 42
    
    # MongoDB configuration
    mongodb_url: Optional[str] = None
    mongodb_database: Optional[str] = None
    mongodb_collection: Optional[str] = None


class DataIngestion:
    """
    Handles data ingestion from multiple sources:
    1. Local CSV files
    2. MongoDB database
    
    Features:
    - Data extraction from local source
    - MongoDB connection and data retrieval
    - Data validation
    - Train-test split
    - Error handling and logging
    """
    
    def __init__(self, config: Optional[DataIngestionConfig] = None):
        """
        Initialize DataIngestion component
        
        Args:
            config: DataIngestionConfig object with paths and settings
        """
        self.config = config if config else DataIngestionConfig()
        logger.info("DataIngestion component initialized")
        
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate the ingested dataframe
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if valid, raises exception otherwise
        """
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame is empty or None")
            
            if df.shape[0] == 0:
                raise ValueError("DataFrame has no rows")
            
            if df.shape[1] == 0:
                raise ValueError("DataFrame has no columns")
            
            logger.info(f"DataFrame validation passed: {df.shape[0]} rows, {df.shape[1]} columns")
            return True
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def ingest_from_local(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Ingest data from local CSV file
        
        Args:
            file_path: Path to CSV file. If None, uses config default
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            # Use provided path or default from config
            data_path = file_path if file_path else self.config.raw_data_path
            
            logger.info(f"Starting data ingestion from local file: {data_path}")
            
            # Check if file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at: {data_path}")
            
            # Read CSV file
            df = pd.read_csv(data_path)
            
            # Validate dataframe
            self._validate_dataframe(df)
            
            logger.info(f"Successfully loaded data from {data_path}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in local data ingestion: {str(e)}")
            raise CustomException(e, sys)
    
    def ingest_from_mongodb(self, 
                           url: Optional[str] = None,
                           database: Optional[str] = None,
                           collection: Optional[str] = None,
                           query: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Ingest data from MongoDB database
        
        Args:
            url: MongoDB connection URL
            database: Database name
            collection: Collection name
            query: MongoDB query filter (default: fetch all documents)
            
        Returns:
            pd.DataFrame: Data from MongoDB
        """
        try:
            # Use provided values or defaults from config
            mongo_url = url if url else self.config.mongodb_url
            db_name = database if database else self.config.mongodb_database
            coll_name = collection if collection else self.config.mongodb_collection
            
            if not all([mongo_url, db_name, coll_name]):
                raise ValueError(
                    "MongoDB configuration incomplete. Provide url, database, and collection."
                )
            
            logger.info(f"Connecting to MongoDB: {db_name}.{coll_name}")
            
            # Connect to MongoDB
            client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
            
            # Test connection
            client.admin.command('ping')
            logger.info("MongoDB connection successful")
            
            # Access database and collection
            db = client[db_name]
            collection_obj = db[coll_name]
            
            # Fetch data
            query_filter = query if query else {}
            cursor = collection_obj.find(query_filter)
            
            # Convert to DataFrame
            df = pd.DataFrame(list(cursor))
            
            # Remove MongoDB _id field if present
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Close connection
            client.close()
            
            # Validate dataframe
            self._validate_dataframe(df)
            
            logger.info(f"Successfully loaded {df.shape[0]} documents from MongoDB")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            raise CustomException(f"Could not connect to MongoDB: {str(e)}", sys)
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed: {str(e)}")
            raise CustomException(f"MongoDB operation error: {str(e)}", sys)
        except Exception as e:
            logger.error(f"Error in MongoDB data ingestion: {str(e)}")
            raise CustomException(e, sys)
    
    def save_data(self, df: pd.DataFrame, file_path: str) -> str:
        """
        Save DataFrame to CSV file
        
        Args:
            df: DataFrame to save
            file_path: Path where to save the file
            
        Returns:
            str: Path where file was saved
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            
            logger.info(f"Data saved successfully to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise CustomException(e, sys)
    
    def split_data(self, df: pd.DataFrame) -> tuple:
        """
        Split data into train and test sets
        
        Args:
            df: DataFrame to split
            
        Returns:
            tuple: (train_df, test_df)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            logger.info(f"Splitting data - Test size: {self.config.test_size}, Random state: {self.config.random_state}")
            
            # Split the data
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            logger.info(f"Train set: {train_df.shape[0]} samples ({(1-self.config.test_size)*100:.0f}%)")
            logger.info(f"Test set: {test_df.shape[0]} samples ({self.config.test_size*100:.0f}%)")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self, 
                               source: str = 'local',
                               split_data: bool = True,
                               save_raw: bool = True,
                               **kwargs) -> tuple:
        """
        Main method to initiate data ingestion from specified source
        
        Args:
            source: Data source ('local' or 'mongodb')
            split_data: Whether to split data into train and test sets
            save_raw: Whether to save raw data to artifacts folder
            **kwargs: Additional arguments for specific ingestion methods
            
        Returns:
            tuple: (train_df, test_df) if split_data=True, otherwise (df, None)
        """
        try:
            logger.info(f"=" * 80)
            logger.info(f"DATA INGESTION STARTED - Source: {source.upper()}")
            logger.info(f"=" * 80)
            
            # Create artifacts directory
            os.makedirs(self.config.artifacts_dir, exist_ok=True)
            logger.info(f"Artifacts directory ready: {self.config.artifacts_dir}")
            
            # Ingest data based on source
            if source.lower() == 'local':
                df = self.ingest_from_local(kwargs.get('file_path'))
            elif source.lower() == 'mongodb':
                df = self.ingest_from_mongodb(
                    url=kwargs.get('url'),
                    database=kwargs.get('database'),
                    collection=kwargs.get('collection'),
                    query=kwargs.get('query')
                )
            else:
                raise ValueError(f"Unsupported data source: {source}. Use 'local' or 'mongodb'")
            
            # Save raw data artifact if requested
            if save_raw:
                self.save_data(df, self.config.raw_data_artifact)
                logger.info(f"Raw data artifact saved")
            
            # Split data if requested
            if split_data:
                train_df, test_df = self.split_data(df)
                
                # Save train and test sets as artifacts
                self.save_data(train_df, self.config.train_data_path)
                self.save_data(test_df, self.config.test_data_path)
                
                logger.info(f"=" * 80)
                logger.info(f"DATA INGESTION COMPLETED SUCCESSFULLY")
                logger.info(f"Total records: {df.shape[0]}")
                logger.info(f"Total features: {df.shape[1]}")
                logger.info(f"Train set saved: {self.config.train_data_path}")
                logger.info(f"Test set saved: {self.config.test_data_path}")
                logger.info(f"=" * 80)
                
                return train_df, test_df
            else:
                logger.info(f"=" * 80)
                logger.info(f"DATA INGESTION COMPLETED SUCCESSFULLY")
                logger.info(f"Total records: {df.shape[0]}")
                logger.info(f"Total features: {df.shape[1]}")
                logger.info(f"=" * 80)
                
                return df, None
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
            raise CustomException(e, sys)


# Example usage and testing
if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print("TESTING DATA INGESTION COMPONENT")
        print("=" * 80)
        
        # Initialize data ingestion
        data_ingestion = DataIngestion()
        
        # Test 1: Ingest from local CSV with train-test split
        print("\n[TEST 1] Ingesting data from local CSV with train-test split...")
        train_df, test_df = data_ingestion.initiate_data_ingestion(
            source='local',
            split_data=True
        )
        print(f"\nLocal ingestion successful!")
        print(f"\nTrain set shape: {train_df.shape}")
        print(f"Test set shape: {test_df.shape}")
        print(f"\nTrain set - First few rows:")
        print(train_df.head())
        print(f"\nArtifacts created:")
        print(f"  - artifacts/raw_data.csv")
        print(f"  - artifacts/train.csv")
        print(f"  - artifacts/test.csv")
        
        # Test 2: MongoDB ingestion (example configuration)
        print("\n" + "-" * 80)
        print("[TEST 2] MongoDB ingestion example (skipped - configure MongoDB first)")
        print("\nTo use MongoDB ingestion, uncomment and configure:")
        print("""
        # Example MongoDB configuration
        config = DataIngestionConfig(
            mongodb_url="mongodb://localhost:27017/",
            mongodb_database="ecommerce_db",
            mongodb_collection="customers"
        )
        data_ingestion_mongo = DataIngestion(config)
        df_mongo = data_ingestion_mongo.initiate_data_ingestion(source='mongodb')
        """)
        
        print("\n" + "=" * 80)
        print("DATA INGESTION TESTING COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

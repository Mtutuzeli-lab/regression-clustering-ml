"""
Load CSV Data into MongoDB Atlas

This script loads the ecommerce customer data from CSV into MongoDB.
Run this once to populate your MongoDB database.
"""

import os
import sys
import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

sys.path.append('src')
from src.logger import logger

def load_data_to_mongodb():
    """Load CSV data into MongoDB Atlas"""
    
    try:
        # Load environment variables
        load_dotenv()
        
        logger.info("=" * 80)
        logger.info("LOADING DATA TO MONGODB")
        logger.info("=" * 80)
        
        # Get MongoDB credentials from .env
        mongodb_uri = os.getenv('MONGODB_URI')
        database_name = os.getenv('MONGODB_DATABASE', 'ecommerce_db')
        collection_name = os.getenv('MONGODB_COLLECTION', 'customers')
        
        if not mongodb_uri or '<db_password>' in mongodb_uri:
            logger.error("MongoDB URI not configured properly!")
            print("\n❌ ERROR: Please update your .env file with your MongoDB password")
            print("Edit the .env file and replace <db_password> with your actual password")
            return False
        
        logger.info(f"Database: {database_name}")
        logger.info(f"Collection: {collection_name}")
        
        # Connect to MongoDB
        logger.info("Connecting to MongoDB Atlas...")
        client = MongoClient(mongodb_uri, server_api=ServerApi('1'))
        
        # Test connection
        client.admin.command('ping')
        logger.info("✓ Successfully connected to MongoDB!")
        
        # Get database and collection
        db = client[database_name]
        collection = db[collection_name]
        
        # Load CSV data
        logger.info("Loading CSV data...")
        csv_path = 'data/ecommerce_customer.csv'
        
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return False
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from CSV")
        
        # Clear existing data (optional - remove if you want to keep existing data)
        existing_count = collection.count_documents({})
        if existing_count > 0:
            logger.info(f"Clearing {existing_count} existing documents...")
            collection.delete_many({})
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Insert data into MongoDB
        logger.info("Inserting data into MongoDB...")
        result = collection.insert_many(records)
        
        logger.info(f"✓ Successfully inserted {len(result.inserted_ids)} documents!")
        
        # Verify insertion
        count = collection.count_documents({})
        logger.info(f"Total documents in collection: {count}")
        
        # Show sample document
        sample = collection.find_one()
        logger.info("\nSample document:")
        logger.info(f"  Email: {sample.get('Email', 'N/A')}")
        logger.info(f"  Avg Session Length: {sample.get('Avg. Session Length', 'N/A')}")
        logger.info(f"  Yearly Amount Spent: {sample.get('Yearly Amount Spent', 'N/A')}")
        
        logger.info("\n" + "=" * 80)
        logger.info("DATA LOADED SUCCESSFULLY")
        logger.info("=" * 80)
        
        print("\n✅ SUCCESS!")
        print(f"Loaded {count} customer records into MongoDB")
        print(f"Database: {database_name}")
        print(f"Collection: {collection_name}")
        
        # Close connection
        client.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading data to MongoDB: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MONGODB DATA LOADER")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Connect to MongoDB Atlas")
    print("2. Load customer data from CSV")
    print("3. Insert into MongoDB database")
    print("\nMake sure you have updated .env with your MongoDB password!")
    print("=" * 80)
    
    success = load_data_to_mongodb()
    sys.exit(0 if success else 1)

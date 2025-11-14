"""
Test script to verify logger functionality
"""
from src.logger import logger

def test_logging():
    """Test different logging levels"""
    print("🧪 Testing logger from external script...")
    
    logger.info("External test: Logger imported successfully")
    logger.warning("External test: This is a warning")
    logger.error("External test: This is an error")
    
    print("✅ External logger test completed!")

if __name__ == "__main__":
    test_logging()
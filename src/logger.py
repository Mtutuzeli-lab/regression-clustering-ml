import logging
import os   
from datetime import datetime

# Create logs directory
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()  # Also log to console
    ]
)

# Create a logger instance
logger = logging.getLogger(__name__)

# Test function for the logger
if __name__ == "__main__":
    print("Testing logger functionality...")
    print(f"Log file will be created at: {LOG_FILE_PATH}")
    
    logger.info("Logger test started")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    print("\n✅ Logger test completed!")
    print(f"Check the log file at: {LOG_FILE_PATH}")
    print("You should also see the messages printed above.")
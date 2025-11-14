import sys
import os
from typing import Any

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Generate detailed error message with file name, line number, and error details
    
    Args:
        error: The original error/exception
        error_detail: sys module to get exception info
    
    Returns:
        str: Formatted error message with file and line details
    """
    try:
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is None:
            return f"Error: {str(error)} (No traceback available)"
        
        file_name = os.path.basename(exc_tb.tb_frame.f_code.co_filename)
        error_message = "Error occurred in python script [{0}] line number [{1}] error message: [{2}]".format(
            file_name, exc_tb.tb_lineno, str(error)
        )
        return error_message
    except Exception as e:
        return f"Error in error handling: {str(e)} | Original error: {str(error)}"

class CustomException(Exception):
    """
    Custom exception class that provides detailed error information
    including file name and line number where the error occurred.
    
    Usage:
        try:
            # Some operation that might fail
            result = risky_operation()
        except Exception as e:
            raise CustomException(e, sys)
    """
    
    def __init__(self, error_message: Any, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
    
    def __repr__(self) -> str:
        return f"CustomException('{self.error_message}')"
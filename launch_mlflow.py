"""
Launch MLflow UI to view experiment tracking

This script launches the MLflow web interface where you can:
- Compare all model runs
- View metrics (R², RMSE, MAE)
- Analyze model parameters
- Download trained models
- Create visualizations

Run this after training your models with pipeline.py
"""

import subprocess
import sys
import webbrowser
import time

def launch_mlflow_ui():
    """Launch MLflow UI"""
    print("=" * 80)
    print("LAUNCHING MLFLOW UI")
    print("=" * 80)
    print("\nMLflow is starting...")
    print("The UI will open automatically in your browser")
    print("\nMLflow UI URL: http://localhost:5000")
    print("\nPress Ctrl+C to stop MLflow server")
    print("=" * 80)
    
    # Wait a moment then open browser
    time.sleep(2)
    webbrowser.open('http://localhost:5000')
    
    # Start MLflow UI
    try:
        subprocess.run(['mlflow', 'ui'], check=True)
    except KeyboardInterrupt:
        print("\n\nMLflow UI stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError launching MLflow UI: {str(e)}")
        print("\nTry running manually: mlflow ui")
        sys.exit(1)

if __name__ == "__main__":
    launch_mlflow_ui()

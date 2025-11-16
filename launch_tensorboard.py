"""
Launch TensorBoard UI for Deep Learning Model Visualization

This script launches TensorBoard to visualize:
- Training & validation loss curves
- MAE and RMSE metrics over epochs
- Model architecture graphs
- Learning rate schedules
- Weight histograms and distributions
- Gradient flow analysis

Run this after training deep learning models with train_deep_learning.py
"""

import subprocess
import sys
import webbrowser
import time
import os

def launch_tensorboard():
    """Launch TensorBoard UI"""
    
    log_dir = 'logs/tensorboard'
    
    # Check if logs exist
    if not os.path.exists(log_dir):
        print("❌ TensorBoard logs not found!")
        print(f"\nPlease train deep learning models first:")
        print("  python train_deep_learning.py")
        print(f"\nThis will create logs in: {log_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("LAUNCHING TENSORBOARD")
    print("=" * 80)
    print(f"\nLog directory: {log_dir}")
    print("\nTensorBoard is starting...")
    print("The UI will open automatically in your browser")
    print("\nTensorBoard URL: http://localhost:6006")
    print("\nPress Ctrl+C to stop TensorBoard server")
    print("=" * 80)
    
    # Wait a moment then open browser
    time.sleep(2)
    webbrowser.open('http://localhost:6006')
    
    # Start TensorBoard
    try:
        subprocess.run(['tensorboard', '--logdir', log_dir], check=True)
    except KeyboardInterrupt:
        print("\n\nTensorBoard stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError launching TensorBoard: {str(e)}")
        print(f"\nTry running manually: tensorboard --logdir={log_dir}")
        sys.exit(1)


if __name__ == "__main__":
    launch_tensorboard()

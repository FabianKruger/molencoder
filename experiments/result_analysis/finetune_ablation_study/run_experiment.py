#!/usr/bin/env python
"""
Simple runner script for the hyperparameter ablation study.
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Path to the config file (in the same directory)
    config_path = script_dir / "config.yaml"
    
    # Path to the main script (in the same directory)
    script_path = script_dir / "hyperparameter_ablation_study.py"
    
    # Check if files exist
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Run the script
    cmd = [
        sys.executable, 
        str(script_path), 
        "--config", 
        str(config_path)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("Hyperparameter ablation study completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running the script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
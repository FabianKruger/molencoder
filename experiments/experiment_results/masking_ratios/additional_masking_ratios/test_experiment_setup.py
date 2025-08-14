#!/usr/bin/env python3
"""
Test script to verify the additional masking ratio experiment setup works correctly.
This script will run a short test with one of the configurations to ensure everything is properly set up.
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.models.pretrain_mlm import run_trial

def test_experiment_setup():
    """Test the experiment setup by running a short training session."""
    
    # Find one of the generated config files to test
    base_dir = Path("/home/ubuntu/smiles_encoder/experiments/additional_masking_ratios")
    config_files = list(base_dir.glob("*/*/masking_*/config.yaml"))
    
    if not config_files:
        print("❌ ERROR: No config files found! Run the creation script first.")
        return False
    
    # Use the first config file for testing
    test_config_path = config_files[0]
    print(f"🧪 Testing with config: {test_config_path}")
    
    # Load the configuration
    with test_config_path.open("r") as file:
        config = yaml.safe_load(file)
    
    # Modify config for testing (make it run faster and shorter)
    original_debug = config.get("debug", False)
    original_max_steps = config["training_arguments"]["max_steps"]
    original_eval_steps = config["training_arguments"]["eval_steps"]
    original_save_steps = config["training_arguments"]["save_steps"]
    
    # Set debug mode and reduce steps for testing
    config["debug"] = True  # This will use smaller dataset
    config["training_arguments"]["max_steps"] = 10  # Just 10 steps for testing
    config["training_arguments"]["eval_steps"] = 5   # Evaluate after 5 steps
    config["training_arguments"]["save_steps"] = 5   # Save after 5 steps
    config["training_arguments"]["eval_on_start"] = True
    config["training_arguments"]["save_total_limit"] = 1  # Only keep 1 checkpoint
    
    # Create a test result folder
    test_result_dir = base_dir / "test_results"
    config["result_folder_path"] = str(test_result_dir)
    config["training_arguments"]["output_dir"] = str(test_result_dir / "trainer")
    config["training_arguments"]["logging_dir"] = str(test_result_dir / "training")
    
    print(f"🔧 Modified config for testing:")
    print(f"   - Debug mode: {config['debug']}")
    print(f"   - Max steps: {config['training_arguments']['max_steps']}")
    print(f"   - Dataset: {config['dataset_name']}")
    print(f"   - Masking probability: {config['masking_probability']}")
    print(f"   - Test results will be saved to: {test_result_dir}")
    
    try:
        print("\n🚀 Starting test run...")
        
        # Run the trial
        run_trial(config)
        
        print("\n✅ SUCCESS: Test completed successfully!")
        print("🎉 Your experiment setup is working correctly.")
        print(f"📁 Test results saved to: {test_result_dir}")
        
        # Clean up test results
        print(f"\n🧹 Cleaning up test results...")
        import shutil
        if test_result_dir.exists():
            shutil.rmtree(test_result_dir)
            print("✅ Test results cleaned up.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with error:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("\n🔍 Please check your setup and try again.")
        
        # Clean up test results even if failed
        import shutil
        if test_result_dir.exists():
            shutil.rmtree(test_result_dir)
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the additional masking ratio experiment setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    print("🧪 Testing Additional Masking Ratio Experiment Setup")
    print("=" * 60)
    
    success = test_experiment_setup()
    
    if success:
        print("\n🎯 CONCLUSION: Your experiment is ready to run!")
        print("💡 You can now execute: ./run_additional_masking_ratio_experiment.sh")
        sys.exit(0)
    else:
        print("\n❌ CONCLUSION: There are issues with your setup.")
        print("🔧 Please fix the errors and run the test again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
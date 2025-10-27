#!/usr/bin/env python3
"""
Setup script for Clash Royale Deep Learning AI

This script helps set up the environment and install dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install Python requirements"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    return run_command(f"pip install -r {requirements_file}", "Installing Python requirements")

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "logs", 
        "data",
        "checkpoints"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_env_file():
    """Create .env file with default configuration"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print("✓ .env file already exists")
        return
    
    env_content = """# Clash Royale AI Configuration

# Frame dimensions
FRAME_WIDTH=1080
FRAME_HEIGHT=1920

# ZeroMQ ports
ZMQ_ADDRESS=tcp://localhost:5550
ELIXIR_PORT=5560
CARDS_PORT=5590
TROOPS_PORT=5580
WIN_PORT=5570

# Roboflow API (optional)
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_WORKFLOW_ID=your_workflow_id_here

# Training configuration
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=100

# Model configuration
HIDDEN_SIZE=128
SEQUENCE_LENGTH=10

# Game player coordinates (adjust for your setup)
CARD_0_X=1489
CARD_0_Y=895
CARD_1_X=1604
CARD_1_Y=901
CARD_2_X=1704
CARD_2_Y=904
CARD_3_X=1795
CARD_3_Y=946
START_MATCH_X=540
START_MATCH_Y=1200
END_MATCH_X=540
END_MATCH_Y=1400
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✓ Created .env file with default configuration")

def check_dependencies():
    """Check if all dependencies are available"""
    dependencies = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("zmq", "PyZMQ"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("tensorboard", "TensorBoard")
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} is available")
        except ImportError:
            print(f"✗ {name} is missing")
            missing.append(name)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def setup_git_hooks():
    """Setup git hooks for code quality"""
    git_dir = Path(__file__).parent / ".git"
    if not git_dir.exists():
        print("ℹ Not a git repository, skipping git hooks setup")
        return
    
    pre_commit_hook = git_dir / "hooks" / "pre-commit"
    pre_commit_content = """#!/bin/sh
# Pre-commit hook for Clash Royale AI

echo "Running pre-commit checks..."

# Check Python syntax
python -m py_compile deeplearning/*.py
if [ $? -ne 0 ]; then
    echo "Python syntax errors found"
    exit 1
fi

# Check imports
python -c "import deeplearning.clash_royale_ai; import deeplearning.integration_layer; import deeplearning.training_pipeline"
if [ $? -ne 0 ]; then
    echo "Import errors found"
    exit 1
fi

echo "Pre-commit checks passed"
"""
    
    with open(pre_commit_hook, 'w') as f:
        f.write(pre_commit_content)
    
    # Make executable
    os.chmod(pre_commit_hook, 0o755)
    print("✓ Git pre-commit hook installed")

def main():
    """Main setup function"""
    print("Setting up Clash Royale Deep Learning AI...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install manually.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("Some dependencies are missing. Please install them manually.")
        sys.exit(1)
    
    # Setup git hooks
    setup_git_hooks()
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Configure your .env file with your specific settings")
    print("2. Set up Roboflow API keys if using computer vision modules")
    print("3. Run 'python main.py --mode status' to check system status")
    print("4. Run 'python main.py --mode collect-data' to start collecting training data")
    print("5. Run 'python main.py --mode train' to train the AI model")
    print("6. Run 'python main.py --mode play' to use the AI in gameplay")

if __name__ == "__main__":
    main()

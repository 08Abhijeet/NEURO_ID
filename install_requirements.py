#!/usr/bin/env python3
"""
Script to install required packages for the NeuroID project.
This script handles the Windows Long Path issue by trying different installation methods.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors gracefully."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🔧 NeuroID Project - Package Installation Script")
    print("=" * 50)
    
    # List of packages to install
    packages = [
        "numpy",
        "pandas", 
        "matplotlib",
        "scikit-learn",
        "seaborn"
    ]
    
    # Install basic packages first
    print("\n📦 Installing basic packages...")
    for package in packages:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️  Failed to install {package}, but continuing...")
    
    # Try different TensorFlow installation methods
    print("\n🧠 Installing TensorFlow...")
    
    # Method 1: Try TensorFlow 2.15.0
    if run_command("pip install tensorflow==2.15.0", "Installing TensorFlow 2.15.0"):
        print("✅ TensorFlow 2.15.0 installed successfully!")
        return
    
    # Method 2: Try TensorFlow CPU only
    if run_command("pip install tensorflow-cpu==2.15.0", "Installing TensorFlow CPU 2.15.0"):
        print("✅ TensorFlow CPU 2.15.0 installed successfully!")
        return
    
    # Method 3: Try with --no-deps flag
    if run_command("pip install tensorflow==2.15.0 --no-deps", "Installing TensorFlow 2.15.0 (no dependencies)"):
        print("✅ TensorFlow 2.15.0 installed (no dependencies)!")
        print("⚠️  You may need to install additional dependencies manually.")
        return
    
    # Method 4: Try conda if available
    print("\n🔄 Trying conda installation...")
    if run_command("conda install tensorflow=2.15.0 -c conda-forge", "Installing TensorFlow via conda"):
        print("✅ TensorFlow installed via conda!")
        return
    
    print("\n❌ All TensorFlow installation methods failed.")
    print("\n🔧 Manual Installation Options:")
    print("1. Enable Windows Long Path support:")
    print("   - Open PowerShell as Administrator")
    print("   - Run: New-ItemProperty -Path \"HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem\" -Name \"LongPathsEnabled\" -Value 1 -PropertyType DWORD -Force")
    print("   - Restart your computer")
    print("   - Then run: pip install tensorflow==2.15.0")
    print("\n2. Use Anaconda/Miniconda:")
    print("   - Download from: https://www.anaconda.com/products/distribution")
    print("   - Create a new environment: conda create -n neuroid python=3.9")
    print("   - Activate environment: conda activate neuroid")
    print("   - Install: conda install tensorflow=2.15.0")
    print("\n3. Use Google Colab or Jupyter Notebook online")
    print("\n4. Try the notebook with limited functionality (data loading and analysis only)")

if __name__ == "__main__":
    main()

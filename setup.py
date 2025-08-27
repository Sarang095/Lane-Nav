#!/usr/bin/env python3
"""
Setup script for Lane-Nav autonomous driving framework.
Installs dependencies and prepares the environment.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and return success status."""
    print(f"🔧 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(
            command.split(), 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"   ❌ Command not found: {command.split()[0]}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    major, minor = sys.version_info[:2]
    print(f"   Python {major}.{minor}")
    
    if major < 3 or (major == 3 and minor < 8):
        print("   ❌ Python 3.8+ required")
        return False
    else:
        print("   ✅ Python version compatible")
        return True

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Try pip3 first, then pip
    pip_commands = ["pip3", "pip"]
    
    for pip_cmd in pip_commands:
        if run_command(f"{pip_cmd} --version", f"Checking {pip_cmd}"):
            break
    else:
        print("❌ No pip found. Please install pip first.")
        return False
    
    # Install requirements
    success = run_command(
        f"{pip_cmd} install -r requirements.txt",
        "Installing requirements from requirements.txt"
    )
    
    if not success:
        print("\n⚠️ Failed to install all dependencies automatically.")
        print("You may need to install manually:")
        print("   pip3 install stable-baselines3[extra] highway-env gymnasium torch imitation")
    
    return success

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "models",
        "logs", 
        "tensorboard_logs",
        "logs/eval"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    return True

def make_scripts_executable():
    """Make training and evaluation scripts executable."""
    print("\n🔧 Making scripts executable...")
    
    scripts = [
        "train_highway.py",
        "train_intersection.py", 
        "train_roundabout.py",
        "train_parking.py",
        "evaluate.py",
        "test_setup.py",
        "example_usage.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            os.chmod(script, 0o755)
            print(f"   ✅ {script}")
    
    return True

def run_verification():
    """Run setup verification."""
    print("\n🧪 Running setup verification...")
    
    success = run_command("python3 test_setup.py", "Verifying installation")
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("\nYou can now:")
        print("   • Train agents: python3 train_highway.py --mode rl")
        print("   • Evaluate agents: python3 evaluate.py highway --mode rl")
        print("   • Run examples: python3 example_usage.py")
    else:
        print("\n⚠️ Setup verification failed. Check the output above.")
    
    return success

def main():
    """Main setup function."""
    print("🚀 Lane-Nav Framework Setup")
    print("=" * 60)
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Dependency Installation", install_dependencies), 
        ("Directory Creation", create_directories),
        ("Script Permissions", make_scripts_executable),
        ("Setup Verification", run_verification)
    ]
    
    for step_name, step_func in steps:
        try:
            success = step_func()
            if not success:
                print(f"\n❌ Setup failed at: {step_name}")
                print("Please fix the issues above and try again.")
                return False
        except KeyboardInterrupt:
            print(f"\n🛑 Setup interrupted during: {step_name}")
            return False
        except Exception as e:
            print(f"\n💥 Unexpected error in {step_name}: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("🎯 Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the README.md for detailed usage instructions")
    print("2. Try the example: python3 example_usage.py") 
    print("3. Train your first agent: python3 train_highway.py --mode rl")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
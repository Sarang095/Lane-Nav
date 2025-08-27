#!/bin/bash

# Lane-Nav Installation Script
# Installs dependencies and sets up the autonomous driving framework

echo "🚀 Lane-Nav Framework Installation"
echo "=================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi

echo "🐍 Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install pip first."
    exit 1
fi

# Determine pip command
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
else
    PIP_CMD="pip"
fi

echo "📦 Using pip: $PIP_CMD"

# Create virtual environment (optional but recommended)
read -p "🤔 Create virtual environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🌍 Creating virtual environment..."
    python3 -m venv lane_nav_env
    source lane_nav_env/bin/activate
    echo "✅ Virtual environment activated"
fi

# Install dependencies
echo "📦 Installing dependencies..."
$PIP_CMD install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    echo "Try manually: $PIP_CMD install stable-baselines3[extra] highway-env gymnasium torch imitation"
    exit 1
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p models logs tensorboard_logs logs/eval
echo "✅ Directories created"

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x *.py
echo "✅ Scripts are now executable"

# Run verification
echo "🧪 Running setup verification..."
python3 test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Train an agent:     python3 train_highway.py --mode rl"
    echo "  2. Evaluate it:        python3 evaluate.py highway --mode rl"
    echo "  3. Try the examples:   python3 example_usage.py"
    echo "  4. Read the docs:      cat README.md"
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Note: Virtual environment is active. To deactivate later, run: deactivate"
    fi
else
    echo ""
    echo "⚠️ Installation verification failed"
    echo "Check the error messages above and try running test_setup.py manually"
fi
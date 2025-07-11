#!/bin/bash

# Multimodal AI API with Meta-Transformer Backend Integration
# Startup script for the integrated backend

set -e

echo "🚀 Starting Multimodal AI API with Meta-Transformer Backend Integration..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION is installed, but Python $REQUIRED_VERSION+ is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/lib/python*/site-packages/fastapi" ]; then
    echo "📥 Installing dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install basic requirements first
    if [ -f "requirements.txt" ]; then
        echo "📦 Installing basic requirements..."
        pip install -r requirements.txt
    fi
    
    # Install integrated requirements
    if [ -f "requirements_integrated.txt" ]; then
        echo "📦 Installing Meta-Transformer integration requirements..."
        pip install -r requirements_integrated.txt
    fi
else
    echo "✅ Dependencies already installed"
fi

# Check if Meta-Transformer backend source exists
if [ ! -d "../backend_source" ]; then
    echo "⚠️ Warning: Meta-Transformer backend source not found at ../backend_source"
    echo "   The API will run in fallback mode without Meta-Transformer integration"
    echo "   To enable full integration, ensure the backend_source directory exists"
fi

# Create models directory if it doesn't exist
mkdir -p models

# Check for Meta-Transformer weights
if [ ! -f "models/Meta-Transformer_base_patch16_encoder.pth" ]; then
    echo "⚠️ Warning: Meta-Transformer weights not found"
    echo "   Download pretrained weights from:"
    echo "   - Base model: https://drive.google.com/file/d/19ahcN2QKknkir_bayhTW5rucuAiX0OXq/view?usp=sharing"
    echo "   - Large model: https://drive.google.com/file/d/15EtzCBAQSqmelhdLz6k880A19_RpcX9B/view?usp=drive_link"
    echo "   Place them in the models/ directory for full functionality"
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export META_TRANSFORMER_MODEL_PATH="models/"
export DEVICE="cpu"  # Change to "cuda" if GPU is available

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected"
    export DEVICE="cuda"
else
    echo "💻 Using CPU for inference"
fi

# Kill any existing processes on port 8000
echo "🔧 Checking for existing processes on port 8000..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "🔄 Killing existing process on port 8000..."
    lsof -ti:8000 | xargs kill -9
    sleep 2
fi

# Start the integrated API
echo "🚀 Starting integrated Multimodal AI API..."
echo "📊 API will be available at: http://localhost:8000"
echo "📚 Documentation will be available at: http://localhost:8000/docs"
echo "🔑 Use API key: demo-api-key for testing"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the integrated application
if [ -f "app_integrated.py" ]; then
    python app_integrated.py
else
    echo "❌ app_integrated.py not found. Running simple version instead..."
    python app_simple.py
fi 
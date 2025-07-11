#!/bin/bash

# Multimodal AI API Startup Script

set -e

echo "🚀 Starting Multimodal AI API..."

# Check if Python is installed
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

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create logs directory
mkdir -p logs

# Check if running in development or production mode
if [ "$1" = "dev" ]; then
    echo "🔬 Starting in development mode..."
    export LOG_LEVEL=DEBUG
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
elif [ "$1" = "prod" ]; then
    echo "🏭 Starting in production mode..."
    export LOG_LEVEL=INFO
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
else
    echo "🔬 Starting in development mode (default)..."
    echo "Usage: ./start.sh [dev|prod]"
    echo "  dev  - Development mode with auto-reload"
    echo "  prod - Production mode with multiple workers"
    echo ""
    export LOG_LEVEL=DEBUG
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
fi 
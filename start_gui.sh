#!/bin/bash
# 🚀 Quick Start Script for Robot Navigation GUI

echo "========================================"
echo "   🤖 Robot Navigation - GUI Launcher"
echo "========================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python not found! Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if dependencies are installed
if ! python3 -c "import PyQt6" &> /dev/null; then
    echo "📦 Installing GUI dependencies..."
    pip3 install -r requirements-gui.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "🚀 Launching GUI..."
echo ""

python3 launcher.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to launch GUI"
    echo ""
    echo "Troubleshooting:"
    echo "1. Make sure you're in the project root directory"
    echo "2. Install dependencies: pip3 install -r requirements-gui.txt"
    echo "3. Check Python version: python3 --version (need 3.8+)"
    echo ""
    exit 1
fi

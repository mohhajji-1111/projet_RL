#!/bin/bash
# Quick Setup Script for Linux/Mac
# Run this to quickly set up the project

echo "========================================"
echo "Robot Navigation RL - Quick Setup"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found! Please install Python 3.8+"
    exit 1
fi

echo "[1/4] Python found!"

# Create virtual environment
echo "[2/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created"
else
    echo "  Virtual environment already exists"
fi

# Activate and install
echo "[3/4] Installing dependencies..."
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "[4/4] Installation complete!"
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Test environment: python -c 'from src.environment import NavigationEnv; print(\"OK\")'"
echo "  3. Start training: python scripts/train.py --config configs/base_config.yaml"
echo ""
echo "For more information, see PROJECT_SUMMARY.md"
echo ""

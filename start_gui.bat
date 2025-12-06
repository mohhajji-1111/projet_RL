@echo off
REM 🚀 Quick Start Script for Robot Navigation GUI

echo ========================================
echo    🤖 Robot Navigation - GUI Launcher
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if dependencies are installed
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing GUI dependencies...
    pip install -r requirements-gui.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed
) else (
    echo ✅ Dependencies already installed
)

echo.
echo 🚀 Launching GUI...
echo.

python launcher.py

if errorlevel 1 (
    echo.
    echo ❌ Failed to launch GUI
    echo.
    echo Troubleshooting:
    echo 1. Make sure you're in the project root directory
    echo 2. Install dependencies: pip install -r requirements-gui.txt
    echo 3. Check Python version: python --version (need 3.8+)
    echo.
    pause
    exit /b 1
)

@echo off
REM Quick Setup Script for Windows
REM Run this to quickly set up the project

echo ========================================
echo Robot Navigation RL - Quick Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo [1/4] Python found!

REM Create virtual environment
echo [2/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo   Virtual environment created
) else (
    echo   Virtual environment already exists
)

REM Activate and install
echo [3/4] Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo [4/4] Installation complete!
echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Activate environment: venv\Scripts\activate
echo   2. Test environment: python -c "from src.environment import NavigationEnv; print('OK')"
echo   3. Start training: python scripts\train.py --config configs\base_config.yaml
echo.
echo For more information, see PROJECT_SUMMARY.md
echo.
pause

@echo off
echo 🚀 Multi-Agent Web Scraping System - Setup Script
echo ================================================

echo.
echo 🔍 Checking Python installation...

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Python found via 'python' command
    set PYTHON_CMD=python
    goto :install_deps
)

python3 --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Python found via 'python3' command
    set PYTHON_CMD=python3
    goto :install_deps
)

REM Try full path
"C:\Users\Office Personal\AppData\Local\Microsoft\WindowsApps\python3.exe" --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Python found at full path
    set PYTHON_CMD="C:\Users\Office Personal\AppData\Local\Microsoft\WindowsApps\python3.exe"
    goto :install_deps
)

echo ❌ Python not found. Please install Python 3.8+ from python.org
echo    Or run: winget install Python.Python.3.12
pause
exit /b 1

:install_deps
echo.
echo 📦 Installing dependencies...
%PYTHON_CMD% -m pip install --upgrade pip
%PYTHON_CMD% -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    echo 💡 Try running as administrator or use: pip install --user -r requirements.txt
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully

echo.
echo 🧪 Testing system...
%PYTHON_CMD% test_system.py

echo.
echo 🎯 Setup complete! Try these commands:
echo   1. %PYTHON_CMD% main.py agents
echo   2. %PYTHON_CMD% main.py scrape --interactive
echo   3. %PYTHON_CMD% examples/simple_scrape.py
echo   4. %PYTHON_CMD% quick_start.py

echo.
echo 📚 Documentation:
echo   - SETUP_GUIDE.md - Detailed setup instructions
echo   - README.md - System documentation
echo   - examples/ - Example scripts

pause

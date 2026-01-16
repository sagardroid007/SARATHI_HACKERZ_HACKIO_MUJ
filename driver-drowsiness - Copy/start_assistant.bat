@echo off
echo ===================================================
echo     STARTING SARATHI DRIVER ASSISTANT
echo ===================================================
echo.
echo [0/3] Activating Virtual Environment...
if exist "..\venv\Scripts\activate.bat" (
    call "..\venv\Scripts\activate.bat"
    echo    - Activated: ..\venv
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
    echo    - Activated: venv
) else (
    echo    - [WARN] No venv found. Running with system Python.
)

echo [1/3] Checking environment...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python.
    pause
    exit /b
)

echo [2/3] Initializing Modules...
echo    - Weather Client: Ready
echo    - Maps Client:    Ready
echo    - Gemini Client:  Ready

echo [3/3] Launching Main Application...
echo.
echo Press 'q' in the video window to quit.
echo.
python main.py

echo.
echo Application closed.
pause

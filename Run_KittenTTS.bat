@echo off
echo ==============================================
echo        KittenTTS Graphical Interface
echo ==============================================
echo.

:: Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Virtual environment not found. Setting it up now...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python is installed and in PATH.
        pause
        exit /b 1
    )
)

:: Activate the environment and install/update dependencies quietly
echo [INFO] Updating dependencies...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

venv\Scripts\python.exe -m pip install -e . -q
if errorlevel 1 (
    echo [ERROR] Initialization failed while installing dependencies.
    pause
    exit /b 1
)

:: Run the UI application
echo [INFO] Launching UI...
venv\Scripts\python.exe -m kittentts.ui

pause

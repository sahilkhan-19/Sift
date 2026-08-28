@echo off
REM ============================================================
REM  Build Sift.exe
REM  Requires: pip install pyinstaller  (one-time)
REM ============================================================

echo.
echo [[ BUILD  ]]  Building Sift.exe...
echo.

pyinstaller ^
    --onefile ^
    --console ^
    --name Sift ^
    --clean ^
    sift_exe.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [[ ERROR  ]]  Build failed.
    pause
    exit /b 1
)

echo.
echo [[ BUILD  ]]  Done!
echo [[ BUILD  ]]  Sift.exe is in: dist\Sift.exe
echo.
echo    Copy dist\Sift.exe into the Sift project folder
echo    (next to launcher.py, main.py, etc.) and double-click to run.
echo.
pause

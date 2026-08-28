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
echo [[ BUILD  ]]  Copying dist\Sift.exe to project root...
copy /Y dist\Sift.exe .

echo.
echo [[ BUILD  ]]  Done! Sift.exe is ready in the project root folder.
echo.
pause

@echo off
rem BUFFER LINE
cls
echo ===========================================
echo    GPU MONITOR - ASCII VERSION
echo ===========================================
echo.
if exist "venv\Scripts\python.exe" goto :OK
echo [ERROR] venv missing
pause
exit /b
:OK
echo Checking CUDA...
.\venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"
echo.
echo Checking GPU...
.\venv\Scripts\python.exe -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
pause

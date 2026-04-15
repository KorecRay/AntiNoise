@echo off
cls
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] venv/Scripts/activate.bat not found.
    pause
    exit /b
)
call venv\Scripts\activate.bat
echo ===========================================
echo    AI Voice Separation - SEPARATOR
echo ===========================================
echo.
echo Scanning models folder...
dir /b models\*.pth
echo.
set /p model_filename="Type the model name (e.g. unet_model_v1.pth): "
if "%model_filename%"=="" (
    echo [ERROR] No model specified.
    pause
    exit /b
)
set "full_model_path=models/%model_filename%"
set /a count=0
if not exist "source" mkdir source
if not exist "output" mkdir output
for %%f in (source\*.wav source\*.mp3 source\*.flac) do (
    set /a count+=1
    echo Processing: %%~nxf
    python separate.py "%%f" --output_file "output/%%~nf_res.wav" --model "%full_model_path%"
)
echo.
echo Processed: %count% files.
pause

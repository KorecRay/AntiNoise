@echo off
cls
echo ===========================================
echo    AI Voice Separation - SEPARATOR V4.0
echo ===========================================
echo.

call venv\Scripts\activate.bat

:: Step 1: ????
echo Choose Architecture to use:
echo  1. M1 (Basic Models)
echo  2. M2 (Attention Models)
echo.
set m_ver=1
set /p user_ver="Enter Architecture [1-2]: "
if not "%user_ver%"=="" set m_ver=%user_ver%

:: Step 2: ????????
echo.
if "%m_ver%"=="1" (
    echo [M1 Models Available]:
    dir /b models\M1\*.pth
    set s_script=separate_M1.py
    set s_dir=models/M1
) else (
    echo [M2 Models Available]:
    dir /b models\M2\*.pth
    set s_script=separate_M2.py
    set s_dir=models/M2
)

:: Step 3: ??????
echo.
set /p m_file="Enter filename from list above (e.g. unet.pth): "
if "%m_file%"=="" (
    echo [ERROR] No model selected.
    pause
    exit /b
)

:: Step 4: ????
echo.
set /a count=0
for %%f in (source\*.wav source\*.mp3) do (
    set /a count+=1
    echo Processing [!count!]: %%~nxf
    python %s_script% "%%f" --output_file "output/%%~nf_M%m_ver%_res.wav" --model "%s_dir%/%m_file%"
)

echo.
echo Processed: %count% files.
pause

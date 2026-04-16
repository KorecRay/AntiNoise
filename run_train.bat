@echo off
cls
echo ===========================================
echo    AI Voice Separation - TRAIN WIZARD V4.1
echo ===========================================
echo.

call venv\Scripts\activate.bat

:: Step 1 Arch
echo Choose Architecture:
echo  1. M1 (Basic U-Net)
echo  2. M2 (Self-Attention + Multi-Noise)
set m_ver=1
set /p user_ver="Enter Architecture [1-2]: "
if not "%user_ver%"=="" set m_ver=%user_ver%

:: Step 2 Name
set s_name=unet_res
set /p user_name="Enter Save Name: "
if not "%user_name%"=="" set s_name=%user_name%

:: Step 3 Mode
echo.
echo [1] Basic (Static)
echo [2] Pro (Real Noise)
echo [3] Ultra (Noise + Augment)
set m_choice=1
set /p user_choice="Choose Difficulty [1-3]: "
if not "%user_choice%"=="" set m_choice=%user_choice%

:: Step 4 Noise Stack
set n_stack=1
if "%m_ver%"=="2" goto :GET_STACK
goto :CONTINUE_TRAIN

:GET_STACK
set /p user_stack="Enter number of Noises to overlay (1-5, Default 1): "
if not "%user_stack%"=="" set n_stack=%user_stack%

:CONTINUE_TRAIN
:: Step 5 Epochs
set ep_count=50
set /p user_ep="Enter Epochs: "
if not "%user_ep%"=="" set ep_count=%user_ep%

:: Final Execution
set d_path=data
set t_script=src/train_M1.py
if "%m_ver%"=="2" set t_script=src/train_M2.py

set args=--save_name %s_name% --data_dir %d_path% --epochs %ep_count%
if "%m_choice%"=="2" set args=%args% --noise_dir %d_path%/noises
if "%m_choice%"=="3" set args=%args% --noise_dir %d_path%/noises --augment
if "%m_ver%"=="2" set args=%args% --num_noises %n_stack%

echo.
echo Launching Arch:M%m_ver% Name:%s_name% Noises:%n_stack%...
pause
python %t_script% %args%
pause

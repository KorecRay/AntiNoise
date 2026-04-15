@echo off
rem BUFFER LINE
cls
echo ===========================================
echo    TRAIN WIZARD - ASCII VERSION
echo ===========================================
echo.
call venv\Scripts\activate.bat
set d_path=data
set s_name=unet_model_v1
set /p user_name="Model Name (Enter for default): "
if not "%user_name%"=="" set s_name=%user_name%
echo.
echo Mode: 1-Basic, 2-Pro, 3-Ultra
set m_choice=1
set /p user_choice="Mode [1-3]: "
if not "%user_choice%"=="" set m_choice=%user_choice%
set ep_count=50
set /p user_ep="Epochs [1-500]: "
if not "%user_ep%"=="" set ep_count=%user_ep%
set train_args=--save_name %s_name% --data_dir %d_path% --epochs %ep_count%
if "%m_choice%"=="2" set train_args=%train_args% --noise_dir %d_path%/noises
if "%m_choice%"=="3" set train_args=%train_args% --noise_dir %d_path%/noises --augment
echo.
echo Launching: %s_name%
python src/train.py %train_args%
pause

@echo off
cls
echo ===========================================
echo    AI Voice Separation - MANUAL MIXER
echo ===========================================
echo.

:: 1. Show some samples
echo [VOCALS SAMPLES]:
dir /b data\vocals | findstr "000 001 002"
echo.
set /p v_file="Enter vocal filename (e.g. sample_0000.wav): "

echo.
echo [NOISE SAMPLES]:
dir /b data\noises | findstr "1-1002 1-115 rain street"
echo.
set /p n_file="Enter noise filename (e.g. rain_sim.wav): "

echo.
set /p snr_val="Enter SNR dB (Recommended: 0 to 15. Default: 5): "
if "%snr_val%"=="" set snr_val=5

echo.
echo Mixing... Please wait.
.\venv\Scripts\python.exe src/manual_mix.py --vocal "data/vocals/%v_file%" --noise "data/noises/%n_file%" --snr %snr_val% --out "source/manual_test_mix.wav"

echo.
echo [DONE] The mixed file is saved to: source/manual_test_mix.wav
echo You can now run separate.bat to test it!
pause

@echo off
REM ============================================================
REM  AI Avatar Chat - Installation Script for Windows
REM  Installs all dependencies for the avatar chat pipeline:
REM    - CUDA-enabled PyTorch
REM    - Gradio (web UI)
REM    - Ollama Python client (local LLM)
REM    - edge-tts (free text-to-speech)
REM    - Coqui TTS / XTTS v2 (voice cloning, optional)
REM ============================================================

echo.
echo ============================================================
echo  AI Avatar Chat - Dependency Installer
echo ============================================================
echo.

REM --- Step 0: Check NVIDIA Driver ---
echo [Step 0] Checking NVIDIA GPU driver...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] nvidia-smi not found. Install NVIDIA drivers first.
    echo Download from: https://www.nvidia.com/Download/index.aspx
    echo.
    pause
    exit /b 1
)

REM Check driver version - needs 522+ for CUDA 11.8
echo.
echo IMPORTANT: Your NVIDIA driver must be version 522.06 or newer for CUDA 11.8.
echo Current driver info:
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
echo.
echo If your driver is older than 522, update it from:
echo   https://www.nvidia.com/Download/index.aspx
echo.

REM --- Step 1: Install CUDA-enabled PyTorch ---
echo [Step 1] Installing PyTorch with CUDA 11.8 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)

REM --- Step 2: Install Gradio ---
echo.
echo [Step 2] Installing Gradio web UI...
pip install gradio>=5.0.0
if errorlevel 1 (
    echo [ERROR] Gradio installation failed.
    pause
    exit /b 1
)

REM --- Step 3: Install edge-tts (free TTS, no GPU needed) ---
echo.
echo [Step 3] Installing edge-tts (free Microsoft text-to-speech)...
pip install edge-tts
if errorlevel 1 (
    echo [WARNING] edge-tts installation failed. You can still use XTTS for voice.
)

REM --- Step 4: Install Ollama Python client ---
echo.
echo [Step 4] Installing Ollama Python client...
pip install ollama requests
if errorlevel 1 (
    echo [WARNING] Ollama client installation failed.
)

REM --- Step 5: Install MuseTalk dependencies ---
echo.
echo [Step 5] Installing MuseTalk dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some MuseTalk dependencies may have failed.
)

REM --- Step 6: Install additional audio dependencies ---
echo.
echo [Step 6] Installing audio processing dependencies...
pip install soundfile
if errorlevel 1 (
    echo [WARNING] soundfile installation failed.
)

REM --- Step 7 (Optional): Install XTTS for voice cloning ---
echo.
echo [Step 7] OPTIONAL: Install Coqui TTS for voice cloning?
echo This requires ~2GB disk space and Python 3.11 or lower.
echo.
set /p INSTALL_XTTS="Install XTTS voice cloning? (y/n): "
if /i "%INSTALL_XTTS%"=="y" (
    echo Installing Coqui TTS...
    pip install TTS
    if errorlevel 1 (
        echo [WARNING] XTTS installation failed. Voice cloning will not be available.
        echo You can still use edge-tts for text-to-speech.
    )
)

echo.
echo ============================================================
echo  Installation complete! Next steps:
echo ============================================================
echo.
echo  1. UPDATE NVIDIA DRIVER (if older than 522):
echo     https://www.nvidia.com/Download/index.aspx
echo.
echo  2. INSTALL OLLAMA (local AI brain):
echo     Download from: https://ollama.com
echo     Then run:  ollama pull llama3.1:8b
echo.
echo  3. DOWNLOAD MUSETALK MODELS (if not done):
echo     Run: download_weights.bat
echo.
echo  4. START THE APP:
echo     python avatar_chat.py
echo.
echo ============================================================
pause

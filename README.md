# AI Avatar Chat

https://github.com/user-attachments/assets/e71a46c4-7090-4557-907c-229fc51c92be

This documentation details a **fully local AI software** designed to create and interact with **personalized digital avatars**. By integrating several specialized engines, the program allows users to generate a **lip-synced video clone** that mimics their appearance and voice. The system utilizes **Ollama** for conversational intelligence and **MuseTalk** for realistic facial animation, ensuring the experience remains private and free of cloud fees. Users can input a single photo and an optional audio sample to build a **custom character** capable of responding to both text and speech. Optimized for consumer hardware with at least **6GB of VRAM**, the project provides a comprehensive guide for installation, resource management, and fine-tuning the avatar’s behavior. Ultimately, the repository serves as a technical blueprint for running a **real-time virtual assistant** entirely on a personal computer.

<img width="1185" height="633" alt="2" src="https://github.com/user-attachments/assets/6736a5bc-5aee-4adf-aaba-2b14599efd76" />

## How It Works

```
You speak/type ──> Whisper STT ──> Ollama LLM ──> edge-tts / XTTS ──> MuseTalk lip-sync ──> Avatar video
```

| Component | Purpose | Runs On |
|-----------|---------|---------|
| **Whisper** | Speech-to-text (voice input) | GPU |
| **Ollama** (llama3.1:8b) | AI conversation | CPU (uses RAM) |
| **edge-tts** | Text-to-speech (free, no cloning) | CPU (needs internet) |
| **XTTS v2** | Voice cloning (optional) | GPU (~2GB, loads/unloads) |
| **MuseTalk** | Lip-sync video generation | GPU (~3-4GB fp16) |

Designed for **6GB VRAM** GPUs. Ollama runs entirely on CPU/RAM so MuseTalk has the GPU to itself.

## Features

<img width="1166" height="612" alt="image" src="https://github.com/user-attachments/assets/c3e0918a-f4f6-4d04-b740-6714bef511a3" />

- **Text or voice input** — type a message or speak into your microphone
- **AI personality** — describe who the avatar should act as, and it stays in character
- **Lip-synced video output** — the avatar's mouth moves naturally with the spoken response
- **Voice cloning** (optional) — clone your voice from a short audio sample using XTTS v2
- **Downloadable results** — browse and download all generated videos from the Gallery tab
- **Fully local and free** — no cloud APIs, no subscriptions, everything runs on your machine

## Requirements

<img width="1162" height="620" alt="image" src="https://github.com/user-attachments/assets/c060ac3f-8173-4293-afb5-ea21345429af" />

- **OS**: Windows 10/11 (Linux also supported)
- **GPU**: NVIDIA with 6GB+ VRAM (RTX 3060 or better recommended)
- **RAM**: 16GB minimum, 32GB recommended (Ollama LLM runs on CPU)
- **NVIDIA Driver**: Version 522+ (for CUDA 11.8)
- **Python**: 3.10-3.11 (3.11 required for XTTS voice cloning)
- **Disk**: ~15GB for models and dependencies
- **FFmpeg**: Must be installed and on PATH

## Installation

### Step 1: Update NVIDIA Driver

Your driver must be version **522 or newer** for CUDA 11.8. Check with:

```bash
nvidia-smi
```

If older, download the latest from: https://www.nvidia.com/Download/index.aspx

### Step 2: Install Dependencies

Run the installer script:

```bash
install_avatar.bat
```

This installs:
- PyTorch with CUDA 11.8
- Gradio (web UI)
- edge-tts (free text-to-speech)
- Ollama Python client
- MuseTalk dependencies
- XTTS v2 (optional, for voice cloning)

### Step 3: Install Ollama

Download and install from https://ollama.com, then:

```bash
# Terminal 1: Start the Ollama server
ollama serve

# Terminal 2: Download the language model (~4.7GB)
ollama pull llama3.1:8b
```

Leave `ollama serve` running whenever you use the app.

### Step 4: Download MuseTalk Models

```bash
download_weights.bat
```

This downloads face animation models (~3GB) from HuggingFace. The models are organized under `./models/`:

```
models/
  musetalkV15/     # UNet + config (lip-sync core)
  sd-vae/          # VAE encoder/decoder
  whisper/         # Audio feature extraction
  dwpose/          # Facial landmark detection
  face-parse-bisent/  # Face segmentation for blending
  syncnet/         # Audio-visual sync (training only)
```

### Step 5: Install FFmpeg

Download from https://github.com/BtbN/FFmpeg-Builds/releases and add the `bin` folder to your system PATH. Verify:

```bash
ffmpeg -version
```

### Step 6: Install MMLab Packages

```bash
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"
```

## Usage

<img width="1202" height="657" alt="image" src="https://github.com/user-attachments/assets/36bd3e32-cdb7-443e-b9de-e58a88c288bf" />

### Start the App

<img width="1217" height="668" alt="image" src="https://github.com/user-attachments/assets/a67eac9a-f8f8-44e3-9260-e9e9d1899131" />

Make sure Ollama is running (`ollama serve`), then:

```bash
python avatar_chat.py --use_float16
```

Open http://127.0.0.1:7861 in your browser.

### Command-Line Options

<img width="1201" height="661" alt="image" src="https://github.com/user-attachments/assets/a9f69820-a403-49b5-b041-519f831b0ea5" />

| Flag | Description |
|------|-------------|
| `--use_float16` | Use fp16 inference (recommended for 6GB GPUs) |
| `--share` | Create a public Gradio link |
| `--host 0.0.0.0` | Listen on all interfaces |
| `--port 7861` | Server port (default: 7861) |

### Using the Web UI

**Tab 1 — Setup:**
1. Upload a clear, front-facing photo of yourself
2. (Optional) Upload a short voice sample (.wav) for voice cloning
3. Write a personality description (e.g., "I'm a software engineer who loves hiking")
4. Choose language and TTS engine
5. Click **Initialize Avatar** — wait for processing to complete

**Tab 2 — Chat:**
1. Type a message or click the microphone to speak
2. Click **Send** (or press Enter)
3. The pipeline runs: STT -> LLM -> TTS -> Lip-sync -> Video
4. Watch the avatar respond with lip-synced video

**Tab 3 — Gallery:**
- Browse and download all generated videos

### Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| **BBox Shift** | 0 | Adjusts mouth openness. Positive = more open, negative = less open |
| **Extra Margin** | 10 | Padding around the detected face region |
| **Language** | en | Affects TTS voice selection and XTTS language |
| **Ollama Model** | llama3.1:8b | Any Ollama model (e.g., `mistral`, `gemma2:9b`) |

## Architecture

The system uses five engine classes orchestrated by `AvatarChat`:

<img width="1152" height="621" alt="image" src="https://github.com/user-attachments/assets/94de4270-e62c-4fa6-99cb-74ff54aefd87" />

```
avatar_chat.py
  STTEngine     — Whisper speech-to-text (lazy-loaded)
  LLMEngine     — Ollama local LLM (CPU, HTTP API)
  TTSEngine     — edge-tts (default) or XTTS v2 (voice cloning)
  AvatarEngine  — MuseTalk lip-sync (VAE + UNet + face parsing)
  AvatarChat    — Orchestrator connecting all engines
  build_ui()    — Gradio 3-tab interface
```

### VRAM Management Strategy

With only 6GB VRAM, models are loaded/unloaded sequentially:

1. **STT Whisper** loads on first voice input, stays resident (~500MB)
2. **XTTS** loads for TTS, unloads immediately after generating speech (~2GB)
3. **MuseTalk** (VAE + UNet + face parsing) stays resident for video generation (~3-4GB fp16)
4. **Ollama** runs entirely on CPU using system RAM (~8GB RAM for 8B model)

## Original MuseTalk

<img width="1172" height="628" alt="image" src="https://github.com/user-attachments/assets/dba956e4-208a-4e16-bd77-bef0fb9c3a9b" />

This project extends [MuseTalk](https://github.com/TMElyralab/MuseTalk), a real-time high-fidelity lip-syncing model by Lyra Lab (Tencent Music Entertainment).

The original MuseTalk tools are still available:

```bash
# Gradio demo (original lip-sync UI)
python app.py --use_float16

# Batch inference from config
python -m scripts.inference --inference_config configs/inference/test.yaml

# Real-time inference
python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml
```

For full details on the original MuseTalk, see the [technical report](https://arxiv.org/abs/2410.10122).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi` not found | Install NVIDIA GPU drivers |
| CUDA out of memory | Use `--use_float16`, close other GPU apps |
| Ollama connection error | Run `ollama serve` in a separate terminal |
| FFmpeg not found | Install FFmpeg and add `bin` to PATH |
| Face not detected | Use a clear, front-facing photo with good lighting |
| Mouth looks wrong | Adjust **BBox Shift** (try -5 to +5) |
| No audio output | Check FFmpeg installation, ensure edge-tts has internet |
| XTTS not available | Requires Python 3.11 or lower, install via `pip install TTS` |
| Disk space error | Need ~15GB free; move project to a larger drive if needed |

## Project Structure

```
MuseTalk/
  avatar_chat.py          # AI Avatar Chat application
  app.py                  # Original MuseTalk Gradio demo
  install_avatar.bat      # Dependency installer
  download_weights.bat    # Model weight downloader
  requirements.txt        # Python dependencies
  models/                 # Pre-trained model weights
  results/                # Generated outputs
    avatar_chat/          # Avatar chat videos
  musetalk/
    models/               # UNet, VAE model definitions
    utils/                # Audio, face detection, blending
    data/                 # Dataset handling (training)
    loss/                 # Loss functions (training)
  scripts/
    inference.py          # Batch inference
    realtime_inference.py # Real-time inference
    preprocess.py         # Data preprocessing
  configs/                # YAML configuration files
  train.py                # Training script
```

## License

- **Code**: MIT License (no limitation for academic or commercial use)
- **MuseTalk model weights**: Available for any purpose, including commercial
- **Third-party models** (Whisper, VAE, DWPose, etc.): Subject to their respective licenses

## Citation

```bib
@article{musetalk,
  title={MuseTalk: Real-Time High-Fidelity Video Dubbing via Spatio-Temporal Sampling},
  author={Zhang, Yue and Zhong, Zhizhou and Liu, Minhao and Chen, Zhaokang and Wu, Bin and Zeng, Yubin and Zhan, Chao and He, Yingjie and Huang, Junxin and Zhou, Wenjiang},
  journal={arxiv},
  year={2025}
}
```

## Acknowledgement

- [MuseTalk](https://github.com/TMElyralab/MuseTalk) — Lyra Lab, Tencent Music Entertainment
- [Ollama](https://ollama.com) — Local LLM runtime
- [edge-tts](https://github.com/rany2/edge-tts) — Free Microsoft TTS
- [Coqui TTS / XTTS v2](https://github.com/coqui-ai/TTS) — Voice cloning
- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition and audio features
- [Gradio](https://gradio.app) — Web UI framework

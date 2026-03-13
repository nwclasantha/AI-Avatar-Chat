"""
AI Avatar Chat — Built on MuseTalk
===================================
Talk to your AI avatar clone via text or voice.
Fully local: Ollama (LLM) + edge-tts/XTTS (voice) + MuseTalk (lip-sync)

Architecture (designed for 6GB VRAM):
  - Ollama LLM runs on CPU (uses RAM, not GPU)
  - edge-tts runs on CPU (free Microsoft TTS, no cloning)
  - XTTS voice cloning runs on GPU briefly (~2GB), then unloads
  - MuseTalk runs on GPU (~3-4GB fp16)

Usage:
  python avatar_chat.py
  python avatar_chat.py --share          # public link
  python avatar_chat.py --use_float16    # faster on GPU
"""

import argparse
import asyncio
import copy
import os
import subprocess
import sys
import tempfile
import time
import uuid

import cv2
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "avatar_chat")
AVATARS_DIR = os.path.join(PROJECT_DIR, "results", "avatars_chat")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AVATARS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Speech-to-Text Engine (Whisper, for voice input mode)
# ──────────────────────────────────────────────────────────────
class STTEngine:
    """Transcribe voice input to text using Whisper."""

    def __init__(self):
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline

            device_id = 0 if torch.cuda.is_available() else -1
            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=os.path.join(MODELS_DIR, "whisper"),
                device=device_id,
            )
        except Exception:
            # Fallback: use openai/whisper-tiny from HuggingFace Hub
            from transformers import pipeline as hf_pipeline

            device_id = 0 if torch.cuda.is_available() else -1
            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=device_id,
            )

    def transcribe(self, audio_path):
        """Return transcribed text from an audio file."""
        if not audio_path or not os.path.isfile(audio_path):
            return ""
        self._load()
        result = self._pipeline(audio_path)
        return result.get("text", "").strip()


# ──────────────────────────────────────────────────────────────
# LLM Engine (Ollama — runs on CPU, zero VRAM)
# ──────────────────────────────────────────────────────────────
class LLMEngine:
    """Generate conversational responses using a local Ollama model."""

    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        self.personality = ""
        self.history = []

    def set_personality(self, description):
        self.personality = description
        self.history = []

    def generate(self, user_message):
        """Send message to Ollama and return the assistant response."""
        import requests

        system_prompt = (
            "You are roleplaying as the following person. "
            "Stay in character at all times. Respond naturally as this person would. "
            "Keep responses concise (2-4 sentences) so they sound like natural speech.\n\n"
            f"Character:\n{self.personality}"
        )

        # Build messages with the candidate user message (don't mutate history yet)
        candidate_messages = (
            [{"role": "system", "content": system_prompt}]
            + self.history
            + [{"role": "user", "content": user_message}]
        )

        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model_name,
                    "messages": candidate_messages,
                    "stream": False,
                    "options": {"num_ctx": 4096, "temperature": 0.7},
                },
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"]
        except requests.ConnectionError:
            return "[Ollama is not running. Start it: ollama serve]"
        except Exception as e:
            return f"[LLM error: {e}]"

        # Only mutate history after a successful response
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": text})
        return text

    def clear_history(self):
        self.history = []


# ──────────────────────────────────────────────────────────────
# TTS Engine (edge-tts default, XTTS optional for voice cloning)
# ──────────────────────────────────────────────────────────────
EDGE_VOICE_MAP = {
    "en": "en-US-GuyNeural",
    "zh": "zh-CN-YunxiNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "pt": "pt-BR-AntonioNeural",
    "ru": "ru-RU-DmitryNeural",
    "ar": "ar-SA-HamedNeural",
    "hi": "hi-IN-MadhurNeural",
}


class TTSEngine:
    """Text-to-speech with optional voice cloning."""

    def __init__(self):
        self.mode = None  # "edge" or "xtts"
        self.edge_voice = "en-US-GuyNeural"
        self.voice_sample = None
        self._xtts = None

    def setup_edge(self, language="en"):
        self.mode = "edge"
        self.edge_voice = EDGE_VOICE_MAP.get(language, "en-US-GuyNeural")

    def setup_xtts(self, voice_sample_path, language="en"):
        self.mode = "xtts"
        self.voice_sample = voice_sample_path

    # ---- edge-tts (CPU, free, needs internet) ----------------

    def _speak_edge(self, text, wav_path):
        import edge_tts

        mp3_path = wav_path + ".mp3"

        async def _gen():
            comm = edge_tts.Communicate(text, self.edge_voice)
            await comm.save(mp3_path)

        # Handle nested event loops (Gradio runs its own async loop)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Cannot call asyncio.run() inside a running loop.
            # Spawn a new event loop in a separate thread.
            import concurrent.futures

            def _run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(_gen())
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(None)  # prevent stale closed-loop ref

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(_run_in_new_loop).result()
        else:
            asyncio.run(_gen())

        # Convert MP3 to WAV 16kHz mono (MuseTalk expects this)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-v", "fatal", "-i", mp3_path,
                 "-ar", "16000", "-ac", "1", wav_path],
                check=True,
            )
        finally:
            # Always clean up the intermediate MP3
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
        return wav_path

    # ---- XTTS v2 (GPU, voice cloning) -----------------------

    def _load_xtts(self):
        if self._xtts is None:
            from TTS.api import TTS
            self._xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            if torch.cuda.is_available():
                self._xtts = self._xtts.to("cuda")

    def _unload_xtts(self):
        if self._xtts is not None:
            del self._xtts
            self._xtts = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _speak_xtts(self, text, wav_path, language="en"):
        self._load_xtts()
        try:
            self._xtts.tts_to_file(
                text=text,
                speaker_wav=self.voice_sample,
                language=language,
                file_path=wav_path,
            )
        finally:
            self._unload_xtts()  # Free VRAM for MuseTalk
        return wav_path

    # ---- Public API ------------------------------------------

    def speak(self, text, wav_path, language="en"):
        """Convert text to speech, return path to wav file."""
        if self.mode == "xtts" and self.voice_sample:
            return self._speak_xtts(text, wav_path, language)
        return self._speak_edge(text, wav_path)


# ──────────────────────────────────────────────────────────────
# MuseTalk Avatar Engine (lip-sync)
# ──────────────────────────────────────────────────────────────
class AvatarEngine:
    """Wraps MuseTalk models for single-photo avatar lip-sync."""

    def __init__(self, use_float16=False):
        self.use_float16 = use_float16 and torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = torch.float16 if self.use_float16 else torch.float32

        # Models (lazy-loaded)
        self._vae = None
        self._unet = None
        self._pe = None
        self._whisper = None
        self._audio_proc = None
        self._fp = None
        self._timesteps = None

        # Cached avatar data
        self.ready = False
        self._photo_path = None
        self._ori_frame = None
        self._bbox = None
        self._latent = None
        self._mask = None
        self._crop_box = None

    def _load_models(self):
        """Load all MuseTalk models once."""
        if self._vae is not None:
            return

        from musetalk.utils.utils import load_all_model
        from musetalk.utils.audio_processor import AudioProcessor
        from transformers import WhisperModel

        # load_all_model uses CWD-relative paths internally (e.g. "models/sd-vae").
        # Ensure CWD is the project root so these resolve correctly.
        os.chdir(PROJECT_DIR)

        print("[AvatarEngine] Loading models...")
        self._vae, self._unet, self._pe = load_all_model(
            unet_model_path=os.path.join(MODELS_DIR, "musetalkV15", "unet.pth"),
            vae_type="sd-vae",
            unet_config=os.path.join(MODELS_DIR, "musetalkV15", "musetalk.json"),
            device=self.device,
        )

        if self.use_float16:
            self._pe = self._pe.half()
            self._vae.vae = self._vae.vae.half()
            self._unet.model = self._unet.model.half()

        self._pe = self._pe.to(self.device)
        self._vae.vae = self._vae.vae.to(self.device)
        self._unet.model = self._unet.model.to(self.device)

        self._timesteps = torch.tensor([0], device=self.device)

        self._audio_proc = AudioProcessor(
            feature_extractor_path=os.path.join(MODELS_DIR, "whisper")
        )
        self._whisper = WhisperModel.from_pretrained(
            os.path.join(MODELS_DIR, "whisper")
        )
        self._whisper = self._whisper.to(device=self.device, dtype=self.weight_dtype)
        self._whisper.requires_grad_(False)

        print("[AvatarEngine] Models loaded.")

    def _load_face_parser(self, left_cheek=90, right_cheek=90, parsing_mode="jaw"):
        # Re-create if parameters changed since last call
        params_changed = (
            self._fp is not None
            and (
                getattr(self, "_fp_left_cheek", None) != left_cheek
                or getattr(self, "_fp_right_cheek", None) != right_cheek
            )
        )
        if self._fp is None or params_changed:
            from musetalk.utils.face_parsing import FaceParsing
            self._fp = FaceParsing(
                left_cheek_width=left_cheek,
                right_cheek_width=right_cheek,
            )
            self._fp_left_cheek = left_cheek
            self._fp_right_cheek = right_cheek
        self._parsing_mode = parsing_mode

    def prepare_avatar(self, photo_path, bbox_shift=0, extra_margin=10):
        """Detect face in photo, encode latent, pre-compute blend mask."""
        self._load_models()
        self._load_face_parser()

        from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
        from musetalk.utils.blending import get_image_prepare_material

        # Read image (force 3-channel BGR — strips alpha from PNGs)
        img = cv2.imread(photo_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {photo_path}")

        # Save a temp file for landmark detection (expects file paths)
        tmp_path = os.path.join(AVATARS_DIR, "avatar_source.png")
        cv2.imwrite(tmp_path, img)

        coord_list, frame_list = get_landmark_and_bbox([tmp_path], bbox_shift)

        if not coord_list or not frame_list:
            raise ValueError("No face detected in photo.")

        bbox = coord_list[0]
        frame = frame_list[0]

        if bbox == coord_placeholder:
            raise ValueError("No face detected. Try adjusting bbox_shift.")

        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, frame.shape[1])
        y2 = y2 + extra_margin
        y2 = min(y2, frame.shape[0])

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Detected face region is too small.")

        # Ensure plain Python ints (not numpy int types) for PIL compatibility
        self._bbox = [int(x1), int(y1), int(x2), int(y2)]
        self._ori_frame = frame
        self._photo_path = photo_path

        # Crop face and encode to VAE latent
        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        self._latent = self._vae.get_latents_for_unet(crop_resized)

        # Pre-compute blending mask
        self._mask, self._crop_box = get_image_prepare_material(
            frame, self._bbox, fp=self._fp, mode=self._parsing_mode,
        )

        self.ready = True
        print("[AvatarEngine] Avatar prepared.")

    @torch.no_grad()
    def generate_video(self, audio_path, output_path, fps=25, batch_size=4):
        """Generate lip-synced video from audio + cached avatar."""
        if not self.ready:
            raise RuntimeError("Avatar not prepared. Call prepare_avatar() first.")

        from musetalk.utils.utils import datagen
        from musetalk.utils.blending import get_image_blending

        # 1. Extract audio features
        audio_result = self._audio_proc.get_audio_feature(audio_path)
        if audio_result is None:
            raise FileNotFoundError(f"Cannot load audio: {audio_path}")

        whisper_input_features, librosa_length = audio_result
        whisper_chunks = self._audio_proc.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self._whisper,
            librosa_length,
            fps=fps,
        )

        if not whisper_chunks:
            raise ValueError("Audio too short or silent — no whisper chunks generated.")

        # 2. Create latent list (single photo = same latent for every frame)
        # datagen cycles through latents via modular index
        latent_list = [self._latent]

        # 3. Generate frames
        gen = datagen(whisper_chunks, latent_list, batch_size=batch_size, device=self.device)

        res_frames = []
        x1, y1, x2, y2 = self._bbox

        for whisper_batch, latent_batch in gen:
            audio_feat = self._pe(whisper_batch.to(dtype=self.weight_dtype))
            latent_batch = latent_batch.to(dtype=self._unet.model.dtype)

            pred = self._unet.model(
                latent_batch,
                self._timesteps,
                encoder_hidden_states=audio_feat,
            ).sample

            pred = pred.to(dtype=self._vae.vae.dtype)
            decoded_frames = self._vae.decode_latents(pred)

            for res_frame in decoded_frames:
                ori = copy.deepcopy(self._ori_frame)
                try:
                    face = cv2.resize(
                        res_frame.astype(np.uint8), (x2 - x1, y2 - y1)
                    )
                except Exception:
                    continue

                blended = get_image_blending(
                    ori, face, self._bbox, self._mask, self._crop_box
                )
                res_frames.append(blended)

        if not res_frames:
            raise RuntimeError("No frames were generated.")

        # 4. Write video (frames -> temp mp4 -> combine with audio)
        h, w = res_frames[0].shape[:2]
        tmp_video = output_path + ".tmp.mp4"
        writer = cv2.VideoWriter(
            tmp_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for f in res_frames:
            writer.write(f)
        writer.release()

        # 5. Mux with audio via ffmpeg
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "fatal",
                    "-i", tmp_video,
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-shortest",
                    output_path,
                ],
                check=True,
            )
        finally:
            # Always clean up tmp_video, even if ffmpeg fails
            if os.path.exists(tmp_video):
                os.remove(tmp_video)

        return output_path


# ──────────────────────────────────────────────────────────────
# Avatar Chat Pipeline (orchestrator)
# ──────────────────────────────────────────────────────────────
class AvatarChat:
    """Connects LLM + TTS + MuseTalk into one pipeline."""

    def __init__(self, use_float16=False):
        self.stt = STTEngine()
        self.llm = LLMEngine()
        self.tts = TTSEngine()
        self.avatar = AvatarEngine(use_float16=use_float16)
        self.language = "en"

    def setup(self, photo_path, voice_sample, personality, language,
              tts_engine, bbox_shift, ollama_model, extra_margin=10):
        """One-time avatar setup. Returns status string."""
        status = []

        # LLM
        self.llm.model_name = ollama_model
        self.llm.set_personality(personality)
        self.language = language
        status.append("[OK] Personality configured")

        # TTS
        if tts_engine == "xtts" and voice_sample:
            self.tts.setup_xtts(voice_sample, language)
            status.append("[OK] Voice cloning ready (XTTS v2)")
        else:
            self.tts.setup_edge(language)
            voice_name = EDGE_VOICE_MAP.get(language, "en-US-GuyNeural")
            status.append(f"[OK] TTS ready (edge-tts: {voice_name})")

        # Avatar face
        try:
            self.avatar.prepare_avatar(
                photo_path, bbox_shift=bbox_shift, extra_margin=extra_margin,
            )
            status.append("[OK] Face detected and VAE latent encoded")
        except Exception as e:
            status.append(f"[FAIL] Face detection: {e}")
            return "\n".join(status)

        status.append("\nAvatar ready! Switch to the Chat tab.")
        return "\n".join(status)

    def chat(self, user_text, user_audio=None):
        """
        Full pipeline: input -> LLM -> TTS -> lip-sync -> video.
        Returns (response_text, video_path_or_None).
        """
        if not self.avatar.ready:
            return "Set up your avatar first (Setup tab).", None

        # 1. Speech-to-text if voice input
        if user_audio and not user_text:
            user_text = self.stt.transcribe(user_audio)
            if not user_text:
                return "Could not understand audio. Please try again.", None

        if not user_text or not user_text.strip():
            return "Please enter a message.", None

        # 2. LLM generates response
        resp_text = self.llm.generate(user_text.strip())
        if resp_text.startswith("["):
            return resp_text, None

        # 3. TTS — convert response to speech
        uid = uuid.uuid4().hex[:8]
        wav_path = os.path.join(RESULTS_DIR, f"speech_{uid}.wav")
        try:
            self.tts.speak(resp_text, wav_path, self.language)
        except Exception as e:
            return f"TTS error: {e}", None

        # 4. MuseTalk — lip-sync avatar to speech
        video_path = os.path.join(RESULTS_DIR, f"avatar_{uid}.mp4")
        try:
            self.avatar.generate_video(wav_path, video_path)
        except Exception as e:
            return f"Lip-sync error: {e}", None
        finally:
            # Clean up intermediate WAV (audio is muxed into the video)
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        return resp_text, video_path

    def get_gallery(self):
        """Return list of generated video paths, newest first."""
        from pathlib import Path
        videos = sorted(
            Path(RESULTS_DIR).glob("avatar_*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [str(v) for v in videos[:50]]

    def clear(self):
        self.llm.clear_history()


# ──────────────────────────────────────────────────────────────
# Dependency checks (non-blocking)
# ──────────────────────────────────────────────────────────────
def check_dependencies():
    """Return a list of status strings for the UI."""
    lines = []

    # GPU
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        lines.append(f"- GPU: {name} ({mem:.1f} GB)")
    else:
        lines.append("- GPU: **Not available** (CPU mode will be slow)")

    # Ollama
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            lines.append(f"- Ollama: Running ({len(models)} model(s))")
        else:
            lines.append("- Ollama: Running but returned unexpected status")
    except Exception:
        lines.append(
            "- Ollama: **Not running** — install from "
            "[ollama.com](https://ollama.com), then run "
            "`ollama serve` and `ollama pull llama3.1:8b`"
        )

    # edge-tts
    try:
        import edge_tts  # noqa: F401
        lines.append("- edge-tts: Installed")
    except ImportError:
        lines.append("- edge-tts: **Not installed** (`pip install edge-tts`)")

    # XTTS
    try:
        import TTS  # noqa: F401
        lines.append("- XTTS voice cloning: Available")
    except ImportError:
        lines.append("- XTTS voice cloning: Not installed (optional)")

    # ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        lines.append("- FFmpeg: Installed")
    except (FileNotFoundError, subprocess.SubprocessError):
        lines.append("- FFmpeg: **Not found** (required for audio/video)")

    return lines


# ──────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────
def build_ui(app_state):
    import gradio as gr

    dep_lines = check_dependencies()

    # Discover available Ollama models
    ollama_models = ["llama3.1:8b"]
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            found = [m["name"] for m in r.json().get("models", [])]
            if found:
                ollama_models = found
    except Exception:
        pass

    with gr.Blocks(
        title="AI Avatar Chat",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# AI Avatar Chat\nTalk to your AI clone — fully local, fully free")
        gr.Markdown("\n".join(dep_lines))

        # ──── Tab 1: Setup ────────────────────────────────────
        with gr.Tab("1. Setup Avatar"):
            with gr.Row():
                with gr.Column(scale=1):
                    photo_input = gr.Image(
                        label="Your Photo (clear front-facing)",
                        type="filepath",
                    )
                    voice_input = gr.Audio(
                        label="Voice Sample 30+ sec (optional for edge-tts)",
                        type="filepath",
                    )
                with gr.Column(scale=1):
                    personality_input = gr.Textbox(
                        label="Who are you? (personality and background)",
                        placeholder=(
                            "I'm a software engineer who loves Python and AI. "
                            "I speak casually, make tech jokes, and I'm always "
                            "happy to help explain complex topics simply."
                        ),
                        lines=5,
                    )
                    language_input = gr.Dropdown(
                        list(EDGE_VOICE_MAP.keys()),
                        label="Language",
                        value="en",
                    )
                    ollama_model_input = gr.Dropdown(
                        ollama_models,
                        label="Ollama Model",
                        value=ollama_models[0],
                        allow_custom_value=True,
                    )
                    tts_choice = gr.Radio(
                        ["edge-tts (free, no cloning)", "XTTS v2 (voice cloning)"],
                        label="Voice Engine",
                        value="edge-tts (free, no cloning)",
                    )
                    bbox_shift = gr.Slider(
                        -20, 20, value=0, step=1,
                        label="Face bbox shift (adjust if lip area misaligned)",
                    )
                    extra_margin_input = gr.Slider(
                        0, 30, value=10, step=1,
                        label="Extra margin below chin",
                    )
                    setup_btn = gr.Button(
                        "Create Avatar", variant="primary", size="lg",
                    )

            setup_status = gr.Textbox(
                label="Status", interactive=False, lines=6,
            )

        # ──── Tab 2: Chat ─────────────────────────────────────
        with gr.Tab("2. Chat"):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=350,
                type="messages",
            )
            with gr.Row():
                text_input = gr.Textbox(
                    label="Type your message",
                    placeholder="Ask me anything...",
                    scale=4,
                )
                audio_input = gr.Audio(
                    label="Or speak",
                    sources=["microphone"],
                    type="filepath",
                    scale=1,
                )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")

            video_output = gr.Video(label="Avatar Response", height=400)
            response_text_box = gr.Textbox(
                label="Response Text", interactive=False,
            )

        # ──── Tab 3: Gallery ──────────────────────────────────
        with gr.Tab("3. Gallery"):
            gallery_files = gr.File(
                label="Generated Videos", file_count="multiple",
            )
            refresh_btn = gr.Button("Refresh Gallery")

        # ──── Event Handlers ──────────────────────────────────

        def on_setup(photo, voice, personality, lang, tts_eng, shift, margin, model):
            if photo is None:
                return "Please upload a photo."
            tts_type = "xtts" if "XTTS" in str(tts_eng) else "edge"
            return app_state.setup(
                photo_path=photo,
                voice_sample=voice,
                personality=personality,
                language=lang,
                tts_engine=tts_type,
                bbox_shift=int(shift),
                ollama_model=model,
                extra_margin=int(margin),
            )

        def on_send(message, audio, history):
            if not message and not audio:
                return history or [], None, "", ""

            # Transcribe voice input so the actual words appear in chat history
            display_msg = message
            if not message and audio:
                display_msg = app_state.stt.transcribe(audio)
                if not display_msg:
                    # Short-circuit: don't send a sentinel string to the LLM
                    return (
                        history or [],
                        None,
                        "Could not transcribe audio. Please try again.",
                        "",
                    )

            history = history or []
            history.append({"role": "user", "content": display_msg})

            # Pass the transcribed text directly (STT already done)
            resp_text, video_path = app_state.chat(display_msg, None)
            history.append({"role": "assistant", "content": resp_text})

            return history, video_path, resp_text, ""

        def on_clear():
            app_state.clear()
            return [], None, ""

        def on_refresh():
            return app_state.get_gallery()

        # Wire up events
        setup_btn.click(
            on_setup,
            inputs=[
                photo_input, voice_input, personality_input,
                language_input, tts_choice, bbox_shift,
                extra_margin_input, ollama_model_input,
            ],
            outputs=[setup_status],
        )

        send_btn.click(
            on_send,
            inputs=[text_input, audio_input, chatbot],
            outputs=[chatbot, video_output, response_text_box, text_input],
        )
        text_input.submit(
            on_send,
            inputs=[text_input, audio_input, chatbot],
            outputs=[chatbot, video_output, response_text_box, text_input],
        )

        clear_btn.click(
            on_clear,
            outputs=[chatbot, video_output, response_text_box],
        )
        refresh_btn.click(on_refresh, outputs=[gallery_files])

    return demo


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI Avatar Chat")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7861, help="Server port")
    parser.add_argument(
        "--share", action="store_true", help="Create public Gradio link",
    )
    parser.add_argument(
        "--use_float16", action="store_true",
        help="Use fp16 for faster GPU inference",
    )
    args = parser.parse_args()

    app_state = AvatarChat(use_float16=args.use_float16)
    demo = build_ui(app_state)
    # Limit to 1 concurrent request — AvatarEngine is not thread-safe
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

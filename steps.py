"""
Individual pipeline steps. Each function is independent and can be tested standalone.

Pipeline stages:
  1. Audio I/O       (load_audio / record_audio / save_audio)
  2. ASR             (transcribe_chinese)   — faster-whisper
  3. Translation     (translate_to_english) — ollama (preferred) or argostranslate
  4. TTS + cloning   (synthesize_english)   — F5-TTS (zero-shot voice cloning)
  5. Speed matching  (time_stretch_to_duration)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Preload cudnn so that F5-TTS / triton find it at startup.
# On Windows, triton tries to load cudnn64_9.dll from PATH; the venv's
# torch\lib dir isn't on PATH by default. Adding it here fixes the
# "Could not load symbol cudnnGetLibConfig. Error code 127" failure.
# ---------------------------------------------------------------------------
def _preload_cudnn() -> None:
    try:
        import ctypes
        torch_lib = os.path.join(
            os.path.dirname(__import__("torch").__file__), "lib"
        )
        if torch_lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        for dll in ("cudnn64_9.dll", "cudnn_graph64_9.dll"):
            p = os.path.join(torch_lib, dll)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p)
                except OSError:
                    pass
    except Exception:
        # If torch isn't installed yet (e.g. during a clean install), skip silently.
        pass


_preload_cudnn()


# =============================================================================
# Utilities
# =============================================================================

def has_cuda() -> bool:
    """Return True if a CUDA-capable GPU + PyTorch are available."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def trim_silence(audio: np.ndarray, top_db: float = 20.0) -> np.ndarray:
    """Trim leading/trailing silence from an audio array."""
    import librosa
    if len(audio) < 2048:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


# =============================================================================
# 1. Audio I/O
# =============================================================================

def load_audio(path: str, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    """
    Load an audio file, resample to mono float32 at target_sr.
    Returns (audio, sample_rate).

    Supports wav/flac/ogg natively via soundfile, and m4a/mp4/aac/mp3
    via ffmpeg fallback (uses bundled ffmpeg from imageio-ffmpeg).
    """
    import os
    import librosa
    import soundfile as sf

    path_lower = path.lower()
    needs_ffmpeg = path_lower.endswith((".m4a", ".mp4", ".aac", ".mp3", ".wma", ".opus"))

    # Try soundfile first (fast path for wav/flac/ogg)
    if not needs_ffmpeg:
        try:
            audio, sr = sf.read(path, always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            return audio, int(sr)
        except Exception:
            pass  # fall through to ffmpeg

    # Fallback: decode with ffmpeg then load as wav
    try:
        import imageio_ffmpeg
        import subprocess
    except ImportError:
        raise RuntimeError(
            f"Cannot read {path}: install ffmpeg or imageio-ffmpeg (pip install imageio-ffmpeg)"
        )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        r = subprocess.run(
            [ffmpeg, "-y", "-i", path, "-ac", "1", "-ar", str(target_sr), tmp_path],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr[-300:]}")
        audio, sr = sf.read(tmp_path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), int(sr)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def save_audio(audio: np.ndarray, path: str, sample_rate: int = 24000) -> None:
    """Save a numpy audio array to a wav file."""
    import soundfile as sf
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)
    print(f"  [saved] {path}")


def record_audio(sample_rate: int = 24000) -> np.ndarray:
    """Record from default microphone. Press ENTER to start, ENTER again to stop."""
    import sounddevice as sd

    print()
    print("  Press ENTER to START recording...")
    input()
    print("  🎤 Recording... press ENTER to STOP.")

    frames: list[np.ndarray] = []

    def _cb(indata, frame_count, time_info, status):
        if status:
            print(f"  [audio warning] {status}", file=sys.stderr)
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=_cb,
    )

    with stream:
        input("  Press ENTER to STOP recording... ")

    if not frames:
        raise RuntimeError("No audio was captured")

    audio = np.concatenate(frames).squeeze()
    audio = trim_silence(audio)
    duration = len(audio) / sample_rate
    print(f"  Recorded {duration:.2f}s of audio")
    return audio.astype(np.float32)


# =============================================================================
# 2. Chinese ASR
# =============================================================================

# Cache for the WhisperModel (loading 3GB takes ~30s, so reuse across calls)
_whisper_cache: dict = {}


def _get_whisper():
    """Return a cached WhisperModel. Loads on first call."""
    if "whisper" not in _whisper_cache:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper not installed.\n"
                "  Install: pip install faster-whisper"
            ) from e
        device = "cuda" if has_cuda() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"  Loading Whisper 'large-v3' on {device}...")
        _whisper_cache["whisper"] = WhisperModel(
            "large-v3",
            device=device,
            compute_type=compute_type,
        )
        print(f"  Whisper loaded.")
    return _whisper_cache["whisper"]


def transcribe_chinese(audio: np.ndarray, sample_rate: int = 24000, ref_duration: float = 0.0):
    """
    Transcribe Chinese audio with faster-whisper.

    Parameters
    ----------
    audio : np.ndarray
    sample_rate : int
    ref_duration : float
        If > 0, also extract the words that fall within the first `ref_duration`
        seconds of audio and return them as the 4th element (ref_text). This
        avoids a second ASR pass when we just need the first ~6s for F5-TTS.

    Returns
    -------
    (text: str, word_timings: list[dict], duration_sec: float)
    (text: str, word_timings: list[dict], duration_sec: float, ref_text: str)
        when ref_duration > 0
    """
    print("\n[1/4] Transcribing Chinese audio...")

    model = _get_whisper()

    duration = len(audio) / sample_rate
    print(f"  Transcribing {duration:.2f}s of audio...")

    segments, info = model.transcribe(
        audio,
        language="zh",
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )

    text_parts: list[str] = []
    word_timings: list[dict] = []
    for seg in segments:
        text_parts.append(seg.text)
        if getattr(seg, "words", None):
            for w in seg.words:
                word_timings.append({"word": w.word, "start": w.start, "end": w.end})

    full_text = "".join(text_parts).strip()
    print(f"  Recognized ({info.language_probability:.0%} confidence):")
    print(f"  > {full_text}")
    print(f"  Duration: {duration:.2f}s · {len(word_timings)} timed words")

    if ref_duration > 0:
        # Extract text of words whose end-time is within ref_duration.
        ref_words = [w["word"] for w in word_timings if w["end"] <= ref_duration]
        ref_text = "".join(ref_words).strip()
        if not ref_text and word_timings:
            # Fallback: take first 7 words
            ref_text = "".join(w["word"] for w in word_timings[:7]).strip()
        print(f"  ref_text (first {ref_duration}s): {ref_text}")
        return full_text, word_timings, duration, (ref_text or full_text)

    return full_text, word_timings, duration


# =============================================================================
# 3. Translation  (Chinese → English)
# =============================================================================

def translate_to_english(chinese_text: str) -> str:
    """
    Translate Chinese to English. Tries ollama (best quality) first,
    then falls back to argostranslate (offline, lighter).
    """
    print("\n[2/4] Translating Chinese → English...")

    # Preferred: local LLM via ollama
    try:
        result = _translate_ollama(chinese_text)
        if result:
            print(f"  > {result}")
            return result
    except Exception as e:
        print(f"  (ollama unavailable: {e})")

    # Fallback: offline argostranslate
    try:
        result = _translate_argos(chinese_text)
        print(f"  > {result}")
        return result
    except Exception as e:
        raise RuntimeError(
            f"All translation backends failed. Last error: {e}"
        ) from e


def _has_non_english(text: str) -> bool:
    """Detect whether the text contains any non-ASCII characters (Chinese, Hebrew, Arabic, etc.)."""
    return any(ord(ch) > 127 for ch in text)


def _translate_ollama(text: str) -> str:
    """Translate via local ollama daemon. Requires `ollama serve` running."""
    import requests

    base_prompt = (
        "You are a translator. Translate the following Chinese text into "
        "natural, fluent English. Output ONLY the English translation, "
        "no quotes, no commentary, no Chinese characters.\n\n"
        f"Chinese: {text}\n\nEnglish:"
    )
    strict_prompt = (
        "Translate this Chinese sentence into English. "
        "Reply with ONLY the English translation, using ASCII letters only. "
        "Do not include any Chinese characters or other non-English text.\n\n"
        f"Chinese: {text}\n\nEnglish:"
    )

    # Try several models in order of preference
    for model in ["qwen2.5:7b", "qwen2:7b", "llama3.1:8b", "mistral:7b", "gemma2:9b"]:
        # Two attempts per model: base prompt, then strict prompt if non-English leaks
        for attempt, prompt in enumerate([base_prompt, strict_prompt], start=1):
            try:
                r = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=120,
                )
                r.raise_for_status()
                result = r.json().get("response", "").strip()
                # Strip any chain-of-thought blocks some models emit
                for tag in ("<think>", "</think>"):
                    result = result.replace(tag, "")
                result = result.strip().strip(chr(34)).strip(chr(39))
                if not result:
                    continue
                if _has_non_english(result):
                    print(f"  (model {model} leaked non-English on attempt {attempt}, retrying)")
                    continue
                return result
            except Exception:
                continue

    raise RuntimeError("No ollama model responded (is `ollama serve` running?)")





def _translate_argos(text: str) -> str:
    """Offline translation via argostranslate. Downloads zh->en pack on first run."""
    import os
    # Disable stanza's runtime download (it tries to fetch from raw.githubusercontent.com)
    os.environ.setdefault("ARGOS_STANZA_AVAILABLE", "false")
    import argostranslate.package as package
    import argostranslate.translate as translate

    installed = translate.get_installed_languages()
    zh = next((l for l in installed if l.code.startswith("zh")), None)
    en = next((l for l in installed if l.code.startswith("en")), None)

    if not (zh and en):
        print("  Downloading zh→en language pack (one-time)...")
        package.update_package_index()
        available = package.get_available_packages()
        pack = next(
            (p for p in available if p.from_code.startswith("zh") and p.to_code.startswith("en")),
            None,
        )
        if not pack:
            raise RuntimeError("Could not find a zh→en argostranslate pack")
        pack.install()
        installed = translate.get_installed_languages()
        zh = next((l for l in installed if l.code.startswith("zh")), None)
        en = next((l for l in installed if l.code.startswith("en")), None)

    if not (zh and en):
        raise RuntimeError("zh or en language pack not installed")

    return zh.get_translation(en).translate(text)


# =============================================================================
# 4. English TTS with voice cloning  (F5-TTS, zero-shot cross-lingual)
# =============================================================================

def _trim_f5_echo(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Trim F5-TTS's tendency to prefix output with a clone of the reference audio.
    Uses RMS energy in short windows to find the start of the actual generated speech.
    Caps trim at 70% of audio to avoid over-trimming.
    """
    if len(wav) < sr:  # < 1s, nothing to trim
        return wav

    # Compute RMS in 50ms windows
    win = int(sr * 0.05)
    hop = win
    n_frames = len(wav) // hop
    rms = np.array([
        np.sqrt(np.mean(wav[i*hop:i*hop+win].astype(np.float32) ** 2))
        for i in range(n_frames)
    ])
    if len(rms) == 0:
        return wav

    # Find the noise floor as the 10th percentile of all frames
    noise_floor = np.percentile(rms, 10)
    # Speech threshold: noise floor * 3, but at least 0.005
    threshold = max(noise_floor * 3.0, 0.005)

    # Find the first frame where RMS crosses threshold AND stays above for
    # at least 5 frames (~250ms). This avoids transient noise.
    min_speech_frames = 5
    speech_threshold_count = 0
    trim_at = 0
    for i, r in enumerate(rms):
        if r > threshold:
            speech_threshold_count += 1
            if speech_threshold_count >= min_speech_frames:
                trim_at = max(0, (i - min_speech_frames + 1) * hop)
                break
        else:
            speech_threshold_count = 0

    # Cap trim to at most 70% of audio (keep most of the audio)
    max_trim = int(len(wav) * 0.7)
    if trim_at > max_trim:
        trim_at = 0

    trimmed = wav[trim_at:]
    removed = len(wav) - len(trimmed)
    print(f"  _trim_f5_echo: trimmed {removed/sr:.2f}s from front "
          f"(threshold={threshold:.4f}, noise_floor={noise_floor:.4f})")
    return trimmed


# Cache the loaded model across calls in the same process.
_tts_cache: dict = {}


def _english_ref_text(ref_text: str) -> str:
    """
    Detect Chinese characters in ref_text and translate to English.
    Most-direct fix for F5-TTS cross-lingual artifacts (Chinese-sounding
    phonemes leaking into English output). Falls back to argostranslate
    if ollama is unavailable.
    """
    if not ref_text:
        return ref_text
    if not _has_non_english(ref_text):
        return ref_text  # already English
    print(f"  [ref_text] translating Chinese ref_text -> English: {ref_text!r}")
    try:
        return translate_to_english(ref_text)
    except Exception as e:
        print(f"  [ref_text] translation failed ({e}), keeping original")
        return ref_text


def _clarify_audio(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Light post-processing to clean up F5-TTS output:
      - high-pass filter at 80 Hz to remove low-frequency rumble ("大舌头")
      - peak-normalize to 0.95 to avoid clipping
    """
    if len(wav) < sr // 10:
        return wav
    import librosa
    # Pre-emphasis (high-pass approximation): y[n] = x[n] - 0.95 * x[n-1]
    emphasized = librosa.effects.preemphasis(wav, coef=0.95)
    # Peak normalize
    peak = float(np.max(np.abs(emphasized)))
    if peak > 1e-6:
        emphasized = emphasized * (0.95 / peak)
    return emphasized.astype(np.float32)


def synthesize_english(
    english_text: str,
    reference_audio: np.ndarray,
    reference_sr: int = 24000,
    ref_text: str = "",
    fix_duration: float | None = None,
    quality_mode: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Synthesize English speech in the speaker's voice using F5-TTS.

    F5-TTS supports cross-lingual zero-shot voice cloning: pass a reference
    audio in any language together with its transcript, and it will produce
    new speech in the target language (here, English) that sounds like the
    reference speaker.

    Parameters
    ----------
    english_text : text to speak (English)
    reference_audio : a clean sample of the speaker's voice (Chinese here)
    reference_sr : sample rate of reference_audio
    ref_text : transcript of reference_audio (Chinese) — helps prosody matching
    fix_duration : float, optional
        Total target duration (ref + generated) in seconds. When provided,
        forces single-pass generation (no chunking) which prevents F5-TTS
        from leaking ref_text content mid-output in cross-lingual runs.
        Total must be <= 30s (F5-TTS single-pass limit).
    quality_mode : bool, default True
        When True, applies two quality fixes:
        1. Auto-translates Chinese ref_text to English (removes Chinese
           phoneme artifacts / "唐音杂音" in English output).
        2. Uses nfe_step=128 (vs 64) for better mel generation.
        3. Runs a high-pass filter on output (removes low-freq rumble).

    Returns
    -------
    (audio_array, sample_rate)
    """
    print("\n[3/4] Synthesizing English speech with your voice...")

    try:
        from f5_tts.api import F5TTS
    except ImportError as e:
        raise RuntimeError(
            "F5-TTS not installed.\n"
            "  Install: pip install f5-tts"
        ) from e

    import soundfile as sf

    # F5-TTS wants a file path for the reference audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ref_path = f.name
    sf.write(ref_path, reference_audio, reference_sr)

    try:
        if "f5" not in _tts_cache:
            print("  Loading F5-TTS base model (one-time, ~1.2GB download)...")
            device = "cuda" if has_cuda() else "cpu"
            _tts_cache["f5"] = F5TTS(model="F5TTS_v1_Base", device=device)
            print(f"  Model loaded on {device}.")
        tts = _tts_cache["f5"]

        # Quality mode: translate ref_text to English to remove Chinese
        # phoneme artifacts ("唐音杂音") in the English output.
        if quality_mode:
            ref_text = _english_ref_text(ref_text)

        # F5-TTS 1.1.22's chunking causes ref_text Chinese to leak into
        # English output when ref_text is short (cross-lingual). The chunking
        # is forced by:
        #   max_chars = len(ref_text.encode("utf-8")) / audio_dur * (22 - audio_dur) * speed
        # For a 6s ref with 11 Chinese chars (33 bytes utf-8), max_chars ≈ 88,
        # so any English text > 88 chars splits into 3-4 chunks. Each chunk
        # independently generates and can leak ref_text.
        #
        # To get clean English output, we bypass chunking entirely by
        # monkey-patching chunk_text to always return [text]. This forces
        # a single inference call, which keeps the model's language
        # conditioning consistent. F5-TTS supports up to 30s total for
        # single-pass generation.
        if fix_duration is not None:
            if fix_duration > 30.0:
                print(f"  [warn] fix_duration={fix_duration:.1f}s > 30s, "
                      f"clamping to 30s (F5-TTS single-pass limit)")
                fix_duration = 30.0
            from f5_tts.infer import utils_infer as _f5_utils
            if not getattr(_f5_utils.chunk_text, "_patched_no_chunk", False):
                _orig_chunk_text = _f5_utils.chunk_text
                def _no_chunk(text, max_chars=135):
                    return [text]
                _no_chunk._patched_no_chunk = True  # type: ignore[attr-defined]
                _f5_utils.chunk_text = _no_chunk
                # Also patch in the api module's namespace (it may have
                # imported the symbol directly).
                try:
                    from f5_tts import api as _f5_api
                    _f5_api.chunk_text = _no_chunk  # type: ignore[attr-defined]
                except Exception:
                    pass
                print(f"  [patch] chunk_text -> no-op (single-pass generation)")
            use_fix_duration = fix_duration
        else:
            use_fix_duration = None

        # F5-TTS requires a non-empty ref_text (used for prosody / duration hints).
        # If we don't have it, F5-TTS has a built-in ASR step we can rely on.
        # nfe_step default is 32. 64 is a good sweet spot (better quality).
        # 128+ can cause torchdiffeq to fail with "t must be strictly
        # increasing or decreasing" — numerical instability.
        nfe_steps = 64 if quality_mode else 32
        print(f"  Generating {len(english_text)} chars of English speech "
              f"(nfe_step={nfe_steps}, quality_mode={quality_mode})...")
        wav, sr, _spec = tts.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=english_text,
            speed=1.0,                # natural speed; we stretch to match later
            nfe_step=nfe_steps,       # higher = better quality, slower
            remove_silence=False,
            fix_duration=use_fix_duration,
            show_info=lambda *a, **k: None,  # quiet F5-TTS' own logger
        )

        # F5-TTS 1.1.22 sometimes prefixes the output with a clone of the
        # reference audio (it leaks ref_text contents before gen_text).
        # Trim it by finding the first real "speech" segment using VAD-like
        # energy detection on a short window.
        wav = _trim_f5_echo(wav, sr)

        if quality_mode:
            wav = _clarify_audio(wav, sr)

        return wav.astype(np.float32), int(sr)

    finally:
        if os.path.exists(ref_path):
            os.unlink(ref_path)


# =============================================================================
# 5. Speed matching  (preserve original speaking rate)
# =============================================================================

def time_stretch_to_duration(
    audio: np.ndarray,
    original_duration: float,
    current_duration: float,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Stretch `audio` so its duration matches `original_duration`.
    Uses pyrubberband (best quality, pitch-preserving) or librosa fallback.
    """
    if abs(original_duration - current_duration) < 0.1:
        print(f"\n[4/4] Speed already matches ({current_duration:.2f}s).")
        return audio

    rate = current_duration / original_duration  # >1 ⇒ speed up
    print(
        f"\n[4/4] Adjusting speed: {current_duration:.2f}s → "
        f"{original_duration:.2f}s (×{rate:.2f})"
    )

    # Best: rubberband (preserves pitch perfectly)
    try:
        import pyrubberband
        stretched = pyrubberband.time_stretch(audio, sample_rate, rate)
        print("  Done (via rubberband).")
        return stretched.astype(np.float32)
    except Exception as e:
        print(f"  (rubberband unavailable: {e}; falling back to librosa)")

    # Fallback: librosa (slight phase artifacts possible at extreme rates)
    import librosa
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    print("  Done (via librosa).")
    return stretched.astype(np.float32)
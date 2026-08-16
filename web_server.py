"""
CopyMyVoice web server — minimal HTML UI + Python HTTP backend.

No Gradio. Uses only Python stdlib + the project's pipeline.

Run:
    python web_server.py     # starts on http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Force UTF-8 stdout on Windows so print() doesn't fail on Chinese chars
# when the terminal codepage is GBK (e.g. Windows 10/11 Chinese locale).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent.resolve()
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
TMP_DIR = ROOT / "tmp_uploads"
TMP_DIR.mkdir(exist_ok=True)

# Lazy import: pipeline modules are heavy (torch, f5-tts, etc.)
_pipeline = None
# Serialize requests so F5-TTS on GPU doesn't get concurrent access
_pipeline_lock = None  # created lazily in main()


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        print("[server] Loading pipeline modules (torch + whisper + f5-tts)...")
        t0 = time.time()
        from steps import (
            load_audio,
            save_audio,
            synthesize_english,
            time_stretch_to_duration,
            transcribe_chinese,
            translate_to_english,
        )
        _pipeline = {
            "load_audio": load_audio,
            "save_audio": save_audio,
            "synthesize_english": synthesize_english,
            "time_stretch_to_duration": time_stretch_to_duration,
            "transcribe_chinese": transcribe_chinese,
            "translate_to_english": translate_to_english,
        }
        print(f"[server] Pipeline loaded in {time.time()-t0:.1f}s")
    return _pipeline


def _do_process(audio_path: str, quality_mode: bool = True) -> dict:
    """Run the full pipeline on an audio file. Returns a dict suitable for JSON."""
    t0 = time.time()
    p = _load_pipeline()

    # 1. Load (handles m4a/mp3 via ffmpeg)
    audio, sr = p["load_audio"](audio_path, target_sr=24000)
    print(f"[server] loaded {len(audio)/sr:.2f}s @ {sr}Hz")

    # 2. ASR full audio + extract ref_text (first 6s) in one pass
    zh, ref_zh, orig_dur = _run_asr(p, audio, sr)

    # 3. Translate + 4. TTS + 5. Stretch + 6. Save
    return _transcribe_synthesize_save(p, audio, sr, zh, ref_zh, orig_dur, t0,
                                       quality_mode=quality_mode)


def _do_regenerate(audio_path: str, zh_text: str, quality_mode: bool = True) -> dict:
    """Skip ASR — use the user-edited Chinese text, re-translate and re-synthesize."""
    t0 = time.time()
    p = _load_pipeline()

    # 1. Load
    audio, sr = p["load_audio"](audio_path, target_sr=24000)
    print(f"[server] loaded {len(audio)/sr:.2f}s @ {sr}Hz (for regenerate)")

    # 2. Get ref_text from the first 6s of audio (skip full ASR)
    REF = 6.0
    ref_samples = min(len(audio), int(REF * sr))
    ref_audio_only = audio[:ref_samples]
    asr_result = p["transcribe_chinese"](ref_audio_only, sr, ref_duration=0.0)
    ref_zh = asr_result[0] if asr_result and asr_result[0].strip() else zh_text
    orig_dur = len(audio) / sr

    if not zh_text.strip():
        raise RuntimeError("中文文本为空")

    print(f"[server] regenerate with edited Chinese: {zh_text[:60]}...")

    # 3-6. Translate + TTS + Stretch + Save
    return _transcribe_synthesize_save(p, audio, sr, zh_text, ref_zh, orig_dur, t0,
                                       quality_mode=quality_mode)


def _do_translate_only(audio_path: str, zh_text: str) -> dict:
    """Just translate Chinese → English. No TTS. Returns (zh, en)."""
    t0 = time.time()
    p = _load_pipeline()
    if not zh_text.strip():
        raise RuntimeError("中文文本为空")
    en = p["translate_to_english"](zh_text)
    if not en.strip():
        raise RuntimeError("Translation failed")
    print(f"[server] translate_only: {zh_text[:40]}... -> {en[:40]}...")
    return {
        "zh": zh_text,
        "en": en,
        "duration": f"{time.time()-t0:.1f}",
    }


def _do_synthesize_only(audio_path: str, en_text: str, quality_mode: bool = True) -> dict:
    """Just run TTS on the provided English text. No translation. Returns (en, wav)."""
    t0 = time.time()
    p = _load_pipeline()

    # Load audio + get ref_text from first 6s
    audio, sr = p["load_audio"](audio_path, target_sr=24000)
    REF = 6.0
    ref_samples = min(len(audio), int(REF * sr))
    ref_audio_only = audio[:ref_samples]
    asr_result = p["transcribe_chinese"](ref_audio_only, sr, ref_duration=0.0)
    ref_zh = asr_result[0] if asr_result and asr_result[0].strip() else en_text
    orig_dur = len(audio) / sr

    if not en_text.strip():
        raise RuntimeError("英文文本为空")

    print(f"[server] synthesize_only: {en_text[:40]}...")

    # Run TTS + stretch + save
    REF = 6.0
    ref_samples = min(len(audio), int(REF * sr))
    ref_audio = audio[:ref_samples]
    target_total = orig_dur

    eng, tts_sr = p["synthesize_english"](
        english_text=en_text,
        reference_audio=ref_audio,
        reference_sr=sr,
        ref_text=ref_zh,
        fix_duration=target_total,
        quality_mode=quality_mode,
    )
    cur = len(eng) / tts_sr
    if tts_sr != sr:
        import librosa
        eng = librosa.resample(eng, orig_sr=tts_sr, target_sr=sr)
        tts_sr = sr
        cur = len(eng) / tts_sr

    final = p["time_stretch_to_duration"](eng, orig_dur, cur, sample_rate=tts_sr)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"cloned_{ts}.wav"
    out_path = OUTPUT_DIR / out_name
    p["save_audio"](final, str(out_path), sample_rate=tts_sr)

    return {
        "en": en_text,
        "out_url": f"/output/{out_name}",
        "out_info": f"{len(final)/tts_sr:.2f} 秒 · {tts_sr} Hz · {out_path.name}",
        "duration": f"{time.time()-t0:.1f}",
    }


def _run_asr(p, audio, sr):
    """Run ASR on the full audio. Returns (zh, ref_zh, orig_dur)."""
    REF = 6.0
    asr_result = p["transcribe_chinese"](audio, sr, ref_duration=REF)
    if len(asr_result) == 4:
        zh, _, orig_dur, ref_zh = asr_result
    else:
        zh, _, orig_dur = asr_result
        ref_zh = zh
    if not zh.strip():
        raise RuntimeError("No Chinese detected in audio")
    return zh, ref_zh, orig_dur


def _transcribe_synthesize_save(p, audio, sr, zh, ref_zh, orig_dur, t0,
                                quality_mode: bool = True):
    """Translate + TTS + time-stretch + save. Returns JSON-ready dict."""
    # 3. Translate
    en = p["translate_to_english"](zh)
    if not en.strip():
        raise RuntimeError("Translation failed")

    # 4. TTS — use ONLY first 6s as voice prompt
    REF = 6.0
    ref_samples = min(len(audio), int(REF * sr))
    ref_audio = audio[:ref_samples]

    # fix_duration: avoid F5-TTS chunking by passing target total duration
    # (ref + generated). This forces single-pass generation, which doesn't
    # leak ref_text mid-output.
    target_total = orig_dur

    eng, tts_sr = p["synthesize_english"](
        english_text=en,
        reference_audio=ref_audio,
        reference_sr=sr,
        ref_text=ref_zh,
        fix_duration=target_total,
        quality_mode=quality_mode,
    )
    cur = len(eng) / tts_sr
    if tts_sr != sr:
        import librosa
        eng = librosa.resample(eng, orig_sr=tts_sr, target_sr=sr)
        tts_sr = sr
        cur = len(eng) / tts_sr

    # 5. Time-stretch to original duration
    final = p["time_stretch_to_duration"](eng, orig_dur, cur, sample_rate=tts_sr)

    # 6. Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"cloned_{ts}.wav"
    out_path = OUTPUT_DIR / out_name
    p["save_audio"](final, str(out_path), sample_rate=tts_sr)

    elapsed = time.time() - t0
    return {
        "zh": zh,
        "en": en,
        "out_url": f"/output/{out_name}",
        "out_info": f"{len(final)/tts_sr:.2f} 秒 · {tts_sr} Hz · {out_path.name}",
        "duration": f"{elapsed:.1f}",
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Less noisy than default
        sys.stderr.write(f"[http] {self.address_string()} {fmt % args}\n")

    def _send_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path

        if path == "/" or path == "/index.html":
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/output/"):
            fname = path[len("/output/"):]
            # Prevent traversal
            if "/" in fname or "\\" in fname or ".." in fname:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            self._send_file(OUTPUT_DIR / fname, "audio/wav")
            return
        if path == "/health":
            self._send_json({"ok": True})
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def _parse_multipart(self):
        """Parse a multipart/form-data body. Returns (audio_bytes, filename, form_fields)."""
        ctype = self.headers.get("Content-Type", "")
        m = re.match(r"multipart/form-data;\s*boundary=(.+)", ctype)
        if not m:
            return None, None, {}
        boundary = ("--" + m.group(1)).encode("utf-8")

        clen = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(clen)

        audio_bytes = None
        filename = "upload.bin"
        form_fields: dict[str, str] = {}
        for part in body.split(boundary):
            if b'name="audio"' in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end < 0:
                    continue
                headers = part[:header_end].decode("utf-8", errors="ignore")
                content = part[header_end + 4:]
                if content.endswith(b"\r\n"):
                    content = content[:-2]
                fnmatch = re.search(r'filename="([^"]+)"', headers)
                if fnmatch:
                    filename = fnmatch.group(1)
                audio_bytes = content
            else:
                # Generic form field (no filename)
                header_end = part.find(b"\r\n\r\n")
                if header_end < 0:
                    continue
                headers = part[:header_end].decode("utf-8", errors="ignore")
                content = part[header_end + 4:]
                if content.endswith(b"\r\n"):
                    content = content[:-2]
                nm = re.search(r'name="([^"]+)"', headers)
                if nm:
                    form_fields[nm.group(1)] = content.decode("utf-8", errors="ignore")
        return audio_bytes, filename, form_fields

    def do_POST(self):
        url = urlparse(self.path)
        if url.path not in ("/api/process", "/api/regenerate",
                            "/api/translate", "/api/synthesize"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        audio_bytes, filename, form = self._parse_multipart()
        if not audio_bytes:
            self._send_json({"error": "No audio file in request"}, status=400)
            return

        quality_mode = form.get("quality_mode", "true").lower() != "false"

        # Save to temp
        tmp_ext = Path(filename).suffix or ".bin"
        tmp_path = TMP_DIR / f"upload_{int(time.time()*1000)}{tmp_ext}"
        tmp_path.write_bytes(audio_bytes)
        print(f"[server] received {filename} ({len(audio_bytes)} bytes) -> {tmp_path} "
              f"({url.path}, quality_mode={quality_mode})")

        try:
            with _pipeline_lock:
                print(f"[server] {threading.current_thread().name} acquired lock, "
                      f"processing ({url.path})...")
                if url.path == "/api/process":
                    result = _do_process(str(tmp_path), quality_mode=quality_mode)
                elif url.path == "/api/regenerate":
                    edited_zh = form.get("zh_text", "").strip()
                    if not edited_zh:
                        raise RuntimeError("缺少 zh_text 字段")
                    result = _do_regenerate(str(tmp_path), edited_zh,
                                            quality_mode=quality_mode)
                elif url.path == "/api/translate":
                    edited_zh = form.get("zh_text", "").strip()
                    if not edited_zh:
                        raise RuntimeError("缺少 zh_text 字段")
                    result = _do_translate_only(str(tmp_path), edited_zh)
                else:  # /api/synthesize
                    en_text = form.get("en_text", "").strip()
                    if not en_text:
                        raise RuntimeError("缺少 en_text 字段")
                    result = _do_synthesize_only(str(tmp_path), en_text,
                                                  quality_mode=quality_mode)
                print(f"[server] {threading.current_thread().name} done in {result['duration']}s")
            self._send_json(result)
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=500)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def main():
    host = "127.0.0.1"
    port = 7860
    print("=" * 60)
    print(f"CopyMyVoice Web UI")
    print(f"  Open http://{host}:{port} in your browser")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Eagerly load pipeline so the first request is fast
    try:
        _load_pipeline()
    except Exception as e:
        print(f"[server] WARN: pipeline preload failed: {e}")

    # Build the lock now (needs threading imported, but we deferred to avoid
    # side effects at module import time)
    global _pipeline_lock
    import threading
    _pipeline_lock = threading.Lock()

    # Auto-open browser
    import threading
    import webbrowser
    def _open():
        time.sleep(1.0)
        webbrowser.open(f"http://{host}:{port}")
    threading.Thread(target=_open, daemon=True).start()

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"[server] serving on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] shutting down")
        srv.shutdown()


if __name__ == "__main__":
    main()

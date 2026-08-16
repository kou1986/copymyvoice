"""
Gradio web UI for copymyvoice.

Run with:
    python app.py

Or double-click `copymyvoice.bat` on the desktop.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

import gradio as gr
import numpy as np


HERE = Path(__file__).parent.resolve()
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _import_pipeline():
    """Import the heavy ML modules lazily so the UI starts fast."""
    from steps import (
        load_audio,
        record_audio,
        save_audio,
        synthesize_english,
        time_stretch_to_duration,
        translate_to_english,
        transcribe_chinese,
    )
    from pipeline import run as pipeline_run
    return {
        "load_audio": load_audio,
        "record_audio": record_audio,
        "save_audio": save_audio,
        "synthesize_english": synthesize_english,
        "time_stretch_to_duration": time_stretch_to_duration,
        "translate_to_english": translate_to_english,
        "transcribe_chinese": transcribe_chinese,
        "pipeline_run": pipeline_run,
    }


def run_pipeline(
    audio_path: str | None,
    record_path: str | None,
    progress: gr.Progress = None,
) -> tuple[str, str, str, str]:
    """The main worker. Called by Gradio with the uploaded/recorded file path."""
    if progress is None:
        progress = gr.Progress()

    if not audio_path and not record_path:
        raise gr.Error("请先录音，或上传一个音频文件。")
    src = record_path or audio_path
    if not src or not os.path.exists(src):
        raise gr.Error("音频文件不存在。")

    progress(0.0, desc="加载模型中...")
    funcs = _import_pipeline()
    transcribe_chinese = funcs["transcribe_chinese"]
    translate_to_english = funcs["translate_to_english"]
    synthesize_english = funcs["synthesize_english"]
    time_stretch_to_duration = funcs["time_stretch_to_duration"]
    save_audio = funcs["save_audio"]
    load_audio = funcs["load_audio"]

    progress(0.1, desc="读取音频...")
    audio, sr = load_audio(src, target_sr=24000)
    duration = len(audio) / sr
    print(f"  Loaded {duration:.2f}s @ {sr}Hz")

    progress(0.2, desc="[1/4] 中文识别 (Whisper large-v3)...")
    REF_DURATION = 6.0
    asr_result = transcribe_chinese(audio, sr, ref_duration=REF_DURATION)
    if len(asr_result) == 4:
        chinese_text, _words, orig_duration, ref_chinese_text = asr_result
    else:
        chinese_text, _words, orig_duration = asr_result
        ref_chinese_text = chinese_text
    if not chinese_text.strip():
        raise gr.Error("没识别到中文，请检查音频。")

    progress(0.45, desc="[2/4] 翻译 (ollama qwen2.5:7b)...")
    english_text = translate_to_english(chinese_text)
    if not english_text.strip():
        raise gr.Error("翻译失败。")

    progress(0.65, desc="[3/4] 合成英文 (F5-TTS 音色克隆)...")
    # Use ONLY the first ~6s as the voice prompt (F5-TTS needs short ref).
    ref_samples = min(len(audio), int(REF_DURATION * sr))
    ref_audio = audio[:ref_samples]

    eng_audio, tts_sr = synthesize_english(
        english_text=english_text,
        reference_audio=ref_audio,
        reference_sr=sr,
        ref_text=ref_chinese_text,
        fix_duration=duration,
    )
    current_duration = len(eng_audio) / tts_sr

    if tts_sr != sr:
        import librosa
        eng_audio = librosa.resample(eng_audio, orig_sr=tts_sr, target_sr=sr)
        tts_sr = sr
        current_duration = len(eng_audio) / tts_sr

    progress(0.85, desc="[4/4] 语速对齐...")
    final_audio = time_stretch_to_duration(
        eng_audio, orig_duration, current_duration, sample_rate=tts_sr
    )

    progress(0.95, desc="保存...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"cloned_{timestamp}.wav"
    save_audio(final_audio, str(out_path), sample_rate=tts_sr)

    progress(1.0, desc="完成 ✓")
    return (
        chinese_text,
        english_text,
        f"{len(final_audio) / tts_sr:.2f} 秒  ·  {tts_sr} Hz  ·  {out_path.name}",
        str(out_path),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="CopyMyVoice — 中文 → 你的音色说英语",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # �️ CopyMyVoice
        **把你说的中文变成「你自己」说的英语，并保留语速。**

        | 步骤 | 模型 |
        |---|---|
        | 中文识别 | faster-whisper `large-v3` |
        | 翻译 | ollama `qwen2.5:7b` (本地) |
        | 音色克隆 | F5-TTS `F5TTS_v1_Base` |
        | 语速对齐 | librosa / rubberband |

        全程本地运行，免费，隐私好。
        """
    )

    with gr.Tabs():
        with gr.Tab("🎤 麦克风录音"):
            mic_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="点下面圆点录音",
                show_download_button=False,
            )
            mic_run = gr.Button("开始转换", variant="primary", size="lg")
            mic_status = gr.Textbox(label="状态", interactive=False)
            mic_out_audio = gr.Audio(label="输出 (你的音色)", type="filepath")

        with gr.Tab("📁 上传音频"):
            file_input = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="选一个 wav / mp3 / m4a 文件",
                show_download_button=False,
            )
            file_run = gr.Button("开始转换", variant="primary", size="lg")
            file_status = gr.Textbox(label="状态", interactive=False)
            file_out_audio = gr.Audio(label="输出 (你的音色)", type="filepath")

    with gr.Accordion("📝 识别 / 翻译文本 (调试用)", open=False):
        out_zh = gr.Textbox(label="识别到的中文", interactive=False, lines=2)
        out_en = gr.Textbox(label="翻译出的英文", interactive=False, lines=2)

    with gr.Accordion("💡 录音技巧", open=False):
        gr.Markdown(
            """
            - **安静环境** + **10 秒以上** → 克隆效果明显更好
            - 普通话自然说，不要刻意放慢或夸张语气
            - 太短 (< 5 秒) 音色克隆会失真
            - 第一次跑会下载模型 (~3 GB)，要等几分钟；之后秒开
            - 输出默认在 `D:\\copymyvoice\\output\\`
            """
        )

    # Wiring
    mic_run.click(
        run_pipeline,
        inputs=[gr.State(None), mic_input],
        outputs=[out_zh, out_en, mic_status, mic_out_audio],
        show_progress="full",
    )
    file_run.click(
        run_pipeline,
        inputs=[file_input, gr.State(None)],
        outputs=[out_zh, out_en, file_status, file_out_audio],
        show_progress="full",
    )


def main():
    print("=" * 60)
    print("CopyMyVoice UI")
    print(f"  Project: {HERE}")
    print(f"  Output:  {OUTPUT_DIR}")
    print("=" * 60)
    # Auto-open browser shortly after the server starts
    import threading
    def _open():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:7860")
    threading.Thread(target=_open, daemon=True).start()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        show_error=True,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()

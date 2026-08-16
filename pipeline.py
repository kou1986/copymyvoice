"""
Pipeline orchestrator — wires the individual steps together.
"""
from __future__ import annotations

import numpy as np

from steps import (
    save_audio,
    synthesize_english,
    time_stretch_to_duration,
    transcribe_chinese,
    translate_to_english,
)


def run(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str,
) -> tuple[str, str]:
    """
    Run the full pipeline:

        Chinese audio
          → Chinese text + duration           (ASR)
          → English text                      (translation)
          → English speech in your voice      (TTS + cloning)
          → Time-stretched to match duration  (speed preservation)

    Returns the English text and the output file path.
    """
    # 1. ASR full audio + extract ref_text (first 6s) in one pass
    REF_DURATION = 6.0
    asr_result = transcribe_chinese(audio, sample_rate, ref_duration=REF_DURATION)
    if len(asr_result) == 4:
        chinese_text, _word_timings, original_duration, ref_chinese_text = asr_result
    else:
        chinese_text, _word_timings, original_duration = asr_result
        ref_chinese_text = chinese_text
    if not chinese_text.strip():
        raise RuntimeError("No Chinese speech detected in audio.")

    # 2. Translation
    english_text = translate_to_english(chinese_text)
    if not english_text.strip():
        raise RuntimeError("Translation produced empty text.")

    # 3. TTS with voice cloning.
    #    F5-TTS works best with a SHORT reference (6-10s of audio + short ref_text).
    #    Use only the first ~6 seconds as the voice prompt.
    ref_samples = min(len(audio), int(REF_DURATION * sample_rate))
    ref_audio = audio[:ref_samples]

    english_audio, tts_sr = synthesize_english(
        english_text=english_text,
        reference_audio=ref_audio,
        reference_sr=sample_rate,
        ref_text=ref_chinese_text,
        fix_duration=original_duration,
    )
    current_duration = len(english_audio) / tts_sr

    # 4. Resample if needed, then time-stretch to match original duration
    if tts_sr != sample_rate:
        import librosa
        english_audio = librosa.resample(
            y=english_audio, orig_sr=tts_sr, target_sr=sample_rate
        )
        tts_sr = sample_rate
        current_duration = len(english_audio) / tts_sr

    final_audio = time_stretch_to_duration(
        english_audio,
        original_duration=original_duration,
        current_duration=current_duration,
        sample_rate=tts_sr,
    )

    # 5. Save
    save_audio(final_audio, output_path, sample_rate=tts_sr)
    return english_text, output_path
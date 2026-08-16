#!/usr/bin/env python3
"""
clone.py — Convert Chinese speech to English speech in YOUR voice.

Usage:
  # Record from microphone (default), output to output/output.wav
  python clone.py

  # Use an existing Chinese audio file
  python clone.py --input recording.wav

  # Specify output
  python clone.py --input recording.wav --out result.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from steps import load_audio, record_audio
from pipeline import run


BANNER = """
============================================================
  Chinese → English  ·  voice-cloned, speed-matched
============================================================
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Chinese speech to English speech in your voice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Path to a Chinese audio file (.wav / .mp3 / .m4a / .flac). "
             "If omitted, records from the microphone.",
    )
    p.add_argument(
        "-o", "--out",
        type=str,
        default="output/output.wav",
        help="Path for the English output wav (default: output/output.wav).",
    )
    p.add_argument(
        "-sr", "--sample-rate",
        type=int,
        default=24000,
        help="Working sample rate in Hz (default: 24000).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(BANNER)

    # 0. Get audio
    if args.input:
        print(f"[0/4] Loading audio: {args.input}")
        try:
            audio, sample_rate = load_audio(args.input, target_sr=args.sample_rate)
        except Exception as e:
            print(f"ERROR loading file: {e}", file=sys.stderr)
            return 2
        print(f"  Loaded {len(audio)/args.sample_rate:.2f}s")
    else:
        try:
            audio = record_audio(sample_rate=args.sample_rate)
        except Exception as e:
            print(f"ERROR recording: {e}", file=sys.stderr)
            return 2

    if len(audio) < args.sample_rate:  # < 1 second
        print("ERROR: Audio is too short (< 1s).", file=sys.stderr)
        return 2

    # Run pipeline
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        english_text, output_path = run(audio, args.sample_rate, args.out)
    except Exception as e:
        print(f"\nPIPELINE ERROR: {e}", file=sys.stderr)
        return 1

    print()
    print("=" * 60)
    print("  ✅  Done.")
    print(f"  English:   {english_text}")
    print(f"  Audio:     {output_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Render four Russian voice-over lines using one of the two test models."""

import argparse
import os
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

KITTEN_DIRECTIONS = ("[joyful]", "[surprised] <gasp>", "[sad] <sigh>", "[joyful]")
QWEN_DIRECTIONS = ("joy", "surprise", "sad", "joy")
QWEN_STYLE = (
    "A cheerful, quick young adult hockey presenter opening her show.",
    "Genuine disbelief at an absurd hockey score. Hit the last two words hard.",
    "Say this quietly and sympathetically. Sound mildly sad.",
    "Begin dry and amused, then give the last sentence a small cheerful lift.",
)


def render_kitten(lines, model_path):
    import torch
    from kittenml import KittenTTS

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = KittenTTS(str(model_path or "KittenML/kitten-tts-2"), device="cpu")
    for index, (line, direction) in enumerate(zip(lines, KITTEN_DIRECTIONS, strict=True)):
        audio = model.generate(f"{direction} {line}", voice="RussianHockeyCheer",
                               preset="expressive", normalize=False,
                               max_new_tokens=350, advanced={"seed": 2026 + index})
        yield audio, model.sample_rate


def render_qwen(lines, model_path, engine, output):
    for index, (line, emotion, style) in enumerate(
        zip(lines, QWEN_DIRECTIONS, QWEN_STYLE, strict=True), start=1
    ):
        path = output / f"qwen_line_{index}.wav"
        # The C runner locates the emotion steering assets from its own directory.
        subprocess.run(
            [str(engine / "qwen_tts"), "-d", str(model_path), "-s", "vivian",
             "-l", "Russian", "--int8", "-j", "4", "--seed", str(2025 + index),
             "--emotion", emotion, "--instruct", style, "--max-duration", "12",
             "--text", line, "-o", str(path)],
            cwd=engine, check=True,
        )
        yield sf.read(path, dtype="float32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("kitten", "qwen"))
    parser.add_argument("script", type=Path)
    parser.add_argument("--engine", type=lambda path: Path(path).resolve())
    parser.add_argument("--model", type=lambda path: Path(path).resolve())
    parser.add_argument("--output", type=lambda path: Path(path).resolve(), required=True)
    args = parser.parse_args()
    if args.backend == "qwen" and (args.engine is None or args.model is None):
        parser.error("Qwen requires --engine and --model")

    lines = args.script.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 4 or any(not line.strip() for line in lines):
        parser.error("Provide four nonempty lines")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.backend == "kitten":
        clips = render_kitten(lines, args.model)
    else:
        clips = render_qwen(lines, args.model, args.engine, args.output)

    parts = []
    for index, (audio, rate) in enumerate(clips, start=1):
        if audio.ndim != 1 or rate != 24000 or not np.isfinite(audio).all() or not audio.any():
            raise RuntimeError(f"Invalid audio for {args.backend} line {index}")
        print(f"{args.backend} line {index}: {len(audio) / rate:.2f}s", flush=True)
        if index > 1:
            parts.append(np.zeros(int(0.35 * rate), dtype=np.float32))
        parts.append(audio)
    sf.write(args.output / f"{args.backend}_polina.wav", np.concatenate(parts), 24000)


if __name__ == "__main__":
    main()

"""Run the same Russian scene twice with Chatterbox Multilingual V3."""

import argparse
import os
from pathlib import Path
from time import monotonic

import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from huggingface_hub import snapshot_download

MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
CHECKPOINT = "t3_mtl23ls_v3.safetensors"
FILES = (
    "ve.pt",
    CHECKPOINT,
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)
EXPRESSIVE = (0.75, 0.9, 0.28, 0.62)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path, help="UTF-8 text, four lines of dialogue")
    parser.add_argument("--reference", type=Path, help="Optional voice reference audio")
    parser.add_argument("--weights", type=Path, help="Use predownloaded model files")
    parser.add_argument("--output", type=Path, default=Path("output-chatterbox"))
    args = parser.parse_args()

    lines = [line.strip() for line in args.script.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(lines) != len(EXPRESSIVE):
        parser.error("The script must contain four nonempty lines")

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    started = monotonic()
    weights = args.weights or snapshot_download(
        "ResembleAI/chatterbox", revision=MODEL_REVISION, allow_patterns=FILES,
    )
    model = ChatterboxMultilingualTTS.from_local(weights, "cpu", t3_model=CHECKPOINT)
    if args.reference:
        model.prepare_conditionals(str(args.reference))
    print(f"Model ready after {monotonic() - started:.1f}s", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    pause = torch.zeros(1, round(0.3 * model.sr))
    for name, values in (("plain", (0.5,) * len(lines)), ("expressive", EXPRESSIVE)):
        parts = []
        for index, (line, exaggeration) in enumerate(zip(lines, values, strict=True), 1):
            started = monotonic()
            torch.manual_seed(2026 + index)
            audio = model.generate(
                line,
                language_id="ru",
                exaggeration=exaggeration,
            )
            if not audio.numel() or not torch.isfinite(audio).all():
                raise RuntimeError(f"Invalid audio for {name}, line {index}")
            print(f"{name} line {index}: {audio.size(-1) / model.sr:.2f}s audio "
                  f"in {monotonic() - started:.1f}s", flush=True)
            parts.extend((audio, pause))
        torchaudio.save(args.output / f"chatterbox_{name}.wav",
                        torch.cat(parts[:-1], dim=-1), model.sr)


if __name__ == "__main__":
    main()

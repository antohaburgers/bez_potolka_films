"""Generate a casting pack using Silero CIS Base (MIT)."""
import os
from pathlib import Path
import subprocess
import wave
import numpy as np
import torch

RATE = 24000
OUT = Path("output")
OUT.mkdir(exist_ok=True)
torch.set_num_threads(min(4, os.cpu_count() or 2))
model_path = Path("v5_cis_base_nostress.pt")
if not model_path.exists():
    torch.hub.download_url_to_file(
        "https://models.silero.ai/models/tts/ru/v5_cis_base_nostress.pt",
        str(model_path))
model = torch.package.PackageImporter(str(model_path)).load_pickle("tts_models", "model")
model.to(torch.device("cpu"))
voices = sorted(v for v in model.speakers if v.startswith("ru_"))
print("Russian voices:", voices, flush=True)
if not voices:
    raise RuntimeError("No Russian voices found")
preferred = ["ru_anatoliy", "ru_dmitriy", "ru_roman", "ru_igor",
             "ru_larisa", "ru_vika", "ru_oksana", "ru_ekaterina"]
selected = [v for v in preferred if v in voices]
selected += [v for v in voices if v not in selected]
selected = selected[:10]
lines = [
    "Здр+авствуйте. Почем+у вы хот+ите раб+отать +именно у нас?",
    "Потом+у что мне нужн+ы д+еньги. Я заеб+ался.",
    "У нас др+ужный коллект+ив. Потолк+а нет.",
    "А пол, бл+ядь, хоть есть?"
]
def speak(text, voice):
    result = model.apply_tts(text=text, speaker=voice, sample_rate=RATE)
    return result.detach().cpu().numpy().reshape(-1)
def silence(seconds):
    return np.zeros(int(RATE*seconds), dtype=np.float32)
def save(name, samples):
    wav = OUT / (name + ".wav")
    mp3 = OUT / (name + ".mp3")
    with wave.open(str(wav), "wb") as f:
        f.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        f.writeframes((np.clip(samples, -1, 1)*32767).astype("<i2").tobytes())
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(wav),
                    "-codec:a","libmp3lame","-b:a","128k",str(mp3)],check=True)
    wav.unlink()

combined = []
manifest = []
for i, voice in enumerate(selected, 1):
    print(f"{i}: {voice}", flush=True)
    audio = np.concatenate([part for line in lines for part in (speak(line, voice), silence(.5))])
    name = f"{i:02d}_{voice}"
    save(name, audio)
    combined.extend([speak("+Голос н+омер " + ["од+ин","два","три","чет+ыре","пять","шесть","семь","в+осемь","д+евять","д+есять"][i-1] + ".", voice), silence(.4), audio, silence(1)])
    manifest.append(name)
save("casting", np.concatenate(combined))
(OUT/"voices.txt").write_text("\n".join(manifest)+"\n", encoding="utf-8")

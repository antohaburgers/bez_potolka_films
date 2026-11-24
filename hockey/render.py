"""Technical-only renderer. No publication or editorial approval bypass."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import textwrap
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RATE, FPS = 24000, 30
W, H = 540, 960
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ROOT = Path(__file__).resolve().parent


def validate(s):
    if set(s) != {"version", "kind", "id", "voice", "location", "lines"}:
        raise ValueError("Unexpected or missing scenario keys")
    if s["version"] != 1 or s["kind"] != "technical_test":
        raise ValueError("Only technical tests are enabled")
    if s["location"] != "test_studio" or s["voice"] not in {"ru_vika", "ru_oksana", "ru_ekaterina"}:
        raise ValueError("Unsupported test location or voice")
    if not isinstance(s["id"], str) or not s["id"].isalnum() and not s["id"].replace("-", "").isalnum():
        raise ValueError("Invalid id")
    if not isinstance(s["lines"], list) or not 1 <= len(s["lines"]) <= 6:
        raise ValueError("Expected 1-6 lines")
    for line in s["lines"]:
        if set(line) != {"text", "emotion"} or line["emotion"] not in {"neutral", "smile"}:
            raise ValueError("Invalid line")
        if not isinstance(line["text"], str) or not 1 <= len(line["text"].strip()) <= 180:
            raise ValueError("Invalid text")
    return s


def write_wav(path, audio):
    with wave.open(str(path), "wb") as f:
        f.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def synthesize(s):
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 2))
    path = ROOT / "v5_cis_base_nostress.pt"
    if not path.exists():
        torch.hub.download_url_to_file("https://models.silero.ai/models/tts/ru/v5_cis_base_nostress.pt", str(path))
    model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    if s["voice"] not in model.speakers:
        raise ValueError("Voice unavailable")
    parts, cues, cursor = [], [], 0
    for line in s["lines"]:
        data = model.apply_tts(text=line["text"], speaker=s["voice"], sample_rate=RATE).detach().cpu().numpy().reshape(-1)
        if not len(data) or not np.isfinite(data).all() or np.max(np.abs(data)) < .001:
            raise ValueError("Invalid speech")
        cues.append(dict(line, start=cursor/RATE, end=(cursor+len(data))/RATE))
        pause = np.zeros(RATE//3, dtype=np.float32)
        parts.extend([data, pause])
        cursor += len(data)+len(pause)
    return np.concatenate(parts), cues


def timestamp(seconds):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def frame(t, mouth, cue):
    im = Image.new("RGB", (W, H), "#111b2b")
    d = ImageDraw.Draw(im)
    d.rectangle((0,0,W,100), fill="#203e57")
    d.text((24,22), "ХОККЕЙ-ТЯН", font=ImageFont.truetype(FONT,32), fill="white")
    d.text((24,65), "ТЕХНИЧЕСКИЙ ТЕСТ / 0.1", font=ImageFont.truetype(FONT,17), fill="#aad9e9")
    dy = round(3*math.sin(t*2))
    d.rounded_rectangle((50,150,490,690), radius=28, fill="#23364a", outline="#426780", width=2)
    d.ellipse((155,435+dy,385,695+dy), fill="#426780")
    d.ellipse((170,240+dy,370,465+dy), fill="#c1d1db")
    blink = t % 3.7 > 3.55
    for x in (225,315):
        d.ellipse((x-9,320+dy,x+9,322+dy if blink else 341+dy), fill="#182536")
    opening = max(2, min(25, round(mouth*190)))
    if cue and cue["emotion"] == "smile":
        d.arc((243,365+dy,297,403+dy), 0, 180, fill="#182536", width=3)
    d.ellipse((254,378+dy-opening//2,286,378+dy+opening//2), fill="#182536")
    d.text((90,610), "ВРЕМЕННАЯ МОДЕЛЬ", font=ImageFont.truetype(FONT,22), fill="white")
    d.rounded_rectangle((28,733,512,891), radius=20, fill="#0a111c")
    if cue:
        font=ImageFont.truetype(FONT,26)
        lines=textwrap.wrap(cue["text"], width=29, break_long_words=True)
        while any(d.textlength(x,font=font)>445 for x in lines) or len(lines)*33>130:
            font=ImageFont.truetype(FONT,max(10,font.size-1))
            if font.size==10: break
        d.multiline_text((48,750), "\n".join(lines), font=font, fill="white", spacing=5)
    d.text((29,920), "НЕ ДЛЯ ПУБЛИКАЦИИ", font=ImageFont.truetype(FONT,17), fill="#9ab3c8")
    return im


def render(s, out):
    out.mkdir(parents=True, exist_ok=True)
    audio, cues = synthesize(s)
    duration = len(audio)/RATE
    if not 1 <= duration <= 60:
        raise ValueError("Test duration must be 1-60 seconds")
    wav=out/"speech.wav"
    write_wav(wav,audio)
    (out/"captions.srt").write_text("\n\n".join(f'{i}\n{timestamp(c["start"])} --> {timestamp(c["end"])}\n{c["text"]}' for i,c in enumerate(cues,1))+"\n",encoding="utf-8")
    (out/"timeline.json").write_text(json.dumps(cues,ensure_ascii=False,indent=2),encoding="utf-8")
    mp4=out/"smoke.mp4"
    cmd=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pixel_format","rgb24","-video_size",f"{W}x{H}","-framerate",str(FPS),"-i","pipe:0","-i",str(wav),"-vf","scale=1080:1920:flags=lanczos","-c:v","libx264","-preset","fast","-crf","22","-pix_fmt","yuv420p","-threads","2","-c:a","aac","-b:a","128k","-movflags","+faststart","-shortest",str(mp4)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as p:
        try:
            for i in range(math.ceil(duration*FPS)):
                t=i/FPS
                chunk=audio[i*RATE//FPS:(i+1)*RATE//FPS]
                rms=float(np.sqrt(np.mean(chunk**2))) if len(chunk) else 0.
                cue=next((c for c in cues if c["start"]<=t<c["end"]),None)
                p.stdin.write(frame(t,rms,cue).tobytes())
        finally:
            p.stdin.close()
        if p.wait()!=0: raise RuntimeError("FFmpeg failed")
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(mp4)]))
    v=next(x for x in probe["streams"] if x["codec_type"]=="video")
    a=next(x for x in probe["streams"] if x["codec_type"]=="audio")
    if (v["codec_name"],v["width"],v["height"],v["r_frame_rate"],a["codec_name"])!=("h264",1080,1920,"30/1","aac"):
        raise RuntimeError("Unexpected output format")
    if abs(float(probe["format"]["duration"])-duration)>.15:
        raise RuntimeError("Duration mismatch")
    subprocess.run(["ffmpeg","-v","error","-i",str(mp4),"-f","null","-"],check=True)
    (out/"ffprobe.json").write_text(json.dumps(probe,indent=2))
    print(f"PASS: H264 1080x1920 30 fps AAC, {duration:.2f}s",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("scenario",type=Path)
    p.add_argument("--out",type=Path,default=Path("hockey-output"))
    args=p.parse_args()
    render(validate(json.loads(args.scenario.read_text(encoding="utf-8"))),args.out)

"""Render a three-scene vertical pilot, entirely on a GitHub Actions runner."""
import json
import math
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reel-output"
WORK = ROOT / "reel-work"
ENGINE = ROOT / "engine"
W, H, FPS = 1080, 1920, 24
PW, PH = 960, 1120
PX, PY = 60, 232
SCENES = [
    ("analyst.png", "АНАЛИТИКА", "В хоккее всё просто. Главное — забить больше, чем пропустить."),
    ("goal.png", "ГОООЛ!", "Но когда твоя команда забивает, аналитика внезапно заканчивается."),
    ("referee_anger.png", "СУДЬЯ?!", "А когда судья отменяет гол, начинается уже совсем другой вид спорта!"),
]

def run(*args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)

def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/" + name, size)

def wrap(draw, text, f, max_width):
    lines, line = [], ""
    for word in text.split():
        candidate = (line + " " + word).strip()
        if line and draw.textlength(candidate, font=f) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines

def compose(index, filename, headline, phrase):
    base = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(base)
    for y in range(H):
        t = y / H
        d.line((0, y, W, y), fill=(int(13 + t*7), int(24 + t*15), int(31 + t*13)))
    d.rounded_rectangle((60, 75, 1020, 180), 25, fill="#203b3c")
    d.text((92, 127), "ХОККЕЙ-ТЯН", font=font(52, True), fill="#f4f8f7", anchor="lm")
    d.text((985, 127), f"0{index}/03", font=font(36, True), fill="#98d7be", anchor="rm")
    d.rounded_rectangle((48, 220, 1032, 1364), 24, fill="#426763")
    d.rounded_rectangle((60, 1390, 1020, 1785), 32, fill="#142528", outline="#36564e", width=3)
    d.text((100, 1430), headline, font=font(66, True), fill="#9ee8c7")
    f = font(52, True)
    lines = wrap(d, phrase, f, 870)
    if len(lines) > 4:
        f = font(45, True)
        lines = wrap(d, phrase, f, 870)
    for n, line in enumerate(lines):
        d.text((100, 1520 + n*65), line, font=f, fill="#ffffff")
    d.text((60, 1830), "BEZ POTOLKA FILMS  /  PILOT 001", font=font(26), fill="#88a9a5")
    for n in range(3):
        x = 60 + n*326
        d.rounded_rectangle((x, 1890, x+308, 1902), 6, fill="#a2e8cc" if n < index else "#304745")
    base_path = WORK / f"base-{index}.png"
    base.save(base_path)
    src = ImageOps.exif_transpose(Image.open(ROOT / "assets" / filename)).convert("RGBA")
    # A dark panel with the complete reaction image; no face/body cropping.
    panel = Image.new("RGBA", (PW, PH), "#17292e")
    art = ImageOps.contain(src, (PW-64, PH-64), Image.Resampling.LANCZOS)
    panel.alpha_composite(art, ((PW-art.width)//2, (PH-art.height)//2))
    panel_path = WORK / f"panel-{index}.png"
    panel.convert("RGB").save(panel_path)
    return base_path, panel_path

def synthesize(index, phrase):
    wav = WORK / f"voice-{index}.wav"
    run(ENGINE / "qwen_tts", "-d", "qwen3-tts-1.7b", "-s", "ono_anna",
        "-l", "Russian", "--int8", "--threads", "4", "--seed", "42",
        "--max-duration", "35", "--text", phrase, "-o", wav, cwd=ENGINE)
    seconds = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(wav)], text=True).strip())
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f"Invalid voice duration: {seconds}")
    frames = math.ceil((seconds + .35) * FPS)
    return wav, frames

def main():
    OUT.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    for filename, _, _ in SCENES:
        if not (ROOT / "assets" / filename).is_file():
            raise FileNotFoundError(filename)
    clips, manifest = [], []
    for index, (filename, title, phrase) in enumerate(SCENES, 1):
        print(f"Scene {index}: {filename} | {phrase}", flush=True)
        base, panel = compose(index, filename, title, phrase)
        wav, frames = synthesize(index, phrase)
        duration = frames/FPS
        clip = WORK / f"scene-{index}.mp4"
        zoom = f"1+0.035*on/{max(1,frames-1)}"
        vf = (f"[1:v]scale=1920:2240,zoompan=z='{zoom}':x='iw/2-iw/zoom/2':"
              f"y='ih/2-ih/zoom/2':d={frames}:s={PW}x{PH}:fps={FPS}[motion];"
              f"[0:v][motion]overlay={PX}:{PY}:shortest=1,format=yuv420p[v]")
        run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-filter_complex_threads", "2", "-loop", "1", "-framerate", FPS, "-i", base,
            "-i", panel, "-i", wav, "-filter_complex", vf, "-map", "[v]", "-map", "2:a:0",
            "-af", "apad", "-t", f"{duration:.6f}", "-r", FPS, "-c:v", "libx264",
            "-threads", "4", "-preset", "fast", "-crf", "19", "-c:a", "aac", "-b:a", "128k",
            "-ar", "48000", "-ac", "2", clip)
        clips.append(clip)
        manifest.append({"scene": index, "asset": filename, "text": phrase, "seconds": duration})
    listing = WORK / "concat.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in clips), encoding="utf-8")
    final = OUT / "hockey-tyan-pilot.mp4"
    run("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", listing, "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", final)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    run("ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height:format=duration",
        "-of", "json", final)

if __name__ == "__main__":
    main()

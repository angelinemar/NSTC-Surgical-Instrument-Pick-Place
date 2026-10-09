"""Build compact six-camera A/B PNG and GIFs from controlled preview captures."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CAMERAS = (("FRONT", "camera"), ("WRIST", "grip_cam_b"), ("TOP", "cam_top"),
           ("LEFT", "cam_left"), ("RIGHT", "cam_right"), ("TRAY", "cam_tray"))


def font(size, bold=False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def labeled(image, title, width=448):
    image = image.convert("RGB").resize((width, width), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, width + 42), "white")
    canvas.paste(image, (0, 42))
    ImageDraw.Draw(canvas).text((12, 10), title, fill="#172033", font=font(20, True))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", type=Path, required=True)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = []
    recorder_frames, usd_frames, ab_frames = [], [], []
    for label, key in CAMERAS:
        a = Image.open(args.recorder / f"baseline_{key}.png")
        b = Image.open(args.usd / f"baseline_{key}.png")
        la, lb = labeled(a, f"{label} | RECORDER MATERIAL"), labeled(b, f"{label} | PURE USD")
        pair = Image.new("RGB", (896, 490), "white")
        pair.paste(la, (0, 0)); pair.paste(lb, (448, 0))
        pairs.append(pair)
        recorder_frames.append(la.quantize(colors=192))
        usd_frames.append(lb.quantize(colors=192))
        ab_frames.extend([la.quantize(colors=192), lb.quantize(colors=192)])
    sheet = Image.new("RGB", (1792, 1470), "white")
    for index, pair in enumerate(pairs):
        sheet.paste(pair, ((index % 2) * 896, (index // 2) * 490))
    sheet.save(args.output / "material_ab_all_6_cameras.png")
    for name, frames in (("recorder_material_all_6_cameras.gif", recorder_frames),
                         ("pure_usd_all_6_cameras.gif", usd_frames),
                         ("material_ab_all_6_cameras.gif", ab_frames)):
        frames[0].save(args.output / name, save_all=True, append_images=frames[1:],
                       duration=900, loop=0, disposal=2, optimize=False)
    print(f"SAVED {args.output.resolve()}")


if __name__ == "__main__":
    main()

"""Build one start-to-finish GIF from every RGB camera in a split episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


CAMERAS = (
    ("CAM_FRONT", "front_rgb"),
    ("GRIPPER", "images_grip_b"),
    ("WRIST", "wrist_rgb"),
    ("TOP", "cam_top_rgb"),
    ("LEFT", "cam_left_rgb"),
    ("RIGHT", "cam_right_rgb"),
    ("TRAY", "cam_tray_rgb"),
)


def as_rgb(array: np.ndarray) -> Image.Image:
    array = np.asarray(array)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array[..., :3], "RGB")


def decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_root", type=Path,
                        help="Folder containing pick_policy and place_policy")
    parser.add_argument("--object", default="scalpel")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--stride", type=int, default=3,
                        help="Use every Nth simulation frame")
    parser.add_argument("--panel-width", type=int, default=288)
    parser.add_argument("--duration-ms", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    filename = f"episode_{args.episode:06d}.h5"
    files = [
        args.episode_root / "pick_policy" / args.object / filename,
        args.episode_root / "place_policy" / args.object / filename,
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing split H5: " + ", ".join(missing))

    output = args.output or (args.episode_root / f"episode_{args.episode:06d}_all_cameras.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    global_frame = 0
    camera_count = 0

    for phase, path in zip(("PICK", "PLACE"), files):
        with h5py.File(path, "r") as h5:
            obs = h5["observations"]
            available = [(label, key) for label, key in CAMERAS if key in obs]
            # The legacy gripper topic aliases the wrist camera in P4.
            if "wrist_rgb" in obs:
                available = [(label, key) for label, key in available if key != "images_grip_b"]
            if not available:
                raise KeyError(f"No RGB cameras found in {path}")
            camera_count = len(available)
            length = min(obs[key].shape[0] for _, key in available)
            indices = list(range(0, length, max(1, args.stride)))
            if indices[-1] != length - 1:
                indices.append(length - 1)
            source_h, source_w = obs[available[0][1]].shape[1:3]
            panel_h = round(args.panel_width * source_h / source_w)
            header_h = 28
            rows = (camera_count + 2) // 3
            canvas_size = (args.panel_width * 3, (panel_h + header_h) * rows + header_h)

            for index in indices:
                canvas = Image.new("RGB", canvas_size, (18, 18, 18))
                draw = ImageDraw.Draw(canvas)
                stage = decode_text(h5["stage_names"][index]) if "stage_names" in h5 else ""
                for camera_index, (label, key) in enumerate(available):
                    row, col = divmod(camera_index, 3)
                    x, y = col * args.panel_width, row * (panel_h + header_h)
                    image = as_rgb(obs[key][index]).resize(
                        (args.panel_width, panel_h), Image.Resampling.BILINEAR)
                    canvas.paste(image, (x, y + header_h))
                    draw.text((x + 7, y + 7), label, fill=(255, 255, 255))
                draw.text(
                    (7, rows * (panel_h + header_h) + 7),
                    f"{phase} | frame {global_frame:04d} | {stage}",
                    fill=(255, 220, 80),
                )
                frames.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
                global_frame += 1

    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    print(f"GIF SAVED: {output.resolve()}")
    print(f"frames={len(frames)} cameras={camera_count} stride={args.stride}")


if __name__ == "__main__":
    main()

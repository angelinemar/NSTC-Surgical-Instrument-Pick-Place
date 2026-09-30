"""Build one pick-to-place semantic GIF from every available Phase-3 camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


CAMERAS = (
    ("CAM_FRONT", "front_semantic"),
    ("GRIPPER", "grip_b_semantic"),
    ("TOP", "cam_top_semantic"),
    ("LEFT", "cam_left_semantic"),
    ("RIGHT", "cam_right_semantic"),
    ("TRAY", "cam_tray_semantic"),
)

COLORS = {
    "background": (20, 20, 20), "robot": (60, 150, 255),
    "surgical_tray": (255, 150, 40), "scalpel": (80, 220, 100),
    "scissor": (245, 220, 60), "love_retractor": (225, 80, 210),
    "kelly": (80, 225, 220), "scalpel_type2": (255, 90, 90),
    "table": (45, 150, 75), "floor": (110, 110, 120),
    "room": (180, 170, 155),
}


def decode_text(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def colorize(segmentation: np.ndarray, classes: dict[str, int]) -> Image.Image:
    segmentation = np.asarray(segmentation)
    rgb = np.zeros((*segmentation.shape[:2], 3), dtype=np.uint8)
    for label, class_id in classes.items():
        rgb[segmentation == class_id] = COLORS.get(label, (255, 255, 255))
    unknown = ~np.isin(segmentation, tuple(classes.values()))
    rgb[unknown] = (255, 255, 255)
    return Image.fromarray(rgb, "RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("--object", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--panel-width", type=int, default=288)
    parser.add_argument("--duration-ms", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    filename = f"episode_{args.episode:06d}.h5"
    files = (
        args.episode_root / "pick_policy" / args.object / filename,
        args.episode_root / "place_policy" / args.object / filename,
    )
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing split H5: " + ", ".join(missing))

    output = args.output or (
        args.episode_root / f"episode_{args.episode:06d}_all_cameras_semantic.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    global_frame = 0

    for phase, path in zip(("PICK", "PLACE"), files):
        with h5py.File(path, "r") as h5:
            if "semantic_class_ids" not in h5.attrs:
                raise KeyError(f"semantic_class_ids metadata missing in {path}")
            classes = json.loads(h5.attrs["semantic_class_ids"])
            obs = h5["observations"]
            available = [(label, key) for label, key in CAMERAS if key in obs]
            if not available:
                raise KeyError(f"No semantic cameras found in {path}")
            length = min(obs[key].shape[0] for _, key in available)
            indices = list(range(0, length, max(1, args.stride)))
            if indices[-1] != length - 1:
                indices.append(length - 1)

            source_h, source_w = obs[available[0][1]].shape[1:3]
            panel_h = round(args.panel_width * source_h / source_w)
            header_h = 28
            canvas = (args.panel_width * 3, (panel_h + header_h) * 3)

            for index in indices:
                frame = Image.new("RGB", canvas, (18, 18, 18))
                draw = ImageDraw.Draw(frame)
                stage = decode_text(h5["stage_names"][index]) if "stage_names" in h5 else ""
                for camera_index, (label, key) in enumerate(available):
                    row, col = divmod(camera_index, 3)
                    x, y = col * args.panel_width, row * (panel_h + header_h)
                    panel = colorize(obs[key][index], classes).resize(
                        (args.panel_width, panel_h), Image.Resampling.NEAREST)
                    frame.paste(panel, (x, y + header_h))
                    draw.text((x + 7, y + 7), label, fill=(255, 255, 255))

                legend_y = 2 * (panel_h + header_h) + 8
                draw.text((7, legend_y), f"{phase} | frame {global_frame:04d}", fill=(255, 220, 80))
                draw.text((7, legend_y + 20), stage, fill=(230, 230, 230))
                ordered_classes = sorted(classes.items(), key=lambda item: item[1])
                for legend_index, (label, class_id) in enumerate(ordered_classes):
                    col, row = divmod(legend_index, 4)
                    lx = args.panel_width + col * 145
                    ly = legend_y + row * 22
                    color = COLORS.get(label, (255, 255, 255))
                    draw.rectangle((lx, ly, lx + 14, ly + 14), fill=color)
                    draw.text((lx + 19, ly), f"{class_id}:{label}", fill=(235, 235, 235))

                frames.append(frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
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
    print(f"SEMANTIC GIF SAVED: {output.resolve()}")
    print(f"frames={len(frames)} semantic_cameras={len(CAMERAS)} stride={args.stride}")


if __name__ == "__main__":
    main()

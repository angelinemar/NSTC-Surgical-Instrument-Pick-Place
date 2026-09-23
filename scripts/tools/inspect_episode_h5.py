"""Print the complete schema and a compact content summary of one Phase-3 H5."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [decode(item) for item in value.tolist()]
    return value


def preview(dataset: h5py.Dataset):
    if dataset.size == 0:
        return "empty"
    if dataset.dtype.kind in "SUO":
        first = decode(dataset[0])
        last = decode(dataset[-1])
    elif dataset.ndim == 1:
        first, last = dataset[0], dataset[-1]
    else:
        first, last = dataset[0].tolist(), dataset[-1].tolist()
    return f"first={first} | last={last}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to episode_XXXXXX.h5")
    parser.add_argument("--values", action="store_true", help="Preview first/last value for every non-image dataset")
    args = parser.parse_args()

    path = args.path.resolve()
    if not path.is_file():
        raise SystemExit(f"H5 not found: {path}")

    with h5py.File(path, "r") as h5:
        print(f"FILE: {path}")
        print("\n=== ROOT ATTRIBUTES / METADATA ===")
        for key in sorted(h5.attrs):
            print(f"{key}: {decode(h5.attrs[key])}")

        print("\n=== ALL DATASETS (topic, shape, dtype) ===")
        datasets = []

        def collect(name, item):
            if isinstance(item, h5py.Dataset):
                datasets.append((name, item))

        h5.visititems(collect)
        for name, dataset in datasets:
            print(f"{name:<48} shape={str(dataset.shape):<22} dtype={dataset.dtype}")
            is_image = dataset.ndim >= 3
            if args.values and not is_image:
                print(f"  {preview(dataset)}")

        if "stage_names" in h5:
            stages = [str(decode(value)) for value in h5["stage_names"][:]]
            counts = Counter(stages)
            print("\n=== STAGE TIMELINE ===")
            start = 0
            for index in range(1, len(stages) + 1):
                if index == len(stages) or stages[index] != stages[start]:
                    print(f"steps {start:04d}..{index - 1:04d} ({index - start:3d})  {stages[start]}")
                    start = index
            print("\n=== STAGE TOTALS ===")
            for stage, count in counts.items():
                print(f"{stage:<32} {count}")

        print("\n=== TRAINING TENSORS ===")
        for name in ("observations/state", "actions", "observations/object_type_id", "observations/skill_id"):
            if name in h5:
                dataset = h5[name]
                print(f"{name}: {preview(dataset)}")


if __name__ == "__main__":
    main()

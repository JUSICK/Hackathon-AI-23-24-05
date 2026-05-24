import argparse
import random
from pathlib import Path

import yaml
from ultralytics.data.utils import visualize_image_annotations


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_label_map(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})

    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}

    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}

    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    label_map = load_label_map(args.data_yaml)

    images = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if not images:
        print("No images found.")
        return

    random.shuffle(images)
    images = images[:args.count]

    print(f"Selected images: {len(images)}")

    for image_path in images:
        label_path = labels_dir / image_path.relative_to(images_dir)
        label_path = label_path.with_suffix(".txt")

        if not label_path.exists():
            print(f"[MISSING LABEL] {image_path} -> {label_path}")
            continue

        print(f"Visualizing: {image_path}")
        print(f"Label:       {label_path}")

        visualize_image_annotations(
            str(image_path),
            str(label_path),
            label_map,
        )


if __name__ == "__main__":
    main()
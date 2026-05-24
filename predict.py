# ABOUTME: Baseline prediction script using ultralytics YOLOv8n for object detection.
# ABOUTME: CLI interface: predict --input DIR --output predictions.json. Outputs COCO-format predictions.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_category_map(
    taxonomy_path: Path = Path("taxonomy.json"),
    data_yaml_path: Path = Path("data.yaml"),
) -> dict[int, int]:
    """
    Load mapping:
    YOLO class index -> official hackathon category_id

    YOLO model returns class indexes: 0, 1, 2...
    Hackathon predictions.json requires official category_id from taxonomy.json.

    This function maps:
    YOLO class index -> class name from data.yaml -> category_id from taxonomy.json
    """

    if not taxonomy_path.exists():
        print(f"WARNING: {taxonomy_path} not found", file=sys.stderr)
        return {}

    if not data_yaml_path.exists():
        print(f"WARNING: {data_yaml_path} not found", file=sys.stderr)
        return {}

    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))

    name_to_category_id = {
        cat["name"]: cat["id"]
        for cat in taxonomy["categories"]
    }

    data_yaml = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    names = data_yaml["names"]

    if isinstance(names, list):
        yolo_id_to_name = {
            i: name
            for i, name in enumerate(names)
        }
    else:
        yolo_id_to_name = {
            int(i): name
            for i, name in names.items()
        }

    category_map: dict[int, int] = {}
    missing = []

    for yolo_id, class_name in yolo_id_to_name.items():
        category_id = name_to_category_id.get(class_name)

        if category_id is None:
            missing.append((yolo_id, class_name))
            continue

        category_map[yolo_id] = category_id

    if missing:
        print("WARNING: Some YOLO classes were not found in taxonomy.json:", file=sys.stderr)
        for yolo_id, class_name in missing[:30]:
            print(f"  YOLO {yolo_id}: {class_name}", file=sys.stderr)

        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more", file=sys.stderr)

    print(f"Loaded category map: {len(category_map)} classes", file=sys.stderr)

    return category_map


def load_image_id_map(annotations_path: Path) -> dict[str, int]:
    """
    Load mapping from image filename to image_id from annotations/test_images JSON.

    Args:
        annotations_path: Path to COCO-format annotations JSON.

    Returns:
        Dict mapping filename to integer image_id.
    """

    if not annotations_path.exists():
        print(f"WARNING: {annotations_path} not found, image_id mapping unavailable", file=sys.stderr)
        return {}

    data = json.loads(annotations_path.read_text(encoding="utf-8"))

    filename_to_id = {}

    for img in data["images"]:
        file_name = img["file_name"]

        filename_to_id[file_name] = img["id"]
        filename_to_id[Path(file_name).name] = img["id"]

    return filename_to_id


def find_images(input_dir: Path) -> list[Path]:
    """
    Find image files recursively.
    """

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
    }

    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )


def predict_directory(
    input_dir: Path,
    model_path: str = "weights/best.pt",
    confidence: float = 0.05,
    taxonomy_path: Path = Path("taxonomy.json"),
    annotations_path: Path = Path("test_images.json"),
    data_yaml_path: Path = Path("data/yolo/dataset.yaml"),
    image_size: int = 1280,
) -> list[dict]:
    """
    Run YOLO inference on all images in a directory.

    Args:
        input_dir: Directory containing input images.
        model_path: Path to YOLO model weights.
        confidence: Minimum confidence threshold.
        taxonomy_path: Path to taxonomy.json for official category_id mapping.
        annotations_path: Path to test_images.json for image_id mapping.
        data_yaml_path: Path to YOLO data.yaml for class names.
        image_size: YOLO inference image size.

    Returns:
        List of COCO-format prediction dicts.
    """

    model = YOLO(model_path)

    cat_map = load_category_map(
        taxonomy_path=taxonomy_path,
        data_yaml_path=data_yaml_path,
    )

    filename_to_id = load_image_id_map(annotations_path)

    image_files = find_images(input_dir)

    if not image_files:
        print(f"WARNING: No images found in {input_dir}", file=sys.stderr)
        return []

    if not cat_map:
        print("WARNING: category map is empty. Predictions may be skipped.", file=sys.stderr)

    predictions: list[dict] = []

    print(f"Found {len(image_files)} images in {input_dir}", file=sys.stderr)

    for idx, img_path in enumerate(image_files, start=1):
        rel_name = str(img_path.relative_to(input_dir))
        base_name = img_path.name

        image_id = filename_to_id.get(rel_name)
        if image_id is None:
            image_id = filename_to_id.get(base_name)

        if image_id is None:
            print(f"WARNING: {img_path} not in annotations, skipping", file=sys.stderr)
            continue

        results = model.predict(
            source=str(img_path),
            conf=confidence,
            imgsz=image_size,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    continue

                cls_id = int(boxes.cls[i].item())
                category_id = cat_map.get(cls_id)

                if category_id is None:
                    continue

                score = float(boxes.conf[i].item())

                predictions.append({
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(float(w), 2),
                        round(float(h), 2),
                    ],
                    "score": round(score, 4),
                })

        if idx % 50 == 0 or idx == len(image_files):
            print(f"[{idx}/{len(image_files)}] processed", file=sys.stderr)

    return predictions


def main() -> None:
    """
    CLI entry point.

    Example:
    python predict.py --input data/public_test --output predictions.json
    """

    parser = argparse.ArgumentParser(
        description="Run object detection and produce COCO-format predictions"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing input images",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.json"),
        help="Output path for predictions JSON",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="weights/best.pt",
        help="Path to YOLO model weights",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.05,
        help="Minimum confidence threshold",
    )

    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("taxonomy.json"),
        help="Path to taxonomy.json",
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("test_images.json"),
        help="Path to test_images.json / COCO images JSON for image_id mapping",
    )

    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("data.yaml"),
        help="Path to YOLO data.yaml for class-name mapping",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference image size",
    )

    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"ERROR: {args.input} is not a directory", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    predictions = predict_directory(
        input_dir=args.input,
        model_path=args.model,
        confidence=args.confidence,
        taxonomy_path=args.taxonomy,
        annotations_path=args.annotations,
        data_yaml_path=args.data_yaml,
        image_size=args.imgsz,
    )

    args.output.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

COCO_JSON = Path("data/train/annotations.json")  # поменяй, если файл называется иначе
IMAGES_DIR = Path("data/train/images")
OUT_DIR = Path("data/yolo")

VAL_RATIO = 0.2
SEED = 42

random.seed(SEED)

with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

categories = sorted(coco["categories"], key=lambda c: c["id"])
cat_id_to_yolo_id = {cat["id"]: i for i, cat in enumerate(categories)}
names = [cat.get("name", str(cat["id"])) for cat in categories]

anns_by_image = defaultdict(list)

for ann in coco["annotations"]:
    if ann.get("iscrowd", 0) == 1:
        continue

    if "bbox" not in ann:
        continue

    image_id = ann["image_id"]

    if image_id not in images:
        continue

    img = images[image_id]
    img_w = img["width"]
    img_h = img["height"]

    x, y, w, h = ann["bbox"]

    if w <= 0 or h <= 0:
        continue

    class_id = cat_id_to_yolo_id[ann["category_id"]]

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    anns_by_image[image_id].append(
        f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
    )

image_ids = list(images.keys())
random.shuffle(image_ids)

val_count = int(len(image_ids) * VAL_RATIO)
val_ids = set(image_ids[:val_count])
train_ids = set(image_ids[val_count:])

for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
    for image_id in split_ids:
        img = images[image_id]
        file_name = img["file_name"]

        src_img = IMAGES_DIR / file_name

        if not src_img.exists():
            src_img = IMAGES_DIR / Path(file_name).name

        if not src_img.exists():
            print(f"Brak obrazu: {file_name}")
            continue

        dst_img = OUT_DIR / "images" / split_name / Path(file_name).name
        dst_label = OUT_DIR / "labels" / split_name / Path(file_name).with_suffix(".txt").name

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_img, dst_img)

        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("\n".join(anns_by_image.get(image_id, [])))

yaml_path = OUT_DIR / "dataset.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {OUT_DIR.resolve().as_posix()}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n\n")
    f.write("names:\n")
    for i, name in enumerate(names):
        f.write(f"  {i}: {name}\n")

print("Gotowe.")
print(f"Dataset YAML: {yaml_path}")
print(f"Classes: {len(names)}")

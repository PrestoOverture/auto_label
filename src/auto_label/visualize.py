import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BOX_COLORS = [
    (30, 144, 255),   # dodger blue
    (255, 80, 80),    # coral red
    (50, 205, 50),    # lime green
    (255, 165, 0),    # orange
    (200, 80, 255),   # purple
    (0, 206, 209),    # turquoise
    (255, 215, 0),    # gold
    (255, 105, 180),  # hot pink
]

# cv2.putText's built-in Hershey fonts only render ASCII, so Chinese class
# names come out as "?". PIL + a real TTF font is used for label text instead.
_CJK_FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def get_label_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _font_cache:
        return _font_cache[size]
    for path in _CJK_FONT_CANDIDATES:
        if Path(path).exists():
            font = ImageFont.truetype(path, size)
            _font_cache[size] = font
            return font
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def draw_detections(image: np.ndarray, detections: list[dict], class_names: list[str]) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]

    box_thickness = max(2, int(min(w, h) / 250))
    font_size = max(16, int(min(w, h) / 30))
    pad = 6

    for det in detections:
        class_id = det["class_id"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        color = BOX_COLORS[class_id % len(BOX_COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_label_font(font_size)

    for det in detections:
        class_id = det["class_id"]
        conf = det["confidence"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        label = f"{class_name} {conf:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top

        label_y = y1 - pad - th if y1 - th - pad > 0 else y2 + pad
        draw.rectangle([x1, label_y, x1 + tw + pad * 2, label_y + th + pad * 2], fill=(0, 0, 0))
        draw.text((x1 + pad - left, label_y + pad - top), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def read_yolo_label(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    if not label_path.exists():
        return []

    detections = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
            x1 = (xc - bw / 2) * img_w
            y1 = (yc - bh / 2) * img_h
            x2 = (xc + bw / 2) * img_w
            y2 = (yc + bh / 2) * img_h
            detections.append({
                "class_id": class_id,
                "bbox_xyxy": [x1, y1, x2, y2],
                "confidence": 1.0,
            })
    return detections


def visualize_split(
    detection_dir: Path,
    split: str,
    class_names: list[str],
    max_images: int | None = None,
) -> tuple[list[np.ndarray], dict]:
    img_dir = detection_dir / split / "images"
    label_dir = detection_dir / split / "labels"

    image_paths = sorted(img_dir.glob("*.jpg")) if img_dir.exists() else []

    with_labels = sum(
        1 for p in image_paths
        if (label_dir / f"{p.stem}.txt").exists() and (label_dir / f"{p.stem}.txt").stat().st_size > 0
    )
    stats = {"total": len(image_paths), "with_labels": with_labels, "empty": len(image_paths) - with_labels}

    items = image_paths[:max_images] if max_images else image_paths
    vis_images = []
    for img_path in items:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        detections = read_yolo_label(label_dir / f"{img_path.stem}.txt", w, h)
        vis = draw_detections(img, detections, class_names)
        vis_images.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    return vis_images, stats


def visualize_dataset(
    detections_json: Path,
    output_dir: Path,
    class_names: list[str],
    max_images: int | None = None,
) -> list[Path]:
    with open(detections_json, encoding="utf-8") as f:
        results = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    items = results[:max_images] if max_images else results
    for item in items:
        img_path = Path(item["image_path"])
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        vis = draw_detections(img, item["detections"], class_names)
        out_path = output_dir / f"vis_{img_path.name}"
        cv2.imwrite(str(out_path), vis)
        written.append(out_path)

    return written

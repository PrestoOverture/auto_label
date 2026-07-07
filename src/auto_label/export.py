from pathlib import Path

import yaml


def xyxy_to_yolo(bbox_xyxy: list[float], img_w: int, img_h: int) -> tuple[float, ...]:
    x1, y1, x2, y2 = bbox_xyxy
    x_center = max(0.0, min(1.0, (x1 + x2) / 2.0 / img_w))
    y_center = max(0.0, min(1.0, (y1 + y2) / 2.0 / img_h))
    width = max(0.0, min(1.0, (x2 - x1) / img_w))
    height = max(0.0, min(1.0, (y2 - y1) / img_h))
    return x_center, y_center, width, height


def write_labels(
    results: list[dict],
    split_map: dict[str, list[Path]],
    detection_dir: Path,
) -> dict[str, int]:
    frame_to_split = {}
    for split_name, paths in split_map.items():
        for p in paths:
            frame_to_split[p.name] = split_name

    stats = {"train": 0, "val": 0, "test": 0, "empty": 0}

    for item in results:
        img_name = Path(item["image_path"]).name
        label_name = Path(img_name).stem + ".txt"
        split_name = frame_to_split.get(img_name)
        if not split_name:
            continue

        label_dir = detection_dir / split_name / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / label_name

        lines = []
        for det in item["detections"]:
            xc, yc, w, h = xyxy_to_yolo(det["bbox_xyxy"], item["width"], item["height"])
            lines.append(f"{det['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        stats[split_name] += 1
        if not lines:
            stats["empty"] += 1

    return stats


def write_data_yaml(detection_dir: Path, class_names: list[str]) -> Path:
    data = {
        "path": ".",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    out_path = detection_dir / "data.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return out_path

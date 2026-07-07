import re
import shutil
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from auto_label.action import (
    ACTION_MODELS,
    DEFAULT_ACTION_MODEL,
    VIDEO_EXTS,
    detect_actions_video,
    get_api_key,
    write_action_outputs,
)
from auto_label.config import get_dataset_config, get_dataset_paths, load_datasets_yaml
from auto_label.consensus import apply_consensus
from auto_label.detect import (
    MODEL_REGISTRY,
    detect_all_generic,
    is_cached,
    load_detector,
    save_detections,
)
from auto_label.export import write_data_yaml, write_labels
from auto_label.extract import collect_images
from auto_label.split import apply_split, compute_split
from auto_label.visualize import BOX_COLORS, draw_detections, get_label_font, read_yolo_label

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

_model_cache: dict[str, tuple] = {}


def _get_model(model_key: str, device: str | None = None):
    if model_key not in _model_cache:
        model, processor, resolved = load_detector(model_key, device)
        _model_cache[model_key] = (model, processor, resolved)
    return _model_cache[model_key]


def _draw_multi_model(image: np.ndarray, per_model: dict, class_names: list[str]) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    box_thickness = max(2, int(min(w, h) / 250))
    font_size = max(14, int(min(w, h) / 38))
    pad = 6

    model_colors = {
        "grounding_dino": (30, 144, 255),
        "owlv2": (50, 205, 50),
        "florence2": (255, 80, 80),
    }

    for model_key, dets in per_model.items():
        color = model_colors.get(model_key, (255, 255, 255))
        for det in dets:
            x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_label_font(font_size)

    for model_key, dets in per_model.items():
        color = model_colors.get(model_key, (255, 255, 255))
        color_rgb = (color[2], color[1], color[0])
        display_name = MODEL_REGISTRY.get(model_key, {}).get("display_name", model_key)
        for det in dets:
            x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
            label = f"[{display_name}] {class_names[det['class_id']]} {det['confidence']:.2f}"
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            tw, th = right - left, bottom - top
            label_y = y1 - pad - th if y1 - th - pad > 0 else y2 + pad
            draw.rectangle([x1, label_y, x1 + tw + pad * 2, label_y + th + pad * 2], fill=(0, 0, 0))
            draw.text((x1 + pad - left, label_y + pad - top), label, fill=color_rgb, font=font)

    legend_y = 30
    for model_key, color in model_colors.items():
        if model_key in per_model:
            color_rgb = (color[2], color[1], color[0])
            display_name = MODEL_REGISTRY.get(model_key, {}).get("display_name", model_key)
            draw.rectangle([10, legend_y - 18, 30, legend_y], fill=color_rgb)
            draw.text((35, legend_y - 18), display_name, fill=(255, 255, 255), font=font)
            legend_y += 30

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def run_pipeline(
    file_input,
    class_names_str: str,
    dataset_name: str,
    selected_models: list[str],
    conflict_strategy: str,
    conf_threshold: float,
    iou_threshold: float,
    stride: int,
    batch_size: int,
    split_ratio_str: str,
):
    if not file_input:
        raise gr.Error("请上传包含视频或图片的文件夹")
    if not class_names_str.strip():
        raise gr.Error("请输入类别名称")
    if not selected_models:
        raise gr.Error("请至少选择一个模型")

    class_names = [c.strip() for c in class_names_str.split(",") if c.strip()]
    ratios = tuple(float(x) for x in split_ratio_str.split(","))
    stride = int(stride)
    batch_size = int(batch_size)

    work_dir = Path(tempfile.mkdtemp(prefix="auto_label_"))
    staging_dir = work_dir / "_frames"
    staging_dir.mkdir()

    upload_paths = [Path(p) for p in file_input]
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        return "\n".join(log_lines)

    no_update = gr.update()
    status = log(f"类别: {class_names}")
    status = log(f"模型: {', '.join(selected_models)}")
    yield status, no_update, no_update, no_update, no_update

    frame_paths = collect_images(upload_paths, staging_dir, stride)
    if not frame_paths:
        raise gr.Error("未找到可处理的图片或视频帧，请确认文件夹里包含 .mp4/.jpg/.png 等文件")
    status = log(f"待处理图片: {len(frame_paths)} 张")
    yield status, no_update, no_update, no_update, no_update

    model_key_map = {
        "Grounding DINO": "grounding_dino",
        "OWLv2": "owlv2",
        "Florence-2": "florence2",
    }
    model_keys = [model_key_map[m] for m in selected_models]

    multi_model = len(model_keys) > 1
    per_model_results = {}

    for mk in model_keys:
        display_name = MODEL_REGISTRY[mk]["display_name"]
        if mk not in _model_cache and not is_cached(MODEL_REGISTRY[mk]["model_id"]):
            status = log(f"{display_name} 尚未下载，首次使用需从 HuggingFace 拉取权重，可能需要几分钟...")
            yield status, no_update, no_update, no_update, no_update
        status = log(f"加载模型: {display_name}...")
        yield status, no_update, no_update, no_update, no_update
        model, processor, device = _get_model(mk)
        status = log(f"运行检测 ({display_name})，共 {len(frame_paths)} 张图片...")
        yield status, no_update, no_update, no_update, no_update
        results = detect_all_generic(mk, model, processor, frame_paths, class_names, device, conf_threshold, batch_size)
        per_model_results[mk] = results
        total = sum(len(r["detections"]) for r in results)
        status = log(f"  {display_name}: {total} 个检测")
        yield status, no_update, no_update, no_update, no_update

    if multi_model:
        status = log(f"共识策略: {conflict_strategy}, IoU阈值: {iou_threshold}")
        strategy_key = "majority_vote" if conflict_strategy == "多数投票" else "flag_conflicts"
        final_results, review_results = apply_consensus(per_model_results, strategy_key, iou_threshold)
        total_final = sum(len(r["detections"]) for r in final_results)
        status = log(f"共识结果: {total_final} 个检测, {len(review_results)} 张需审核")
        yield status, no_update, no_update, no_update, no_update
    else:
        final_results = list(per_model_results.values())[0]
        review_results = []

    detection_dir = work_dir / "detection"
    for split in ("train", "val", "test"):
        (detection_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (detection_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Only images with a consensus result go into the exported dataset — images
    # sent to review have no agreed label yet and would otherwise end up as
    # image-without-label entries in train/val/test.
    exportable_paths = [Path(r["image_path"]) for r in final_results]
    split_map = compute_split(exportable_paths, ratios)
    apply_split(split_map, detection_dir)

    stats = write_labels(final_results, split_map, detection_dir)
    write_data_yaml(detection_dir, class_names)
    status = log(f"导出标签 - train: {stats['train']}, val: {stats['val']}, test: {stats['test']}")
    yield status, no_update, no_update, no_update, no_update

    detections_path = detection_dir / "_detections.json"
    save_detections(final_results, detections_path)

    vis_images = []
    for item in final_results[:50]:
        img_path = Path(item["image_path"])
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        vis = draw_detections(img, item["detections"], class_names)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        vis_images.append(vis_rgb)

    review_images = []
    if review_results:
        for item in review_results[:30]:
            img_path = Path(item["image_path"])
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            per_model = item.get("per_model", {})
            vis = _draw_multi_model(img, per_model, class_names)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            review_images.append(vis_rgb)

    zip_path = work_dir / f"{dataset_name}_yolo.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=detection_dir)

    status = log("完成!")

    stats_data = [
        ["总检测数", str(sum(len(r["detections"]) for r in final_results))],
        ["有检测的图片", str(sum(1 for r in final_results if r["detections"]))],
        ["空帧", str(sum(1 for r in final_results if not r["detections"]))],
        ["训练集", str(stats["train"])],
        ["验证集", str(stats["val"])],
        ["测试集", str(stats["test"])],
    ]
    if multi_model:
        stats_data.append(["待审核", str(len(review_results))])

    yield (
        status,
        stats_data,
        vis_images if vis_images else None,
        review_images if review_images else None,
        str(zip_path) if zip_path.exists() else None,
    )


def on_model_change(selected_models):
    n = len(selected_models)
    show_consensus = n >= 2
    can_majority_vote = n > 2 and n % 2 == 1

    if not show_consensus:
        strategy_update = gr.update(visible=False)
    elif can_majority_vote:
        strategy_update = gr.update(visible=True, interactive=True)
    else:
        strategy_update = gr.update(visible=True, interactive=False, value="标记冲突")

    return strategy_update, gr.update(visible=show_consensus)


def list_dataset_names() -> list[str]:
    try:
        data = load_datasets_yaml(DATASETS_DIR)
        return sorted(data.get("datasets", {}).keys())
    except Exception:
        return []


def refresh_dataset_list():
    names = list_dataset_names()
    return gr.update(choices=names, value=names[0] if names else None)


def _list_review_items(detection_dir: Path, split: str, max_images: int) -> list[dict]:
    img_dir = detection_dir / split / "images"
    label_dir = detection_dir / split / "labels"
    if not img_dir.exists():
        return []

    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if max_images:
        image_paths = image_paths[:max_images]

    return [
        {
            "image_path": str(img_path),
            "label_path": str(label_dir / f"{img_path.stem}.txt"),
            "name": img_path.name,
        }
        for img_path in image_paths
    ]


def _read_review_detections(item: dict) -> tuple[np.ndarray, list[dict]]:
    img_path = Path(item["image_path"])
    img = cv2.imread(str(img_path))
    if img is None:
        raise gr.Error(f"无法读取图片: {img_path}")
    h, w = img.shape[:2]
    detections = read_yolo_label(Path(item["label_path"]), w, h)
    return img, detections


def _draw_numbered_detections(
    image: np.ndarray,
    detections: list[dict],
    class_names: list[str],
    deleted_indexes: set[int] | None = None,
) -> np.ndarray:
    deleted_indexes = deleted_indexes or set()
    img = image.copy()
    h, w = img.shape[:2]
    box_thickness = max(2, int(min(w, h) / 250))
    font_size = max(16, int(min(w, h) / 30))
    pad = 6

    for idx, det in enumerate(detections, start=1):
        class_id = det["class_id"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        color = (128, 128, 128) if idx in deleted_indexes else BOX_COLORS[class_id % len(BOX_COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)
        if idx in deleted_indexes:
            cv2.line(img, (x1, y1), (x2, y2), color, box_thickness)
            cv2.line(img, (x1, y2), (x2, y1), color, box_thickness)

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_label_font(font_size)

    for idx, det in enumerate(detections, start=1):
        class_id = det["class_id"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        label = f"{idx}. {class_name}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        label_y = y1 - pad - th if y1 - th - pad > 0 else y2 + pad
        fill = (80, 80, 80) if idx in deleted_indexes else (0, 0, 0)
        draw.rectangle([x1, label_y, x1 + tw + pad * 2, label_y + th + pad * 2], fill=fill)
        draw.text((x1 + pad - left, label_y + pad - top), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _render_review_item(review_state: dict, index: int, delete_choices: list[str] | None = None):
    if not review_state or not review_state.get("items"):
        return None, gr.update(choices=[], value=[]), "请先加载标注"

    items = review_state["items"]
    index = max(0, min(index, len(items) - 1))
    item = items[index]
    try:
        img, detections = _read_review_detections(item)
    except Exception:
        return None, gr.update(choices=[], value=[]), f"无法读取图片: {item.get('name', '')}"
    deleted = _selected_box_numbers(delete_choices or [])
    vis = _draw_numbered_detections(
        img,
        detections,
        review_state["class_names"],
        deleted_indexes=deleted,
    )
    choices = []
    for idx, det in enumerate(detections, start=1):
        class_id = det["class_id"]
        class_name = review_state["class_names"][class_id] if 0 <= class_id < len(review_state["class_names"]) else f"class_{class_id}"
        choices.append(f"{idx}: {class_name}")

    status = f"{index + 1}/{len(items)}: {item['name']}，共 {len(detections)} 个框"
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), gr.update(choices=choices, value=delete_choices or []), status


def _selected_box_numbers(selected: list[str]) -> set[int]:
    numbers = set()
    for value in selected or []:
        match = re.match(r"^(\d+)", str(value))
        if match:
            numbers.add(int(match.group(1)))
    return numbers


def _build_review_gallery(review_state: dict) -> list[tuple[np.ndarray, str]]:
    gallery = []
    readable_items = []
    for item in review_state.get("items", []):
        try:
            img, detections = _read_review_detections(item)
        except Exception:
            continue
        vis = _draw_numbered_detections(img, detections, review_state["class_names"])
        gallery.append((cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), item["name"]))
        readable_items.append(item)
    review_state["items"] = readable_items
    return gallery


def _review_summary(review_state: dict) -> str:
    items = review_state.get("items", [])
    label_count = 0
    for item in items:
        label_path = Path(item["label_path"])
        if label_path.exists() and label_path.stat().st_size > 0:
            label_count += 1
    split = review_state.get("split", "")
    return f"{split}: 当前加载 {len(items)} 张，{label_count} 张有标注，{len(items) - label_count} 张为空"


def load_review_dataset(dirty: bool, dataset_name: str, split: str, max_images: int):
    if dirty:
        raise gr.Error("当前图片有未保存修改，请先点击“保存修改”再重新加载")
    if not dataset_name:
        raise gr.Error("请选择一个数据集")

    try:
        config = get_dataset_config(DATASETS_DIR, dataset_name)
        paths = get_dataset_paths(DATASETS_DIR, dataset_name)
    except Exception as e:
        raise gr.Error(str(e))

    class_names = config.get("class_names") or []
    if not class_names:
        raise gr.Error(f"数据集 '{dataset_name}' 没有配置 class_names，无法显示标签名称")

    detection_dir = paths["detection"]
    items = _list_review_items(detection_dir, split, int(max_images))
    if not items:
        raise gr.Error(f"'{dataset_name}' 的 {split} 分割里没有图片")

    review_state = {
        "dataset_name": dataset_name,
        "split": split,
        "detection_dir": str(detection_dir),
        "class_names": class_names,
        "items": items,
        "current_index": 0,
    }
    gallery = _build_review_gallery(review_state)
    if not review_state["items"]:
        raise gr.Error(f"'{dataset_name}' 的 {split} 分割里没有可读取的图片")
    image, checkbox_update, item_status = _render_review_item(review_state, 0, [])
    summary = _review_summary(review_state)
    return gallery, summary, review_state, image, checkbox_update, item_status, False, None


def select_review_image(review_state: dict, dirty: bool, evt: gr.SelectData):
    if dirty:
        raise gr.Error("当前图片有未保存修改，请先点击“保存修改”再切换图片")
    if not review_state or not review_state.get("items"):
        raise gr.Error("请先加载标注")

    index = evt.index
    if isinstance(index, (list, tuple)):
        index = index[0]
    review_state["current_index"] = int(index)
    image, checkbox_update, item_status = _render_review_item(review_state, review_state["current_index"], [])
    return review_state, image, checkbox_update, item_status, False


def navigate_review_image(direction: int, review_state: dict, dirty: bool):
    if dirty:
        raise gr.Error("当前图片有未保存修改，请先点击“保存修改”再切换图片")
    if not review_state or not review_state.get("items"):
        raise gr.Error("请先加载标注")

    current = int(review_state.get("current_index", 0))
    next_index = max(0, min(current + direction, len(review_state["items"]) - 1))
    review_state["current_index"] = next_index
    image, checkbox_update, item_status = _render_review_item(review_state, next_index, [])
    return review_state, image, checkbox_update, item_status, False


def update_review_deletions(selected_boxes: list[str], review_state: dict):
    if not review_state or not review_state.get("items"):
        return None, "请先加载标注", False
    index = int(review_state.get("current_index", 0))
    image, _, item_status = _render_review_item(review_state, index, selected_boxes)
    removed_count = len(_selected_box_numbers(selected_boxes))
    if removed_count:
        item_status = f"{item_status}，待删除 {removed_count} 个框"
    return image, item_status, bool(removed_count)


def save_review_edits(review_state: dict, selected_boxes: list[str]):
    if not review_state or not review_state.get("items"):
        raise gr.Error("请先加载标注")

    index = int(review_state.get("current_index", 0))
    item = review_state["items"][index]
    img, detections = _read_review_detections(item)
    h, w = img.shape[:2]
    delete_numbers = _selected_box_numbers(selected_boxes)
    kept = [det for det_idx, det in enumerate(detections, start=1) if det_idx not in delete_numbers]

    result = {
        "image_path": item["image_path"],
        "width": w,
        "height": h,
        "detections": kept,
    }
    write_labels([result], {review_state["split"]: [Path(item["image_path"])]}, Path(review_state["detection_dir"]))

    vis_dir = Path(review_state["detection_dir"]) / "_visualize"
    if vis_dir.exists():
        vis = draw_detections(img, kept, review_state["class_names"])
        cv2.imwrite(str(vis_dir / f"vis_{Path(item['image_path']).name}"), vis)

    gallery = _build_review_gallery(review_state)
    image, checkbox_update, item_status = _render_review_item(review_state, index, [])
    message = f"已保存：删除 {len(delete_numbers)} 个框，保留 {len(kept)} 个框"
    return gallery, image, checkbox_update, f"{item_status}\n{message}", False, _review_summary(review_state)


def export_reviewed_dataset(review_state: dict):
    if not review_state or not review_state.get("detection_dir"):
        raise gr.Error("请先加载标注")

    detection_dir = Path(review_state["detection_dir"])
    if not detection_dir.exists():
        raise gr.Error("检测目录不存在，无法导出")

    zip_dir = Path(tempfile.mkdtemp(prefix="auto_label_review_export_"))
    export_root = zip_dir / "export"
    split = review_state["split"]
    src_split_dir = detection_dir / split
    dst_split_dir = export_root / split
    dst_split_dir.mkdir(parents=True)

    for subdir in ("images", "labels"):
        src = src_split_dir / subdir
        dst = dst_split_dir / subdir
        if src.exists():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True)

    data_yaml = detection_dir / "data.yaml"
    if data_yaml.exists():
        shutil.copy2(data_yaml, export_root / "data.yaml")
    else:
        write_data_yaml(export_root, review_state.get("class_names", []))

    zip_base = zip_dir / f"{review_state['dataset_name']}_reviewed_yolo"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=export_root)
    return zip_path, f"已导出复核后的 YOLO 数据集: {Path(zip_path).name}"


def run_action_detection(file_input, actions_str: str, model_name: str, fps: float):
    if not file_input:
        raise gr.Error("请上传至少一个视频")
    actions = [a.strip() for a in re.split(r"[\n,，、;；]+", actions_str or "") if a.strip()]
    if not actions:
        raise gr.Error("请输入动作列表（建议动词+物体的完整短语，如“贴黑垫子”）")
    if not get_api_key():
        raise gr.Error("缺少 DASHSCOPE_API_KEY：请设置环境变量或写入项目根目录 .env 文件")

    video_paths = [Path(p) for p in file_input if Path(p).suffix.lower() in VIDEO_EXTS]
    if not video_paths:
        raise gr.Error(f"上传文件里没有视频（支持 {', '.join(sorted(VIDEO_EXTS))}）")

    work_dir = Path(tempfile.mkdtemp(prefix="auto_label_action_"))
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        return "\n".join(log_lines)

    no_update = gr.update()
    results = {}
    download_paths = []
    total_tokens = 0

    status = log(f"动作列表: {', '.join(actions)}")
    status = log(f"模型: {model_name}, 抽帧率: {fps} fps, 共 {len(video_paths)} 个视频")
    yield status, no_update, no_update, no_update, no_update, no_update

    for video_path in video_paths:
        status = log(f"检测 {video_path.name} ...")
        yield status, no_update, no_update, no_update, no_update, no_update
        try:
            result = None
            for kind, payload in detect_actions_video(video_path, actions, model=model_name, fps=fps):
                if kind == "log":
                    status = log(f"  {payload}")
                    yield status, no_update, no_update, no_update, no_update, no_update
                else:
                    result = payload
        except Exception as e:
            status = log(f"  失败: {e}")
            yield status, no_update, no_update, no_update, no_update, no_update
            continue

        json_path, srt_path = write_action_outputs(result, work_dir)
        download_paths += [str(json_path), str(srt_path)]
        total_tokens += result["usage"].get("total_tokens", 0)

        rows = [[s["action"], s["start"], s["end"], round(s["end"] - s["start"], 2)]
                for s in result["segments"]]
        results[video_path.name] = {
            "video": str(video_path),
            "srt": str(srt_path),
            "rows": rows,
        }
        status = log(f"  {video_path.name}: {len(result['segments'])} 个动作段, "
                     f"tokens {result['usage'].get('total_tokens', 0)}")
        yield status, no_update, no_update, no_update, no_update, no_update

    if not results:
        raise gr.Error("所有视频都检测失败，请查看运行状态里的错误信息")

    names = list(results.keys())
    first = results[names[0]]
    status = log(f"完成! 共 {len(results)} 个视频, 总消耗 tokens: {total_tokens}")
    yield (
        status,
        gr.update(choices=names, value=names[0], visible=len(names) > 1),
        gr.update(value=first["video"], subtitles=first["srt"]),
        first["rows"],
        download_paths,
        results,
    )


def show_action_result(video_name: str, results: dict):
    if not video_name or not results or video_name not in results:
        return gr.update(), gr.update()
    r = results[video_name]
    return gr.update(value=r["video"], subtitles=r["srt"]), r["rows"]


def build_app():
    with gr.Blocks(title="Auto-Label 自动标注工具") as app:
        gr.Markdown("# Auto-Label 自动标注工具")

        with gr.Tabs():
            with gr.TabItem("物品检测"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dataset_name = gr.Textbox(
                            label="数据集名称",
                            placeholder="例如: coffee_cup",
                            value="my_dataset",
                        )
                        class_names_input = gr.Textbox(
                            label="类别名称（逗号分隔）",
                            placeholder="例如: coffee cup, bottle",
                        )
                        file_input = gr.File(
                            label="上传数据文件夹（自动识别其中的视频和图片，可混合）",
                            file_count="directory",
                            file_types=[".mp4", ".jpg", ".jpeg", ".png", ".bmp"],
                        )

                        with gr.Accordion("模型设置", open=True):
                            model_select = gr.CheckboxGroup(
                                choices=["Grounding DINO", "OWLv2", "Florence-2"],
                                value=["Grounding DINO"],
                                label="选择模型",
                            )
                            conflict_strategy = gr.Radio(
                                choices=["标记冲突", "多数投票"],
                                value="标记冲突",
                                label="冲突处理策略",
                                visible=False,
                            )
                            iou_threshold = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                                label="IoU 阈值（模型间一致性判定）",
                                visible=False,
                            )

                        with gr.Accordion("高级设置", open=False):
                            conf_threshold = gr.Slider(
                                minimum=0.05, maximum=0.95, value=0.3, step=0.05,
                                label="置信度阈值",
                            )
                            stride = gr.Slider(
                                minimum=1, maximum=60, value=15, step=1,
                                label="视频帧提取步长",
                            )
                            batch_size = gr.Slider(
                                minimum=1, maximum=16, value=2, step=1,
                                label="批量大小（内存有限的机器建议保持较低，如 1-2）",
                            )
                            split_ratio = gr.Textbox(
                                label="Train/Val/Test 比例",
                                value="0.8,0.1,0.1",
                            )

                        run_btn = gr.Button("开始检测", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        status_output = gr.Textbox(
                            label="运行状态",
                            lines=12,
                            interactive=False,
                        )
                        stats_output = gr.Dataframe(
                            headers=["指标", "数值"],
                            label="检测统计",
                        )
                        gallery_output = gr.Gallery(
                            label="检测结果预览",
                            columns=3,
                            height=400,
                        )
                        review_gallery = gr.Gallery(
                            label="冲突图片（需人工审核）",
                            columns=3,
                            height=400,
                            visible=True,
                        )
                        download_output = gr.File(label="下载 YOLO 数据集")

                model_select.change(
                    fn=on_model_change,
                    inputs=[model_select],
                    outputs=[conflict_strategy, iou_threshold],
                )

                run_btn.click(
                    fn=run_pipeline,
                    inputs=[
                        file_input, class_names_input, dataset_name,
                        model_select, conflict_strategy, conf_threshold,
                        iou_threshold, stride, batch_size, split_ratio,
                    ],
                    outputs=[
                        status_output, stats_output, gallery_output,
                        review_gallery, download_output,
                    ],
                )

            with gr.TabItem("数据集复核"):
                gr.Markdown("浏览并编辑已导出数据集的 YOLO 标注，删除误检框后导出干净数据集。")
                with gr.Row():
                    with gr.Column(scale=1):
                        dataset_dropdown = gr.Dropdown(
                            choices=list_dataset_names(),
                            label="选择数据集",
                        )
                        refresh_btn = gr.Button("刷新数据集列表")
                        split_dropdown = gr.Radio(
                            choices=["train", "val", "test"],
                            value="train",
                            label="选择分割",
                        )
                        max_images_slider = gr.Slider(
                            minimum=5, maximum=200, value=30, step=5,
                            label="最多显示多少张",
                        )
                        review_btn = gr.Button("加载标注", variant="primary")
                    with gr.Column(scale=2):
                        review_summary = gr.Textbox(label="统计", interactive=False)
                        review_dataset_gallery = gr.Gallery(
                            label="标注预览",
                            columns=3,
                            height=360,
                        )
                        review_large_image = gr.Image(
                            label="当前图片（框号对应下方勾选项）",
                            interactive=False,
                            height=520,
                        )
                        review_item_status = gr.Textbox(label="当前状态", interactive=False)
                        review_delete_boxes = gr.CheckboxGroup(
                            choices=[],
                            label="勾选要删除的框",
                        )
                        with gr.Row():
                            review_prev_btn = gr.Button("上一张")
                            review_next_btn = gr.Button("下一张")
                            review_save_btn = gr.Button("保存修改", variant="primary")
                        review_export_btn = gr.Button("导出复核后数据集")
                        review_download = gr.File(label="下载干净 YOLO 数据集")

                review_state = gr.State({})
                review_dirty_state = gr.State(False)

                refresh_btn.click(
                    fn=refresh_dataset_list,
                    inputs=[],
                    outputs=[dataset_dropdown],
                )

                review_btn.click(
                    fn=load_review_dataset,
                    inputs=[review_dirty_state, dataset_dropdown, split_dropdown, max_images_slider],
                    outputs=[
                        review_dataset_gallery,
                        review_summary,
                        review_state,
                        review_large_image,
                        review_delete_boxes,
                        review_item_status,
                        review_dirty_state,
                        review_download,
                    ],
                )
                review_dataset_gallery.select(
                    fn=select_review_image,
                    inputs=[review_state, review_dirty_state],
                    outputs=[
                        review_state,
                        review_large_image,
                        review_delete_boxes,
                        review_item_status,
                        review_dirty_state,
                    ],
                )
                review_delete_boxes.change(
                    fn=update_review_deletions,
                    inputs=[review_delete_boxes, review_state],
                    outputs=[review_large_image, review_item_status, review_dirty_state],
                )
                review_prev_btn.click(
                    fn=lambda state, dirty: navigate_review_image(-1, state, dirty),
                    inputs=[review_state, review_dirty_state],
                    outputs=[
                        review_state,
                        review_large_image,
                        review_delete_boxes,
                        review_item_status,
                        review_dirty_state,
                    ],
                )
                review_next_btn.click(
                    fn=lambda state, dirty: navigate_review_image(1, state, dirty),
                    inputs=[review_state, review_dirty_state],
                    outputs=[
                        review_state,
                        review_large_image,
                        review_delete_boxes,
                        review_item_status,
                        review_dirty_state,
                    ],
                )
                review_save_btn.click(
                    fn=save_review_edits,
                    inputs=[review_state, review_delete_boxes],
                    outputs=[
                        review_dataset_gallery,
                        review_large_image,
                        review_delete_boxes,
                        review_item_status,
                        review_dirty_state,
                        review_summary,
                    ],
                )
                review_export_btn.click(
                    fn=export_reviewed_dataset,
                    inputs=[review_state],
                    outputs=[review_download, review_item_status],
                )

            with gr.TabItem("动作检测"):
                gr.Markdown(
                    "上传视频 + 自定义动作短语列表 → 输出每个动作的起止时间段。"
                    "使用云端 Qwen-VL（需要 DASHSCOPE_API_KEY，放在项目根目录 .env 或环境变量）。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        action_file_input = gr.File(
                            label="上传视频（可多个）",
                            file_count="multiple",
                            file_types=sorted(VIDEO_EXTS),
                        )
                        action_names_input = gr.Textbox(
                            label="动作列表（每行一个，建议“动词+物体”完整短语）",
                            lines=4,
                            placeholder="拿起产品\n贴黑垫子\n扫码\n放下产品",
                        )
                        action_model_select = gr.Dropdown(
                            choices=ACTION_MODELS,
                            value=DEFAULT_ACTION_MODEL,
                            label="模型（plus 质量优先 / flash 便宜快速）",
                        )
                        with gr.Accordion("高级设置", open=False):
                            action_fps = gr.Slider(
                                minimum=0.5, maximum=4, value=2, step=0.5,
                                label="抽帧率 fps（边界精度约 ±1/fps 秒，越高越贵）",
                            )
                        action_run_btn = gr.Button("开始动作检测", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        action_status = gr.Textbox(label="运行状态", lines=8, interactive=False)
                        action_video_select = gr.Dropdown(
                            label="选择要复核的视频", choices=[], visible=False,
                        )
                        action_player = gr.Video(
                            label="复核播放（字幕即检测出的动作段）", interactive=False,
                        )
                        action_table = gr.Dataframe(
                            headers=["动作", "开始(秒)", "结束(秒)", "时长(秒)"],
                            label="动作段落",
                        )
                        action_download = gr.Files(label="下载结果（JSON + SRT 字幕）")

                action_results_state = gr.State({})

                action_run_btn.click(
                    fn=run_action_detection,
                    inputs=[action_file_input, action_names_input, action_model_select, action_fps],
                    outputs=[
                        action_status, action_video_select, action_player,
                        action_table, action_download, action_results_state,
                    ],
                )
                action_video_select.change(
                    fn=show_action_result,
                    inputs=[action_video_select, action_results_state],
                    outputs=[action_player, action_table],
                )

    return app


def main():
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import click

from auto_label.config import (
    get_dataset_config,
    get_dataset_paths,
    load_datasets_yaml,
    resolve_classes,
    update_datasets_yaml,
)
from auto_label.export import write_data_yaml, write_labels
from auto_label.extract import extract_all
from auto_label.split import apply_split, compute_split


@click.group()
def cli():
    """Auto-label: automated YOLO annotation pipeline using Grounding DINO."""


@cli.command()
@click.argument("dataset_name")
@click.option("--classes", default=None, help="Comma-separated class names")
@click.option("--conf-threshold", default=0.3, type=float, help="Detection confidence threshold")
@click.option("--stride", default=15, type=int, help="Extract every Nth frame")
@click.option("--split-ratio", default="0.8,0.1,0.1", help="Train/val/test ratio")
@click.option("--model", default="IDEA-Research/grounding-dino-tiny", help="HuggingFace model ID")
@click.option("--batch-size", default=4, type=int, help="Images per inference batch")
@click.option("--device", default=None, help="cuda/mps/cpu, auto-detected if omitted")
@click.option("--datasets-dir", default="./datasets", type=click.Path(exists=True), help="Datasets root")
@click.option("--skip-extract", is_flag=True, help="Skip frame extraction")
@click.option("--force", is_flag=True, help="Re-extract/re-detect even if outputs exist")
def run(dataset_name, classes, conf_threshold, stride, split_ratio, model, batch_size, device, datasets_dir, skip_extract, force):
    """Run the full auto-labeling pipeline on a dataset."""
    datasets_dir = Path(datasets_dir)
    config = get_dataset_config(datasets_dir, dataset_name)
    class_names = resolve_classes(config, classes)
    paths = get_dataset_paths(datasets_dir, dataset_name)
    ratios = tuple(float(x) for x in split_ratio.split(","))

    click.echo(f"Dataset: {dataset_name}")
    click.echo(f"Classes: {class_names}")

    # Stage 1: Extract
    if skip_extract:
        click.echo("Skipping extraction (--skip-extract)")
        frame_paths = sorted(paths["staging"].glob("*.jpg"))
        if not frame_paths:
            for split in ("train", "val", "test"):
                frame_paths.extend(sorted((paths["detection"] / split / "images").glob("*.jpg")))
        if not frame_paths:
            raise click.UsageError("No frames found. Run without --skip-extract first.")
        click.echo(f"Found {len(frame_paths)} existing frames")
    else:
        click.echo(f"Extracting frames (stride={stride})...")
        frame_paths = extract_all(paths["raw"], paths["staging"], stride, force)
        if not frame_paths:
            raise click.UsageError(f"No videos found in {paths['raw']}")
        click.echo(f"Extracted {len(frame_paths)} frames")

    # Stage 2: Detect
    click.echo("Loading detection model...")
    from auto_label.detect import detect_all, load_model, save_detections

    model_obj, processor, resolved_device = load_model(model, device)
    click.echo(f"Running detection (conf>={conf_threshold}, batch_size={batch_size})...")
    results = detect_all(model_obj, processor, frame_paths, class_names, resolved_device, conf_threshold, batch_size)
    save_detections(results, paths["detections_json"])

    total_dets = sum(len(r["detections"]) for r in results)
    frames_with_dets = sum(1 for r in results if r["detections"])
    frames_empty = len(results) - frames_with_dets
    click.echo(f"Found {total_dets} detections across {frames_with_dets} frames ({frames_empty} frames empty)")

    if total_dets == 0:
        raise click.UsageError("No detections found. Check class names and confidence threshold.")

    # Stage 3: Split
    click.echo("Splitting dataset...")
    split_map = compute_split(frame_paths, ratios)
    apply_split(split_map, paths["detection"])
    for s, p in split_map.items():
        click.echo(f"  {s}: {len(p)} images")

    # Stage 4: Export
    click.echo("Writing YOLO labels...")
    stats = write_labels(results, split_map, paths["detection"])
    click.echo(f"Wrote labels - train: {stats['train']}, val: {stats['val']}, test: {stats['test']} ({stats['empty']} empty)")

    data_yaml_path = write_data_yaml(paths["detection"], class_names)
    update_datasets_yaml(datasets_dir / "datasets.yaml", dataset_name, class_names)

    click.echo(f"\nDone! Dataset ready at {paths['detection']}")
    click.echo(f"  data.yaml: {data_yaml_path}")
    click.echo(f"  Classes: {', '.join(f'{name} ({i})' for i, name in enumerate(class_names))}")


@cli.command()
@click.argument("dataset_name")
@click.option("--stride", default=15, type=int, help="Extract every Nth frame")
@click.option("--datasets-dir", default="./datasets", type=click.Path(exists=True))
@click.option("--force", is_flag=True)
def extract(dataset_name, stride, datasets_dir, force):
    """Extract frames from videos."""
    datasets_dir = Path(datasets_dir)
    paths = get_dataset_paths(datasets_dir, dataset_name)
    frames = extract_all(paths["raw"], paths["staging"], stride, force)
    click.echo(f"Extracted {len(frames)} frames to {paths['staging']}")


@cli.command()
@click.argument("dataset_name")
@click.option("--classes", default=None, help="Comma-separated class names")
@click.option("--conf-threshold", default=0.3, type=float)
@click.option("--model", default="IDEA-Research/grounding-dino-tiny")
@click.option("--batch-size", default=4, type=int)
@click.option("--device", default=None)
@click.option("--datasets-dir", default="./datasets", type=click.Path(exists=True))
def detect(dataset_name, classes, conf_threshold, model, batch_size, device, datasets_dir):
    """Run detection on extracted frames."""
    datasets_dir = Path(datasets_dir)
    config = get_dataset_config(datasets_dir, dataset_name)
    class_names = resolve_classes(config, classes)
    paths = get_dataset_paths(datasets_dir, dataset_name)

    frame_paths = sorted(paths["staging"].glob("*.jpg"))
    if not frame_paths:
        for split in ("train", "val", "test"):
            frame_paths.extend(sorted((paths["detection"] / split / "images").glob("*.jpg")))
    if not frame_paths:
        raise click.UsageError("No frames found. Run 'extract' first.")

    from auto_label.detect import detect_all, load_model, save_detections

    model_obj, processor, resolved_device = load_model(model, device)
    results = detect_all(model_obj, processor, frame_paths, class_names, resolved_device, conf_threshold, batch_size)
    save_detections(results, paths["detections_json"])

    total_dets = sum(len(r["detections"]) for r in results)
    click.echo(f"Found {total_dets} detections, saved to {paths['detections_json']}")


@cli.command(name="export")
@click.argument("dataset_name")
@click.option("--classes", default=None, help="Comma-separated class names")
@click.option("--split-ratio", default="0.8,0.1,0.1")
@click.option("--datasets-dir", default="./datasets", type=click.Path(exists=True))
def export_cmd(dataset_name, classes, split_ratio, datasets_dir):
    """Split dataset and export YOLO labels from detection results."""
    datasets_dir = Path(datasets_dir)
    config = get_dataset_config(datasets_dir, dataset_name)
    class_names = resolve_classes(config, classes)
    paths = get_dataset_paths(datasets_dir, dataset_name)
    ratios = tuple(float(x) for x in split_ratio.split(","))

    from auto_label.detect import load_detections

    if not paths["detections_json"].exists():
        raise click.UsageError("No detections found. Run 'detect' first.")

    results = load_detections(paths["detections_json"])
    frame_paths = [Path(r["image_path"]) for r in results]

    split_map = compute_split(frame_paths, ratios)
    apply_split(split_map, paths["detection"])

    stats = write_labels(results, split_map, paths["detection"])
    write_data_yaml(paths["detection"], class_names)
    update_datasets_yaml(datasets_dir / "datasets.yaml", dataset_name, class_names)

    click.echo(f"Exported - train: {stats['train']}, val: {stats['val']}, test: {stats['test']}")


@cli.command()
@click.argument("dataset_name")
@click.option("--classes", default=None, help="Comma-separated class names")
@click.option("--max-images", default=None, type=int, help="Max images to visualize (default: all)")
@click.option("--datasets-dir", default="./datasets", type=click.Path(exists=True))
def visualize(dataset_name, classes, max_images, datasets_dir):
    """Draw bounding boxes on images for visual inspection."""
    datasets_dir = Path(datasets_dir)
    config = get_dataset_config(datasets_dir, dataset_name)
    class_names = resolve_classes(config, classes)
    paths = get_dataset_paths(datasets_dir, dataset_name)

    if not paths["detections_json"].exists():
        raise click.UsageError("No detections found. Run 'detect' or 'run' first.")

    from auto_label.visualize import visualize_dataset

    vis_dir = paths["detection"] / "_visualize"
    written = visualize_dataset(paths["detections_json"], vis_dir, class_names, max_images)
    click.echo(f"Saved {len(written)} visualized images to {vis_dir}")


@cli.command(name="list")
@click.option("--datasets-dir", default="./datasets", type=click.Path(exists=True))
def list_cmd(datasets_dir):
    """List available datasets."""
    datasets_dir = Path(datasets_dir)
    data = load_datasets_yaml(datasets_dir)
    for name, info in data["datasets"].items():
        classes = info.get("class_names") or []
        desc = info.get("description", "")
        classes_str = ", ".join(classes) if classes else "(no classes defined)"
        click.echo(f"  {name}: {desc} [{classes_str}]")


@cli.command(name="detect-actions")
@click.argument("dataset_name")
@click.option("--actions", "actions_str", required=True,
              help="逗号分隔的动作短语, 如 '拿起产品,贴黑垫子,扫码,放下产品'")
@click.option("--model", default=None, help="qwen3-vl-plus (默认) / qwen3-vl-flash")
@click.option("--fps", default=2.0, show_default=True, type=float, help="抽帧率, 边界精度约 ±1/fps 秒")
@click.option("--videos", default=None, help="只处理这些视频文件名(逗号分隔), 默认 raw/ 下全部")
@click.option("--datasets-dir", default="./datasets", type=click.Path(exists=True))
def detect_actions_cmd(dataset_name, actions_str, model, fps, videos, datasets_dir):
    """Zero-shot 动作定位: raw/ 视频 + 动作列表 -> 每段动作的起止时间 (Qwen-VL API)."""
    from auto_label.action import (
        DEFAULT_ACTION_MODEL,
        VIDEO_EXTS,
        detect_actions_video,
        write_action_outputs,
    )

    model = model or DEFAULT_ACTION_MODEL
    datasets_dir = Path(datasets_dir)
    paths = get_dataset_paths(datasets_dir, dataset_name)
    action_dir = paths["root"] / "annotations" / "action"

    video_paths = sorted(p for p in paths["raw"].iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if videos:
        wanted = {v.strip() for v in videos.split(",") if v.strip()}
        video_paths = [p for p in video_paths if p.name in wanted]
        missing = wanted - {p.name for p in video_paths}
        if missing:
            raise click.UsageError(f"raw/ 里找不到: {sorted(missing)}")
    if not video_paths:
        raise click.UsageError(f"No videos found in {paths['raw']}")

    actions = [a.strip() for a in actions_str.split(",") if a.strip()]
    click.echo(f"动作列表: {actions}")
    click.echo(f"模型: {model}, fps={fps}, 共 {len(video_paths)} 个视频\n")

    summary, total_tokens = [], 0
    for video_path in video_paths:
        click.echo(f"{video_path.name}:")
        result = None
        for kind, payload in detect_actions_video(video_path, actions, model=model, fps=fps):
            if kind == "log":
                click.echo(f"  {payload}")
            else:
                result = payload
        write_action_outputs(result, action_dir)
        for seg in result["segments"]:
            click.echo(f"  {seg['start']:7.1f}s - {seg['end']:7.1f}s  {seg['action']}")
        if not result["segments"]:
            click.echo("  (未检出任何动作段)")
        total_tokens += result["usage"].get("total_tokens", 0)
        summary.append({"video": result["video"], "duration": result["duration"],
                        "segments": len(result["segments"]),
                        "tokens": result["usage"].get("total_tokens", 0)})

    (action_dir / "_summary.json").write_text(
        json.dumps({"model": model, "fps": fps, "actions": actions, "videos": summary},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    click.echo(f"\nDone! 输出目录: {action_dir} (每视频 .json + .srt 字幕)")
    click.echo(f"总消耗 tokens: {total_tokens}")


@cli.command(name="download-models")
@click.option("--models", default="all", help="Comma-separated model keys, or 'all'")
def download_models_cmd(models):
    """Pre-download detector model weights from HuggingFace (run once after install)."""
    from auto_label.detect import MODEL_REGISTRY, load_detector

    keys = list(MODEL_REGISTRY.keys()) if models == "all" else [m.strip() for m in models.split(",")]

    downloaded = 0
    for key in keys:
        if key not in MODEL_REGISTRY:
            click.echo(f"Unknown model: {key}. Available: {list(MODEL_REGISTRY.keys())}")
            continue
        display_name = MODEL_REGISTRY[key]["display_name"]
        model_id = MODEL_REGISTRY[key]["model_id"]
        click.echo(f"Downloading {display_name} ({model_id})...")
        load_detector(key)
        click.echo(f"  Done: {display_name}")
        downloaded += 1

    if downloaded:
        click.echo(f"{downloaded} model(s) cached locally.")
    else:
        raise click.UsageError("No valid models were downloaded.")

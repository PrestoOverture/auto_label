from pathlib import Path

import click
import yaml


def load_datasets_yaml(datasets_dir: Path) -> dict:
    yaml_path = datasets_dir / "datasets.yaml"
    if not yaml_path.exists():
        raise click.UsageError(f"datasets.yaml not found at {yaml_path}")
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "datasets" not in data:
        raise click.UsageError(f"datasets.yaml is malformed: missing 'datasets' key")
    return data


def get_dataset_config(datasets_dir: Path, dataset_name: str) -> dict:
    data = load_datasets_yaml(datasets_dir)
    datasets = data["datasets"]
    if dataset_name in datasets:
        return datasets[dataset_name]
    dataset_dir = datasets_dir / dataset_name
    if dataset_dir.is_dir():
        return {"class_names": [], "description": dataset_name}
    available = list(datasets.keys())
    raise click.UsageError(
        f"Dataset '{dataset_name}' not found.\nAvailable: {available}"
    )


def resolve_classes(dataset_config: dict, cli_classes: str | None) -> list[str]:
    if cli_classes:
        return [c.strip() for c in cli_classes.split(",") if c.strip()]
    names = dataset_config.get("class_names") or []
    if not names:
        raise click.UsageError(
            "No class names specified. Use --classes or set class_names in datasets.yaml"
        )
    return names


def get_dataset_paths(datasets_dir: Path, dataset_name: str) -> dict:
    root = datasets_dir / dataset_name
    detection = root / "annotations" / "detection"
    paths = {
        "root": root,
        "raw": root / "raw",
        "detection": detection,
        "staging": detection / "_frames",
        "detections_json": detection / "_detections.json",
        "data_yaml": detection / "data.yaml",
    }
    for key in ("staging",):
        paths[key].mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (detection / split / "images").mkdir(parents=True, exist_ok=True)
        (detection / split / "labels").mkdir(parents=True, exist_ok=True)
    return paths


def update_datasets_yaml(datasets_yaml_path: Path, dataset_name: str, class_names: list[str]) -> None:
    with open(datasets_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {"datasets": {}}
    if dataset_name not in data["datasets"]:
        data["datasets"][dataset_name] = {}
    data["datasets"][dataset_name]["class_names"] = class_names
    with open(datasets_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

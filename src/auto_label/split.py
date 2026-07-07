import random
import re
import shutil
from pathlib import Path

from tqdm import tqdm


def _video_stem(frame_path: Path) -> str:
    match = re.match(r"^(.+?)_f\d{6}_", frame_path.name)
    if match:
        return match.group(1)
    return frame_path.stem


def compute_split(
    image_paths: list[Path],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in image_paths:
        stem = _video_stem(p)
        groups.setdefault(stem, []).append(p)

    video_stems = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(video_stems)

    n = len(video_stems)
    n_train = max(1, round(n * ratios[0]))
    n_val = max(0, round(n * ratios[1]))

    if n < 3:
        tqdm.write(f"Warning: only {n} video(s), putting all in train. Consider adding more videos.")
        n_train, n_val = n, 0

    n_test = n - n_train - n_val
    if n_test < 0:
        n_val = n - n_train
        n_test = 0

    train_stems = video_stems[:n_train]
    val_stems = video_stems[n_train:n_train + n_val]
    test_stems = video_stems[n_train + n_val:]

    result = {"train": [], "val": [], "test": []}
    for s in train_stems:
        result["train"].extend(sorted(groups[s]))
    for s in val_stems:
        result["val"].extend(sorted(groups[s]))
    for s in test_stems:
        result["test"].extend(sorted(groups[s]))

    return result


def apply_split(
    split_map: dict[str, list[Path]],
    detection_dir: Path,
) -> None:
    for split_name, paths in split_map.items():
        img_dir = detection_dir / split_name / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for src in tqdm(paths, desc=f"Copying to {split_name}", leave=False):
            dst = img_dir / src.name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)

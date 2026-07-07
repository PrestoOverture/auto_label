from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(
    video_path: Path,
    output_dir: Path,
    stride: int = 15,
    force: bool = False,
) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        tqdm.write(f"Warning: cannot open {video_path.name}, skipping")
        return []

    stem = video_path.stem
    frame_idx = 0
    written = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            name = f"{stem}_f{frame_idx:06d}_stride.jpg"
            out_path = output_dir / name
            if not out_path.exists() or force:
                cv2.imwrite(str(out_path), frame)
            written.append(out_path)
        frame_idx += 1

    cap.release()
    return written


def extract_all(
    raw_dir: Path,
    output_dir: Path,
    stride: int = 15,
    force: bool = False,
) -> list[Path]:
    videos = sorted(raw_dir.glob("*.mp4"))
    if not videos:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    all_frames = []

    for video in tqdm(videos, desc="Extracting videos"):
        frames = extract_frames(video, output_dir, stride, force)
        all_frames.extend(frames)

    return all_frames


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def collect_images(input_paths: list[Path], output_dir: Path, stride: int = 15) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for p in input_paths:
        suffix = p.suffix.lower()
        if suffix == ".mp4":
            all_frames.extend(extract_frames(p, output_dir, stride))
        elif suffix in IMAGE_SUFFIXES:
            all_frames.append(p)

    return sorted(all_frames)

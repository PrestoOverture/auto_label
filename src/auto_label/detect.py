import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


MODEL_REGISTRY = {
    "grounding_dino": {
        "model_id": "IDEA-Research/grounding-dino-tiny",
        "display_name": "Grounding DINO",
    },
    "owlv2": {
        "model_id": "google/owlv2-base-patch16-ensemble",
        "display_name": "OWLv2",
    },
    "florence2": {
        "model_id": "microsoft/Florence-2-large",
        "display_name": "Florence-2",
    },
}


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_oom_error(e: Exception) -> bool:
    if isinstance(e, torch.OutOfMemoryError):
        return True
    msg = str(e).lower()
    return "out of memory" in msg or "insufficient memory" in msg


def _release_device_memory(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def is_cached(model_id: str) -> bool:
    from huggingface_hub import scan_cache_dir

    try:
        cache_info = scan_cache_dir()
        return any(repo.repo_id == model_id for repo in cache_info.repos)
    except Exception:
        return False


def match_label(predicted_label: str, class_names: list[str]) -> int | None:
    predicted = predicted_label.strip().lower()
    for idx, class_name in enumerate(class_names):
        cn = class_name.strip().lower()
        if predicted == cn or predicted in cn or cn in predicted:
            return idx
    return None


# ---------------------------------------------------------------------------
# Grounding DINO
# ---------------------------------------------------------------------------

def load_grounding_dino(device: str) -> tuple:
    model_id = MODEL_REGISTRY["grounding_dino"]["model_id"]
    tqdm.write(f"Loading {MODEL_REGISTRY['grounding_dino']['display_name']} on {device}...")
    local_only = is_cached(model_id)
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_only)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=local_only).to(device)
    model.eval()
    return model, processor


def detect_grounding_dino_batch(
    model, processor, image_paths: list[Path], class_names: list[str],
    device: str, conf_threshold: float = 0.3,
) -> list[dict]:
    text_prompt = " . ".join(class_names) + " ."
    images = [Image.open(p).convert("RGB") for p in image_paths]

    inputs = processor(
        images=images,
        text=[text_prompt] * len(images),
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[img.size[1], img.size[0]] for img in images], device=device)
    all_results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=target_sizes,
        input_ids=inputs["input_ids"],
        threshold=conf_threshold,
        text_threshold=conf_threshold,
    )

    results_list = []
    for i, img_path in enumerate(image_paths):
        w, h = images[i].size
        batch_results = all_results[i]

        detections = []
        boxes = batch_results["boxes"].cpu().tolist()
        scores = batch_results["scores"].cpu().tolist()
        labels = batch_results.get("text_labels", batch_results.get("labels", []))

        for box, score, label in zip(boxes, scores, labels):
            class_id = match_label(label, class_names)
            if class_id is None:
                continue
            detections.append({
                "class_name": class_names[class_id],
                "class_id": class_id,
                "bbox_xyxy": [round(v, 2) for v in box],
                "confidence": round(score, 4),
            })

        results_list.append({
            "image_path": str(img_path),
            "width": w,
            "height": h,
            "detections": detections,
        })

    return results_list


# ---------------------------------------------------------------------------
# OWLv2
# ---------------------------------------------------------------------------

def load_owlv2(device: str) -> tuple:
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    model_id = MODEL_REGISTRY["owlv2"]["model_id"]
    tqdm.write(f"Loading {MODEL_REGISTRY['owlv2']['display_name']} on {device}...")
    local_only = is_cached(model_id)
    processor = Owlv2Processor.from_pretrained(model_id, local_files_only=local_only)
    model = Owlv2ForObjectDetection.from_pretrained(model_id, local_files_only=local_only).to(device)
    model.eval()
    return model, processor


def detect_owlv2_batch(
    model, processor, image_paths: list[Path], class_names: list[str],
    device: str, conf_threshold: float = 0.3,
) -> list[dict]:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    text_queries = [class_names] * len(images)

    inputs = processor(text=text_queries, images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[img.size[1], img.size[0]] for img in images], device=device)
    results = processor.post_process_object_detection(
        outputs, threshold=conf_threshold, target_sizes=target_sizes,
    )

    results_list = []
    for i, img_path in enumerate(image_paths):
        w, h = images[i].size
        batch_results = results[i]

        detections = []
        boxes = batch_results["boxes"].cpu().tolist()
        scores = batch_results["scores"].cpu().tolist()
        label_indices = batch_results["labels"].cpu().tolist()

        for box, score, label_idx in zip(boxes, scores, label_indices):
            label_idx = int(label_idx)
            if label_idx < 0 or label_idx >= len(class_names):
                continue
            detections.append({
                "class_name": class_names[label_idx],
                "class_id": label_idx,
                "bbox_xyxy": [round(v, 2) for v in box],
                "confidence": round(score, 4),
            })

        results_list.append({
            "image_path": str(img_path),
            "width": w,
            "height": h,
            "detections": detections,
        })

    return results_list


# ---------------------------------------------------------------------------
# Florence-2
# ---------------------------------------------------------------------------

def load_florence2(device: str) -> tuple:
    from transformers import AutoModelForCausalLM

    model_id = MODEL_REGISTRY["florence2"]["model_id"]
    tqdm.write(f"Loading {MODEL_REGISTRY['florence2']['display_name']} on {device}...")
    local_only = is_cached(model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16, local_files_only=local_only,
    ).to(device)
    model.eval()
    return model, processor


def detect_florence2_batch(
    model, processor, image_paths: list[Path], class_names: list[str],
    device: str, conf_threshold: float = 0.3,
) -> list[dict]:
    task = "<OPEN_VOCABULARY_DETECTION>"
    text_prompt = task + ", ".join(class_names)

    results_list = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"].to(model.dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=pixel_values,
                max_new_tokens=1024,
                num_beams=1,
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = processor.post_process_generation(
            generated_text, task=task, image_size=(w, h),
        )
        _release_device_memory(device)

        detections = []
        if task in result:
            bboxes = result[task].get("bboxes", [])
            bbox_labels = result[task].get("bboxes_labels", [])
            for box, label in zip(bboxes, bbox_labels):
                class_id = match_label(label, class_names)
                if class_id is None:
                    continue
                detections.append({
                    "class_name": class_names[class_id],
                    "class_id": class_id,
                    "bbox_xyxy": [round(v, 2) for v in box],
                    "confidence": 1.0,
                })

        results_list.append({
            "image_path": str(img_path),
            "width": w,
            "height": h,
            "detections": detections,
        })

    return results_list


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_LOADERS = {
    "grounding_dino": load_grounding_dino,
    "owlv2": load_owlv2,
    "florence2": load_florence2,
}

_DETECTORS = {
    "grounding_dino": detect_grounding_dino_batch,
    "owlv2": detect_owlv2_batch,
    "florence2": detect_florence2_batch,
}


def load_detector(model_key: str, device: str | None = None) -> tuple:
    if model_key not in _LOADERS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(_LOADERS.keys())}")
    resolved = device or auto_device()
    model, processor = _LOADERS[model_key](resolved)
    return model, processor, resolved


def detect_all_generic(
    model_key: str,
    model,
    processor,
    image_paths: list[Path],
    class_names: list[str],
    device: str,
    conf_threshold: float = 0.3,
    batch_size: int = 4,
) -> list[dict]:
    detect_fn = _DETECTORS[model_key]
    all_results = []
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    model_name = MODEL_REGISTRY[model_key]["display_name"]
    for batch in tqdm(batches, desc=f"Detecting ({model_name})"):
        try:
            results = detect_fn(model, processor, batch, class_names, device, conf_threshold)
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            tqdm.write("OOM detected, falling back to batch_size=1")
            _release_device_memory(device)
            results = []
            for single in batch:
                r = detect_fn(model, processor, [single], class_names, device, conf_threshold)
                results.extend(r)
        all_results.extend(results)
        _release_device_memory(device)

    return all_results


# ---------------------------------------------------------------------------
# Backward-compatible API (used by CLI)
# ---------------------------------------------------------------------------

def load_model(
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    device: str | None = None,
) -> tuple:
    resolved = device or auto_device()
    tqdm.write(f"Loading model {model_id} on {resolved}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(resolved)
    model.eval()
    return model, processor, resolved


def detect_batch(
    model, processor, image_paths: list[Path], class_names: list[str],
    device: str, conf_threshold: float = 0.3,
) -> list[dict]:
    return detect_grounding_dino_batch(model, processor, image_paths, class_names, device, conf_threshold)


def detect_all(
    model, processor, image_paths: list[Path], class_names: list[str],
    device: str, conf_threshold: float = 0.3, batch_size: int = 4,
) -> list[dict]:
    all_results = []
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    for batch in tqdm(batches, desc="Detecting"):
        try:
            results = detect_grounding_dino_batch(model, processor, batch, class_names, device, conf_threshold)
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            tqdm.write("OOM detected, falling back to batch_size=1")
            _release_device_memory(device)
            results = []
            for single in batch:
                r = detect_grounding_dino_batch(model, processor, [single], class_names, device, conf_threshold)
                results.extend(r)
        all_results.extend(results)
        _release_device_memory(device)

    return all_results


def save_detections(results: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_detections(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

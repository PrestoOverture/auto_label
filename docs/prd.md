# Auto-Label: Product Requirements Document

## Overview

Auto-Label is a **semi-automated annotation and video analysis tool** for industrial inspection, usable via CLI or a Gradio Web UI. It has two main capabilities:

1. **物品检测（半自动标注）**: Uses open-vocabulary object detection (Grounding DINO, optionally OWLv2 / Florence-2) to pre-annotate video frames with YOLO-format bounding boxes. The output is a draft that requires human review — the user keeps correct labels and discards errors, significantly faster than labeling from scratch.

2. **动作检测（推理工具）**: Given a video and a user-defined list of action phrases (e.g. "拿起产品", "贴黑垫子"), it outputs start/end timestamps for each action occurrence via Qwen-VL (cloud API), with SRT subtitles for instant review. This is a direct inference tool — the Qwen-VL results are the final output, not training data for another model.

## Problem

Industrial inspection projects require annotated datasets and video-level action analysis. Manual labeling is slow and expensive; manually watching videos to log action timestamps doesn't scale. The team needs a reusable tool that non-CLI users can operate directly.

## Solution

**物品检测**: Grounding DINO pre-annotates frames in one click — the human reviewer only needs to delete bad boxes instead of drawing from zero. This is a **semi-automated** workflow: model output is a draft, not ground truth.

**动作检测**: Qwen-VL directly answers "what happened and when" for each video at negligible cost (~0.0004 元/视频秒). No model training required — the API output (JSON + SRT) is the deliverable.

## Key Features

- Open-vocabulary detection: works with any class name, not limited to a fixed set
- Action detection: video + custom action phrase list → timestamped segments (JSON + SRT), zero-shot via Qwen-VL, long videos auto-chunked — results are the final deliverable, not intermediate training data
- Human review workflow: browse pre-annotated frames, open a single image with numbered boxes, mark bad boxes for deletion, save the edited YOLO label file in place, and export a clean reviewed YOLO dataset
- YOLO-native output: labels drop directly into standard YOLO directory structure
- Visualization: draw annotated bounding boxes on images for quality inspection, both for a fresh run and for browsing any previously-exported dataset
- Web UI: folder upload, model selection, live progress, results gallery — no CLI needed for day-to-day use
- Incremental: new datasets plug in without modifying existing code

## Project Structure

```
auto_label/
├── datasets/                        # Data root (continuously growing)
│   ├── datasets.yaml                # Central dataset registry
│   └── {dataset_name}/
│       ├── raw/                     # Source MP4 videos
│       ├── annotations/action/      # 动作检测结果 ({video}.json + {video}.srt + _summary.json)
│       ├── annotations/detection/
│       │   ├── data.yaml            # YOLO dataset config (generated)
│       │   ├── _frames/             # Extracted frames (staging)
│       │   ├── _detections.json     # Raw detection results (intermediate)
│       │   ├── _visualize/          # Bbox visualization images
│       │   ├── train/{images,labels}
│       │   ├── val/{images,labels}
│       │   └── test/{images,labels}
│       ├── features/
│       ├── models/
│       └── runs/
├── docs/
│   ├── prd.md                       # This file
│   ├── roadmap.md                   # What's done and what's next
│   └── tech_stack.md                # Technology stack
├── poc/                             # 动作检测 POC (X-CLIP vs Qwen 对比实验, 已完结, 结论见 README)
├── src/auto_label/
│   ├── __init__.py                  # Package version
│   ├── __main__.py                  # CLI entry point
│   ├── cli.py                       # Click command definitions
│   ├── config.py                    # Dataset config management
│   ├── extract.py                   # Video/image frame collection
│   ├── detect.py                    # Multi-model inference (Grounding DINO / OWLv2 / Florence-2)
│   ├── action.py                    # 动作检测 (Qwen-VL API, 时序定位 + 长视频切块)
│   ├── consensus.py                 # Multi-model IoU matching + conflict/majority-vote strategies
│   ├── split.py                     # Train/val/test splitting
│   ├── export.py                    # YOLO label + data.yaml generation
│   ├── visualize.py                 # Bbox visualization (CJK-capable, PIL-based text)
│   └── app.py                       # Gradio Web UI (物品检测 / 数据集复核 / 动作检测)
├── README.md                        # 使用说明（安装 / Web UI / CLI / 输出格式 / FAQ）
├── pyproject.toml
└── requirements.txt
```

## Usage

```bash
# One-time setup
pip install -r requirements.txt
pip install -e .
python -m auto_label download-models  # pre-download detector weights, avoids first-use delay in the UI

# Full pipeline
python -m auto_label run <dataset_name> --classes "class1, class2"

# Individual stages
python -m auto_label extract <dataset_name>
python -m auto_label detect <dataset_name>
python -m auto_label export <dataset_name>
python -m auto_label visualize <dataset_name>

# List datasets
python -m auto_label list

# Action detection (需要 DASHSCOPE_API_KEY, 放环境变量或项目根 .env)
python -m auto_label detect-actions <dataset_name> --actions "拿起产品,贴黑垫子,扫码,放下产品"

# Web UI (multi-model detection: Grounding DINO / OWLv2 / Florence-2)
python -m auto_label.app
```

## 数据集复核 Workflow

The `数据集复核` tab closes the object-labeling loop after pre-annotation:

1. Select a registered dataset and split (`train` / `val` / `test`), then click `加载标注`.
2. Click an image in the gallery to open a large single-image view. Boxes are drawn with numbers (`1`, `2`, `3`, ...).
3. Check the boxes that should be deleted, then click `保存修改`.
4. The paired YOLO label file is rewritten in place with only the kept boxes. If `_visualize/` exists, the visualization image is refreshed. The summary is recalculated after saving.
5. Click `导出复核后数据集` to download a zip containing the reviewed split's `images/`, `labels/`, and `data.yaml`.

Unsaved deletion selections block gallery switching, previous/next navigation, and reload so the user does not accidentally discard review work.

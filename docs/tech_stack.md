# Tech Stack

## Detection Models

| Component | Choice | Notes |
|---|---|---|
| Grounding DINO (tiny) | `IDEA-Research/grounding-dino-tiny` | Open-vocabulary, zero-shot object detection via text prompts. Default/primary model. |
| OWLv2 | `google/owlv2-base-patch16-ensemble` | Second teacher model, used for multi-model consensus |
| Florence-2 (large) | `microsoft/Florence-2-large` | Third teacher model; generative (uses `model.generate()` with `<OPEN_VOCABULARY_DETECTION>` task), needs `trust_remote_code=True` |
| Model interface | HuggingFace `transformers` | Registry/dispatch pattern in `detect.py` (`MODEL_REGISTRY`, `load_detector()`, `detect_all_generic()`) — each model has its own loader + batch-detect function, normalized to a common detection dict format |
| Output format | YOLO bbox | `class_id x_center y_center width height` (normalized 0-1) |

All three models are optional and loaded lazily — a run can use just Grounding DINO (default, matches the original single-model pipeline) or any combination for cross-model consensus.

## Action Detection (动作检测)

| Component | Choice | Notes |
|---|---|---|
| Teacher model | `qwen3-vl-plus` (DashScope API) | Zero-shot temporal action localization：视频 + 中文动作短语列表 → 起止时间段。Qwen3-VL 原生支持"文本-时间戳对齐"，中文动作名无损（绕开本地检测模型英文编码器的弱点）。`qwen3-vl-flash` 为降本选项 |
| 接口 | DashScope OpenAI 兼容 `/chat/completions` | 帧列表（base64 JPEG，最长边 640、质量 80）+ `{"type": "video", "video": [...], "fps": 2}`；纯 `requests` 调用，无 SDK 依赖 |
| 抽帧 | 2 fps（可调 0.5-4） | 边界精度 ≈ ±1/fps 秒；总监需求 ±1s，2fps 够用 |
| 长视频 | 自动切块 | 单请求 ≤64 帧 → >36s 视频切成 ≤30s 段、重叠 3s，分段请求后全局合并（同名段并集、异名重叠取中点，依据"单人顺序操作"假设） |
| 输出 | `{stem}.json` + `{stem}.srt` | JSON 供机器/后续训练；SRT 供人工复核（Web UI 的 gr.Video 直接加载字幕播放，IINA/VLC 也能用） |
| API key | `DASHSCOPE_API_KEY` | 读取顺序：环境变量 → 项目根 `.env`。网络错误自动重试 3 次（本机代理不稳） |
| 实测成本 | 22s@2fps ≈ 7K tokens ≈ **0.008 元**；87s ≈ 31K tokens ≈ **0.035 元** | 折合 ≈0.0004 元/视频秒（2fps、640px、qwen3-vl-plus 基础档 输入 1.06 元/百万 + 输出 10.6 元/百万，2026-07 第三方源牌价，以百炼控制台为准）。规模感：1 小时视频 ≈ 1.4 元；1000 个 20s 视频 ≈ 8 元；百炼批量推理半价、缓存命中输入 0.2 元/百万可再降。消耗 98% 在输入侧（视频帧），fps 和分辨率是成本的两个直接旋钮 |
| 选型对比 | 本地 X-CLIP 滑窗 POC 不合格 | 细粒度工位动作漏检/错位，见 `poc/`；qwen3-vl-flash 更便宜但当前成本已可忽略，没必要降质量档 |

X-CLIP 本地路线的 POC 代码保留在 `poc/action_poc.py`（`xclip` 子命令），仅供未来交叉验证实验，不在生产链路里。

## Multi-Model Consensus

When 2+ models are selected, `consensus.py` reconciles their outputs:

- **Agreement test**: same `class_id` + IoU ≥ threshold (default 0.5, user-adjustable)
- **`flag_conflicts`**: images where models disagree go to a review set instead of being auto-labeled
- **`majority_vote`**: only available for odd counts > 2 (e.g. 3 models) — a 2-vs-2 split has no real majority, so the option is shown but locked to `flag_conflicts` for even counts
- Review images get a side-by-side visualization with each model's boxes in a distinct color + legend

## Dataset Review UI

The Gradio `数据集复核` tab edits already-exported YOLO labels in place:

| Component | Choice | Notes |
|---|---|---|
| Source of truth | `train/`, `val/`, `test` image/label pairs | The UI reads current `.txt` label files through `visualize.read_yolo_label()` instead of relying on `_detections.json`, so saved review edits are immediately reflected. |
| Box selection | Numbered bbox overlay + `gr.CheckboxGroup` | The large image view draws each bbox with a stable number; checked numbers are treated as delete candidates and shown crossed out before save. |
| Save path | `export.write_labels()` | `保存修改` rewrites the paired YOLO `.txt` file with only kept boxes, preserving the normalized `class_id x_center y_center width height` format. |
| Visualization refresh | `visualize.draw_detections()` | If `_visualize/` exists, the reviewed image visualization is regenerated after saving. Stale/out-of-range class IDs are rendered as `class_{id}` instead of crashing. |
| Dirty-state guard | `gr.State` | Reload, gallery selection, and previous/next navigation reject unsaved delete selections to avoid losing review work. |
| Download | Temporary zip of reviewed split | `导出复核后数据集` packages only the active split's `images/`, `labels/`, and `data.yaml`; staging files such as `_frames/`, `_visualize/`, and `_detections.json` are intentionally excluded. |

## Python Dependencies

| Package | Purpose |
|---|---|
| `torch` + `torchvision` | Model runtime (CPU / CUDA / MPS) |
| `transformers==4.49.0` | Model loading and inference for all three detectors. **Pinned, not a floor** — Florence-2's `trust_remote_code=True` code on the Hub predates transformers 4.50's `PretrainedConfig` refactor and breaks (`AttributeError: forced_bos_token_id`) on newer versions. Grounding DINO / OWLv2 are natively integrated and unaffected by the pin. |
| `scipy` | Required by `Owlv2ImageProcessor`'s resize step (not obvious until you actually run inference — `from_pretrained()` alone won't surface this) |
| `opencv-python-headless` | Video frame extraction, image I/O, box drawing |
| `Pillow` | Image preprocessing for transformers; also used for text rendering (see below) |
| `PyYAML` | datasets.yaml and data.yaml read/write |
| `click` | CLI framework |
| `tqdm` | Progress bars |
| `gradio` | Web UI |
| `requests` | 动作检测调用 DashScope API |
| `einops`, `timm` | Transitive requirements for Florence-2 |

## Runtime

| Aspect | Detail |
|---|---|
| Python | >= 3.10 |
| GPU | Optional. Auto-detects CUDA > MPS > CPU |
| OS | macOS (dev), Linux (production), Windows (untested) |
| Model weights | Downloaded from HuggingFace Hub on first use, cached under `~/.cache/huggingface`. `python -m auto_label download-models` pre-fetches all three so the Web UI never blocks on a download mid-session. Loaders check the cache first (`detect.is_cached()`) and pass `local_files_only=True` when already cached, so a flaky proxy/network doesn't break a model that's already on disk. |

## Why Multiple Models

The pipeline needs to label **any object class** defined by the user in natural language, without retraining. Grounding DINO is the default because it's the best single-model tradeoff (open-vocabulary, decent accuracy, small footprint). OWLv2 and Florence-2 are added as independent "teachers" so that cross-model agreement can substitute for ground truth on datasets that don't have any — this is the core of the multi-model consensus workflow.

This is essentially a **knowledge distillation** setup: one or more large, slow, general detectors generate labels that are used to train YOLO (small, fast, specialized) for deployment.

## Known Limitations

- **Chinese-language prompts detect noticeably worse than English.** All three models' text encoders are trained predominantly on English corpora. Empirically, on the same frames, an English prompt ("washer") reached 0.40–0.50 confidence while the Chinese equivalent ("垫圈") never exceeded 0.19 even at a lowered threshold — the object is visible and detectable, but the Chinese text embedding is weak. Since this project's real class names are almost always Chinese (industrial part names), this is a live constraint, not a hypothetical. No fix is implemented yet; the likely direction is decoupling "prompt text sent to the model" from "display/label name stored in the dataset" so a user can type a Chinese class name but detect against an English prompt.
- **MPS (Apple Silicon) memory is not automatically reclaimed between batches** by PyTorch. `detect.py` calls `torch.mps.empty_cache()` after every batch to counter this — without it, long videos on lower-memory Macs (8GB unified memory) progressively slow down and eventually hit a native `Insufficient Memory` GPU driver error that isn't a catchable Python exception (so batch-level try/except retries don't help; the only real fix is not letting the pressure build up in the first place). Default batch size in the Web UI is kept conservative (2) for this reason.
- **`cv2.putText` cannot render CJK text** — it silently renders unsupported glyphs as `?`. `visualize.py` renders label text via PIL (`ImageDraw` + a TTF font resolved from common macOS/Linux/Windows font paths) instead; box drawing still uses cv2, only text went through PIL.

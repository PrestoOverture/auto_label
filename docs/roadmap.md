# Roadmap

## Done

### v0.1.0 (2026-06-25)

- [x] Project scaffolding (src layout, pyproject.toml, CLI entry point)
- [x] Dataset config management (datasets.yaml read/write, class name resolution)
- [x] Video frame extraction (stride-based, cv2, idempotent)
- [x] Grounding DINO detection (HuggingFace transformers, batch inference, auto device)
- [x] Train/val/test split (video-level grouping to prevent data leakage)
- [x] YOLO label export (normalized xywh format, data.yaml generation)
- [x] Bbox visualization (draw detections on images for quality check)
- [x] End-to-end validation on coffee_cup dataset (83 frames, 97 detections)

### v0.2.0 (2026-07-02)

- [x] Web UI (Gradio)：顶部两个 Tab，物品检测 + 动作检测
  - [x] 物品检测页面：上传视频/图片 → 选择模型 → 检测 → 展示结果和可视化 → 下载 YOLO 数据集
  - [x] 动作检测页面：占位（待总监确认需求）
  - [x] 支持图片输入（collect_images 函数，支持目录/单图/视频）
- [x] Multi-model support: Grounding DINO + OWLv2 + Florence-2（detect.py 函数分发架构）
- [x] Multi-model consensus: IoU + 类别匹配判定一致性（consensus.py）
  - [x] 两种策略：标记冲突（flag_conflicts）/ 多数投票（majority_vote）
  - [x] 冲突图片多模型 bbox 可视化对比
- [x] `download-models` command: 一次性预下载所有模型权重，避免 UI 里首次切换模型时卡住
- [x] `pip install -e .` support（补上 `[build-system]`），`python -m auto_label` 不再需要手动设 PYTHONPATH
- [x] 数据集复核 Tab：浏览任意已导出数据集（train/val/test）的 YOLO 标注可视化，人工检查历史标注是否正确（读取 .txt label 文件直接转 bbox 画框，不依赖 _detections.json）
- [x] MPS 显存泄漏修复：每个 batch 后 `torch.mps.empty_cache()`，修复 Apple Silicon 上长视频检测内存持续爬升导致的 OOM
- [x] 可视化中文标签：画框文字改用 PIL + TTF 字体，修复 cv2.putText 无法渲染中文类名（显示问号）的问题
- [x] 模型离线加载：已缓存的模型改用 `local_files_only=True`，代理/网络故障时不再卡在联网校验上（`detect.py` 的 `is_cached()`）


### v0.3.0 (2026-07-07)

动作检测（需求 2026-07-06 总监确认：视频 + 自定义动作短语列表 → 每段动作起止时间戳；zero-shot 无标注数据；单人顺序操作；边界 ±1s；允许云端 API）：

- [x] POC（`poc/action_poc.py`，两条路线对比，输出 JSON + SRT）：
  - 路线 A X-CLIP 本地滑窗：**不合格**——"贴黑垫子"漏检、"扫码"错位，细粒度工位动作 + 英文 prompt 的 zero-shot 迁移不够；弃用（最多留作未来交叉验证）
  - 路线 B `qwen3-vl-plus`（DashScope）：**通过**——140944 密集抽帧裁决 4/4 动作全对、边界 ±1s 内，且列表外的装配动作被正确留空（不编造）
- [x] `action.py` 模块：抽帧（JPEG bytes，内存友好）→ OpenAI 兼容接口（帧列表 + fps 参数）→ 容错解析（code fence / HH:MM:SS 时间戳 / 列表外动作过滤）→ 段落合并；网络错误重试 3 次（代理不稳的教训）；API key 从环境变量或项目根 .env 读取
- [x] 长视频自动切块：>36s 切成 ≤30s 段（重叠 3s），分段请求后按全局时间轴合并（同名段并集、异名重叠中点切开）；87s 视频实测 4 段、跨块合并正确，中段空窗经抽帧核实为真实无列表内动作
- [x] CLI：`python -m auto_label detect-actions <dataset> --actions "拿起产品,贴黑垫子,..."`（`--videos` 可选子集），输出 `datasets/<name>/annotations/action/{stem}.json/.srt` + `_summary.json`（含 token 消耗）
- [x] Web UI 动作检测 Tab：上传多视频 + 动作列表（每行一个）→ 实时进度 → **复核播放器（gr.Video 加载 SRT 字幕，播放视频直接看检测段落）** + 段落表格 + JSON/SRT 下载；多视频下拉切换
- [x] 实测成本参考：22s 视频 @2fps ≈ 7K tokens，87s ≈ 31K tokens（qwen3-vl-plus）
- [x] 成本核算（2026-07-07，牌价：输入 1.06 元/百万 + 输出 10.6 元/百万，基础档 ≤32K）：**≈0.0004 元/视频秒** → 20s 视频不到 1 分钱（0.008 元）、87s 视频 0.035 元、276s 视频 ≈0.11 元、贴黑垫子全数据集（11 视频）≈0.21 元、1000 个 20s 视频 ≈8 元、1 小时视频量 ≈1.4 元。批量推理半价可再省一半；消耗 98% 在输入帧，fps/分辨率是成本旋钮。结论：当前及可预见规模下成本可忽略，不构成选型约束
- [x] 数据集复核闭环：复核 Tab 支持点击图库图片后进入单图大图复核，bbox 编号显示，勾选误检框后 `保存修改` 会原地重写对应 YOLO `.txt` 标签、刷新 `_visualize/`（如果存在）、更新统计，并可导出当前复核 split 的干净 YOLO zip
- [x] 复核 Tab bugfix：重新加载会拦截未保存修改；导出 zip 不再混入 `_frames`、`_visualize` 或其他 split；坏图会跳过/显示错误状态而不是拖垮整个图库；过期 `class_id` 显示为 `class_{id}`，避免复核保存时崩溃

## Planned

> **方向说明（2026-07-07 总监对齐后）**：总监目标是"尽量把标注做到自动化"，不要求自动 YOLO 训练。
> - **物品检测**定位为半自动标注工具：Grounding DINO 预标注 → 人工 review 删除误检 → 导出干净数据集。实测 OWLv2/Florence-2 效果不佳，全自动不现实，重心放在 review 交互。
> - **动作检测**定位为推理工具：Qwen-VL 直接输出结果，成本可忽略，不需要蒸馏本地模型。

### Short-term

- [ ] 中文类别名检测效果差：实测同一物体，英文提示词（"washer" 0.40-0.50）置信度远高于中文（"垫圈" 从未超过 0.19，阈值降到 0.15 也一样）。三个模型的文本编码器都是英文语料为主，这个项目的真实类别名几乎全是中文，是个实际会踩到的限制，不是个例。可能方向：把"给模型的检测提示词"和"最终存储/显示的类名"拆成两个字段，让用户中文命名、英文检测。目前未实现。
- [ ] `validate` command: 校验导出的 YOLO 数据集格式合规（data.yaml 完整、图片/标签一一对应、class_id 合法、bbox 在 [0,1] 范围内、图片可读）
- [ ] `report` command: 质量报告（每类检测数量、空帧比例、confidence 分布、bbox 面积分布、每视频检测数量）
- [ ] Run pipeline on remaining datasets (抬起屏幕上螺丝, 贴黑垫子)
- [ ] Confidence threshold tuning per dataset
- [ ] Add `--skip-detect` flag for re-exporting with different split ratios

### 动作检测后续

- [ ] 手标 ~10 个视频当 golden set，量化 segment-level precision/recall @ tIoU（nice-to-have，回答"准确率多少"）
- [ ] 动作检测结果写回 datasets.yaml（action_names 字段），与物品检测的 class_names 对齐

### Medium-term

- [ ] Temporal consistency filtering: 利用视频连续性过滤误检（只在单帧出现的检测大概率是误检）
- [x] ~~Multi-model voting~~ (done in v0.2.0)
- [ ] Advanced frame sampling (histogram-based, pixel-diff-based) beyond fixed stride
- [ ] NMS deduplication for overlapping detections across similar frames
- [ ] Batch processing: run pipeline on all datasets at once

### Long-term

- [ ] SAM2 tracking: 关键帧检测 + 后续帧跟踪导出 bbox，比逐帧独立检测更稳定
- [ ] Model comparison (grounding-dino-tiny vs base, confidence/recall trade-off)

### Dropped

- ~~Active learning loop~~ — 总监不要求自动训练 YOLO
- ~~Integration with YOLO training pipeline~~ — 同上
- ~~蒸馏路线设计~~ — 动作检测是推理工具，Qwen-VL 直接出结果，不需要训练本地模型
- ~~自一致性校验~~ — 成本已低到不需要省 API 调用，直接用 Qwen 结果即可

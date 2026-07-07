# Auto-Label 自动标注工具

工业质检数据集的自动标注流水线。两大能力：

- **物品检测自动标注**：视频/图片 + 自定义类别名 → YOLO 格式 bbox 数据集（本地开放词汇检测模型，免训练）
- **动作检测**：视频 + 自定义动作短语列表 → 每个动作的起止时间段（云端 Qwen-VL，zero-shot），附 SRT 字幕即放即核

日常使用走 Web UI，批量/自动化走 CLI，两者功能等价。

## 功能一览

| 功能 | 输入 | 输出 | 算力 |
|---|---|---|---|
| 物品检测 | 视频/图片 + 类别名（如 `washer, screw`） | YOLO 数据集（train/val/test + data.yaml） | 本地（CPU/CUDA/MPS 自动） |
| 多模型共识 | 同上，选 2-3 个检测器 | 一致的自动标注 + 分歧图片送人工复核 | 本地 |
| 动作检测 | 视频 + 动作短语（如 `贴黑垫子`） | `{视频}.json` 时间段 + `{视频}.srt` 字幕 | 云端 API（无本地算力要求） |
| 数据集复核 | 任意已导出数据集 | 标注可视化画廊 / 字幕复核播放 | 本地 |

## 环境要求

- Python ≥ 3.10，macOS / Linux
- 物品检测：无 GPU 也能跑；8GB 内存的机器把 batch size 保持在 1-2（Web UI 默认 2）
- 动作检测：不需要本地算力，需要阿里云百炼 API key

## 安装（一次性）

```bash
pip install -r requirements.txt
pip install -e .
python -m auto_label download-models   # 预下载三个检测模型权重，避免 UI 里首次选模型时卡住等下载
```

> ⚠️ `transformers` 锁定在 `4.49.0`，**不要手动升级**（Florence-2 的 remote code 在 4.50+ 上直接崩，详见 docs/tech_stack.md）。安装时 pip 对 gradio 的 `huggingface-hub` 版本告警可以忽略，实测不影响运行。

动作检测需要配置 API key（[阿里云百炼控制台](https://bailian.console.aliyun.com/) 创建）：

```bash
echo 'DASHSCOPE_API_KEY=sk-xxx' > .env   # 项目根目录；或 export 成环境变量
```

`.env` 只留在本机，不要提交到任何代码仓库。

## 数据集组织

```
datasets/
├── datasets.yaml          # 数据集登记表（运行时自动更新，可手动填 class_names 当默认类别）
└── <数据集名>/
    └── raw/               # 源视频放这里（.mp4/.avi/.mov/.mkv）
```

新数据集只需要建目录、放视频，不用改代码。

## 方式一：Web UI（推荐日常使用）

```bash
python -m auto_label.app   # 打开 http://127.0.0.1:7860
```

### Tab 1「物品检测」

1. 上传**文件夹**（里面的视频、图片自动识别，可混合）
2. 填数据集名称、类别名称（逗号分隔；**英文提示词置信度明显更高**，如用 `washer` 而不是 `垫圈`，中文命名方案见 roadmap）
3. 选 1-3 个模型：默认 Grounding DINO 即可；选 2 个以上出现共识选项（**标记冲突**=分歧图片送复核；**多数投票**=3 个模型时可选）
4. 高级设置里可调：置信度阈值（默认 0.3）、抽帧步长（默认 15）、batch size（内存小的机器保持 1-2）
5. 点「开始检测」，状态框实时刷新进度
6. 完成后：检测结果画廊、冲突图片画廊（多模型时）、**下载 YOLO 数据集 zip**

### Tab 2「数据集复核」

选数据集 → 选 train/val/test → 「加载标注」→ 画廊里逐张检查历史标注是否正确（直接读 label 文件画框，任何存量数据集都能看）。

### Tab 3「动作检测」

1. 上传视频（可多个）
2. 动作列表**每行一个**，用"动词+物体"的完整短语（`贴黑垫子` 好于 `贴`），中文即可——不在列表里的动作不会被编造，视频里没发生的动作不会输出
3. 模型默认 `qwen3-vl-plus`（质量优先；`flash` 更便宜但没必要）
4. 点「开始动作检测」→ 完成后**复核播放器直接播视频，检测出的动作段以字幕显示**；下方表格列出所有段落；可下载 JSON + SRT
5. 多个视频用下拉框切换；超过 ~36 秒的视频自动切块处理，无需关心

## 方式二：CLI

```bash
# 物品检测全流程：抽帧 → 检测 → 划分 → 导出 YOLO
python -m auto_label run 贴黑垫子 --classes "washer, screw"

# 分步执行
python -m auto_label extract 贴黑垫子              # 抽帧（--stride 15）
python -m auto_label detect 贴黑垫子 --classes "washer"
python -m auto_label export 贴黑垫子               # --split-ratio 0.8,0.1,0.1
python -m auto_label visualize 贴黑垫子            # 画框质检图

# 动作检测（结果写入 datasets/贴黑垫子/annotations/action/）
python -m auto_label detect-actions 贴黑垫子 --actions "拿起产品,贴黑垫子,扫码,放下产品"
#   --videos a.mp4,b.mp4   只跑指定视频（默认 raw/ 全部）
#   --fps 2                抽帧率，边界精度约 ±1/fps 秒
#   --model qwen3-vl-flash 换降本模型

# 其他
python -m auto_label list                          # 列出数据集
python -m auto_label download-models               # 预下载检测模型权重
```

## 输出格式

**物品检测**（`datasets/<名>/annotations/detection/`）：标准 YOLO 结构，`train|val|test/{images,labels}` + `data.yaml`，label 每行 `class_id x_center y_center width height`（归一化 0-1），可直接喂 YOLO 训练。

**动作检测**（`datasets/<名>/annotations/action/`）：

```jsonc
// 20260521_140944.json
{
  "video": "20260521_140944.mp4",
  "duration": 22.1,
  "model": "qwen3-vl-plus",
  "fps": 2.0,
  "actions": ["拿起产品", "贴黑垫子", "扫码", "放下产品"],
  "segments": [
    {"action": "拿起产品", "start": 0.0, "end": 1.5},
    {"action": "贴黑垫子", "start": 2.0, "end": 6.0}
  ],
  "usage": {"prompt_tokens": 6846, "completion_tokens": 99, "total_tokens": 6945}
}
```

同名 `.srt` 字幕：Web UI 播放器自动加载；也可以用 IINA/VLC 打开原视频后把 .srt 拖进去，播放时段落直接叠在画面上。`_summary.json` 汇总本次运行的动作列表和每个视频的 token 消耗。

## 动作检测成本参考（2026-07 牌价，以百炼控制台为准）

| 规模 | 成本 |
|---|---|
| 20 秒工位视频 | ≈ 0.008 元（不到 1 分钱） |
| 87 秒视频 | ≈ 0.035 元 |
| 1 小时视频量 | ≈ 1.4 元 |
| 1000 个 20s 视频 | ≈ 8 元 |

折合 ≈0.0004 元/视频秒（2fps、640px）。98% 消耗在输入帧，想省钱只有两个旋钮：降 fps（伤边界精度）、降分辨率；批量推理接口半价。当前规模下成本可忽略。

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 中文类别名检测置信度很低（物品检测） | 三个本地模型的文本编码器以英文为主，实测 `washer` 0.40+ 而 `垫圈` <0.19。先用英文提示词，中文命名/英文检测的解耦方案在 roadmap |
| 检测越跑越慢甚至 OOM（Apple Silicon） | 已内置每 batch 释放 MPS 显存；仍紧张就把 batch size 降到 1 |
| UI 里首次选新模型卡住 | 在下载权重（OWLv2 ~1GB、Florence-2 ~1.5GB）。先跑 `download-models` 可避免 |
| 代理/网络抖动导致模型加载失败 | 已缓存的模型自动离线加载不联网；动作检测 API 自动重试 3 次 |
| Florence-2 报 `AttributeError` | transformers 被升级了，装回 `==4.49.0` |
| 动作边界差零点几秒 | 精度上限 ≈ ±1/fps 秒（默认 2fps 即 ±0.5s），需求内属正常；要更细可提高 `--fps`（成本线性上涨） |
| 动作检测报缺少 API key | `DASHSCOPE_API_KEY` 写入项目根 `.env` 或环境变量 |

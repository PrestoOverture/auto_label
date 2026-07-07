# 动作检测 POC

> **状态：POC 已完成并通过（Qwen 路线），已 pipeline 化。** 正式实现见
> `src/auto_label/action.py`（CLI：`python -m auto_label detect-actions`，Web UI：动作检测 Tab）。
> 本目录保留 X-CLIP 对比实验和当时的判断依据。

验证 zero-shot 时序动作定位(视频 + 自定义动作列表 → 起止时间戳 + 动作)在本项目工位视频上的实际效果,**先验证质量,再决定投入搭完整 pipeline**。

需求(2026-07-06 与总监对齐):

- 输入:视频 + 完整短语动作列表(如 "拿起产品"、"贴黑垫子"),开放词汇可自定义
- 输出:每段动作的起止时间,如 `0:30-0:35 拿起垫圈`
- 约束:无标注数据(zero-shot);单人顺序操作;边界误差 ±1s 可接受
- 远期:效果好 → 蒸馏可部署小模型;效果差 → 转人工 review 辅助工具

## 两条路线

| 路线 | 原理 | 跑在哪 |
|---|---|---|
| A `xclip` | X-CLIP 2s 滑窗对英文动作描述打分 → 平滑 → 阈值 → 合并成段 | 本机(8GB M1 可跑) |
| B `qwen` | 抽帧整段送 Qwen-VL(DashScope API),直接输出 JSON 时间段 | 云端 API |

## 用法

```bash
# 路线 A(首次会下载 ~600MB 模型)
.venv/bin/python poc/action_poc.py xclip datasets/贴黑垫子/raw/*.mp4 \
    --actions poc/actions_贴黑垫子.json

# 路线 B(需要阿里云百炼 API key)
export DASHSCOPE_API_KEY=sk-xxx
.venv/bin/python poc/action_poc.py qwen datasets/贴黑垫子/raw/20260521_140944.mp4 \
    --actions poc/actions_贴黑垫子.json
# 不想花钱先看看会发什么: 加 --dry-run
```

## 动作配置文件

每个动作 = `name`(中文显示名,给 Qwen 和最终输出)+ `prompt_en`(给 X-CLIP 的英文描述)——
"显示名/提示词"解耦,和物品检测规划的中文类名方案同一设计。`actions_*.json` 里的动作列表
是按抽帧画面猜的工序,**用之前请按实际工序修改**。

## 输出

`poc/output/{xclip,qwen}/`:

- `{video}.json` — 机器可读段落:`{action, start, end, confidence}`
- `{video}.srt` — 用 IINA/VLC 打开原视频再拖入此字幕,播放时动作段落直接叠在画面上,肉眼复核最快
- `{video}.raw.txt` — (仅 qwen)模型原始回复,便于排查

## 判断标准(目检)

1. 段落数量和顺序是否符合实际工序(拿起 → 贴 → 扫码 → 放下)
2. 边界误差是否 ≤1s
3. 有没有编造不存在的动作 / 漏掉明显的动作

结论写回 docs/roadmap.md,由结果决定:继续搭 pipeline(接入 CLI + Web UI Tab + 双路 consensus)还是调整路线。

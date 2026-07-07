"""POC: zero-shot 时序动作定位 (temporal action localization).

验证"视频 + 自定义动作列表 -> 时间段 + 动作"两条路线的实际效果:

  A. xclip  本地 X-CLIP 滑窗打分, 轻量 (~600MB), 8GB M1 可跑
  B. qwen   云端 Qwen-VL API (DashScope), 整段视频直接输出 JSON 时间段

用法:
  python poc/action_poc.py xclip <video.mp4>... --actions poc/actions_贴黑垫子.json
  python poc/action_poc.py qwen  <video.mp4>... --actions poc/actions_贴黑垫子.json
  (qwen 需要环境变量 DASHSCOPE_API_KEY; --dry-run 可以不调 API 只看 payload)

输出: poc/output/{route}/{video_stem}.json + .srt
  .srt 用 IINA/VLC 打开视频后拖进去, 播放时动作段落直接显示为字幕。
"""

import base64
import json
import os
import re
import sys
from pathlib import Path

import click
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from auto_label.detect import _release_device_memory, auto_device, is_cached  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
XCLIP_MODEL_ID = "microsoft/xclip-base-patch32"
XCLIP_NUM_FRAMES = 8  # base-patch32 权重固定 8 帧输入


# ---------------------------------------------------------------- 公共部分

def load_actions(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg.get("actions"), f"{path} 里没有 actions"
    return cfg


def video_meta(path: Path) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return nframes / fps, fps


def sample_frames(path: Path, sample_fps: float, max_side: int) -> tuple[list[np.ndarray], list[float]]:
    """按 sample_fps 均匀抽帧, 返回 (RGB帧列表, 每帧秒数)."""
    duration, native_fps = video_meta(path)
    n_samples = max(2, int(duration * sample_fps))
    want = {min(int(round(k / sample_fps * native_fps)), int(duration * native_fps) - 1): k
            for k in range(n_samples)}
    frames: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            h, w = frame.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()
    ordered = sorted(frames)
    return [frames[i] for i in ordered], [i / native_fps for i in ordered]


def fmt_ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(video: Path, duration: float, route: str, segments: list[dict]) -> Path:
    out_dir = OUTPUT_DIR / route
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{video.stem}.json").write_text(
        json.dumps({"video": video.name, "duration": round(duration, 2),
                    "source": route, "segments": segments},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        conf = f" ({seg['confidence']:.2f})" if seg.get("confidence") is not None else ""
        srt_lines += [str(i), f"{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}",
                      seg["action"] + conf, ""]
    (out_dir / f"{video.stem}.srt").write_text("\n".join(srt_lines), encoding="utf-8")
    return out_dir


def print_segments(video: Path, segments: list[dict]) -> None:
    click.echo(f"\n== {video.name} ==")
    if not segments:
        click.echo("  (没有检出任何动作段)")
    for seg in segments:
        conf = f"  conf={seg['confidence']:.2f}" if seg.get("confidence") is not None else ""
        click.echo(f"  {fmt_ts(seg['start'])[3:8]} - {fmt_ts(seg['end'])[3:8]}  {seg['action']}{conf}")


@click.group()
def cli():
    """Zero-shot 动作定位 POC."""


# ---------------------------------------------------------------- 路线 A: X-CLIP 滑窗

@cli.command("xclip")
@click.argument("videos", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--actions", "actions_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--window", default=2.0, show_default=True, help="滑窗时长(秒)")
@click.option("--stride", default=0.5, show_default=True, help="滑窗步长(秒)")
@click.option("--threshold", default=0.4, show_default=True, help="动作概率阈值")
@click.option("--min-duration", default=0.8, show_default=True, help="过滤短于此时长的段")
@click.option("--batch-size", default=4, show_default=True)
def run_xclip(videos, actions_path, window, stride, threshold, min_duration, batch_size):
    """本地 X-CLIP 滑窗: 每 2s 窗口对动作英文描述打分, 合并成段."""
    import torch
    from transformers import AutoProcessor, XCLIPModel

    cfg = load_actions(actions_path)
    actions = cfg["actions"]
    bg_prompts = cfg.get("background_prompts_en", ["no action, hands idle"])
    texts = [a["prompt_en"] for a in actions] + bg_prompts
    n_act = len(actions)

    device = auto_device()
    kw = {"local_files_only": True} if is_cached(XCLIP_MODEL_ID) else {}
    click.echo(f"加载 {XCLIP_MODEL_ID} (device={device}, cached={bool(kw)}) ...")
    processor = AutoProcessor.from_pretrained(XCLIP_MODEL_ID, **kw)
    model = XCLIPModel.from_pretrained(XCLIP_MODEL_ID, **kw).to(device).eval()

    sample_fps = XCLIP_NUM_FRAMES / window          # 窗口内恰好 8 帧
    stride_n = max(1, int(round(stride * sample_fps)))

    for video in videos:
        duration, _ = video_meta(video)
        frames, times = sample_frames(video, sample_fps, max_side=256)
        while len(frames) < XCLIP_NUM_FRAMES:       # 视频过短时补尾帧
            frames.append(frames[-1]); times.append(times[-1])

        starts = list(range(0, len(frames) - XCLIP_NUM_FRAMES + 1, stride_n)) or [0]
        clips = [frames[s:s + XCLIP_NUM_FRAMES] for s in starts]
        centers = [float(np.mean(times[s:s + XCLIP_NUM_FRAMES])) for s in starts]

        probs = []
        with torch.no_grad():
            for b in range(0, len(clips), batch_size):
                inputs = processor(text=texts, videos=clips[b:b + batch_size],
                                   return_tensors="pt", padding=True).to(device)
                out = model(**inputs)
                probs.append(out.logits_per_video.softmax(dim=1).cpu().numpy())
                _release_device_memory(device)
        probs = np.concatenate(probs)               # (n_win, n_act + n_bg)

        # 平滑(滑动平均 k=3) -> 逐窗决策 -> 合并
        k = min(3, len(probs))
        kernel = np.ones(k) / k
        smoothed = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, probs)
        labels = []
        for row in smoothed:
            bg = row[n_act:].sum()
            best = int(np.argmax(row[:n_act]))
            labels.append(best if row[best] > max(bg, threshold) else None)

        segments, cur = [], None
        half = window / 2
        for lab, center, row in zip(labels, centers, smoothed):
            if lab is not None and cur and cur["idx"] == lab and center - half <= cur["end"] + stride:
                cur["end"] = min(center + half, duration)
                cur["probs"].append(row[lab])
            else:
                if cur:
                    segments.append(cur)
                cur = ({"idx": lab, "start": max(center - half, 0.0),
                        "end": min(center + half, duration), "probs": [row[lab]]}
                       if lab is not None else None)
        if cur:
            segments.append(cur)
        for prev, nxt in zip(segments, segments[1:]):   # 单人顺序假设: 相邻段不重叠, 取中点切开
            if nxt["start"] < prev["end"]:
                prev["end"] = nxt["start"] = (nxt["start"] + prev["end"]) / 2
        segments = [
            {"action": actions[s["idx"]]["name"],
             "start": round(s["start"], 2), "end": round(s["end"], 2),
             "confidence": round(float(np.mean(s["probs"])), 3)}
            for s in segments if s["end"] - s["start"] >= min_duration
        ]
        out_dir = write_outputs(video, duration, "xclip", segments)
        print_segments(video, segments)
    click.echo(f"\n输出目录: {out_dir}")


# ---------------------------------------------------------------- 路线 B: Qwen-VL API

QWEN_PROMPT = """你是工业视频动作标注助手。这是一段工位操作视频, 总时长 {duration:.1f} 秒。
以上传入的 {n_frames} 帧从 0 秒到 {last_ts:.1f} 秒均匀采样, 相邻帧间隔 {interval:.2f} 秒, 第 i 帧(从1数)对应时间 (i-1)*{interval:.2f} 秒。

需要定位的动作列表:
{action_list}

要求:
1. 找出视频中上述动作每次出现的起止时间(秒), 同一动作出现多次就分别列出
2. 操作者是单人, 动作顺序发生、互不重叠
3. 列表之外的动作、以及未出现的动作, 不要输出, 不要编造
4. 只输出 JSON 数组, 不要任何其他文字, 格式:
[{{"action": "动作名", "start": 起始秒, "end": 结束秒}}]"""


@cli.command("qwen")
@click.argument("videos", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--actions", "actions_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--model", default="qwen3-vl-plus", show_default=True)
@click.option("--fps", default=2.0, show_default=True, help="送给模型的抽帧率(长视频会自动降到64帧以内)")
@click.option("--dry-run", is_flag=True, help="只打印 prompt 和 payload 大小, 不调 API")
def run_qwen(videos, actions_path, model, fps, dry_run):
    """云端 Qwen-VL: 抽帧整段送 API, 直接要 JSON 时间段."""
    import requests

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key and not dry_run:
        raise click.ClickException("请先设置环境变量 DASHSCOPE_API_KEY (阿里云百炼控制台获取)")
    base_url = os.environ.get("DASHSCOPE_BASE_URL",
                              "https://dashscope.aliyuncs.com/compatible-mode/v1")

    cfg = load_actions(actions_path)
    names = [a["name"] for a in cfg["actions"]]

    for video in videos:
        duration, _ = video_meta(video)
        eff_fps = min(fps, 64 / duration)           # DashScope 单请求帧数上限保守按 64
        if eff_fps < 1.5:
            click.echo(f"  !! {video.name} 时长 {duration:.0f}s, 抽帧降到 {eff_fps:.2f}fps, "
                       f"边界精度会明显退化; POC 建议先用 <40s 的视频, 长视频等切块逻辑", err=True)
        frames, times = sample_frames(video, eff_fps, max_side=640)
        data_urls = []
        for f in frames:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            data_urls.append("data:image/jpeg;base64," + base64.b64encode(buf).decode())

        interval = times[1] - times[0] if len(times) > 1 else duration
        prompt = QWEN_PROMPT.format(
            duration=duration, n_frames=len(frames), last_ts=times[-1], interval=interval,
            action_list="\n".join(f"- {n}" for n in names))
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "video", "video": data_urls, "fps": round(eff_fps, 2)},
                {"type": "text", "text": prompt},
            ]}],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        size_mb = len(json.dumps(payload)) / 1e6
        click.echo(f"\n== {video.name} ==  {len(frames)} 帧, payload {size_mb:.1f} MB, model={model}")
        if dry_run:
            click.echo(prompt)
            continue

        resp = requests.post(f"{base_url}/chat/completions",
                             headers={"Authorization": f"Bearer {api_key}"},
                             json=payload, timeout=300)
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        out_dir = OUTPUT_DIR / "qwen"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{video.stem}.raw.txt").write_text(reply, encoding="utf-8")

        segments = parse_qwen_reply(reply, names, duration)
        write_outputs(video, duration, "qwen", segments)
        print_segments(video, segments)
    if not dry_run:
        click.echo(f"\n输出目录: {OUTPUT_DIR / 'qwen'}")


def _to_seconds(v) -> float:
    """兼容 12.5 / "12.5" / "00:12" / "00:00:12.5" 几种时间戳写法."""
    if isinstance(v, (int, float)):
        return float(v)
    parts = [float(p) for p in str(v).strip().split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def parse_qwen_reply(reply: str, valid_names: list[str], duration: float) -> list[dict]:
    """容错解析: 剥 code fence, 找最外层 JSON 数组, 校验并 clamp 时间戳."""
    text = re.sub(r"```(?:json)?|```", "", reply).strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        click.echo(f"  !! 模型回复里找不到 JSON 数组: {reply[:200]}", err=True)
        return []
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        click.echo(f"  !! JSON 解析失败: {e}", err=True)
        return []
    segments = []
    for item in raw:
        action = str(item.get("action", "")).strip()
        if action not in valid_names:
            click.echo(f"  !! 丢弃列表外动作: {action}", err=True)
            continue
        try:
            start, end = _to_seconds(item["start"]), _to_seconds(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        start, end = max(0.0, start), min(end, duration)
        if end > start:
            segments.append({"action": action, "start": round(start, 2),
                             "end": round(end, 2), "confidence": None})
    segments.sort(key=lambda s: s["start"])
    return segments


if __name__ == "__main__":
    cli()

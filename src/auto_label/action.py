"""Zero-shot 时序动作定位: 视频 + 自定义动作短语列表 -> [(动作, 起始秒, 结束秒)].

调用云端 Qwen-VL (DashScope OpenAI 兼容接口), 无本地模型。
长视频自动切块(带重叠)分段请求, 结果按全局时间轴合并。
"""

import base64
import json
import os
import re
import time
from pathlib import Path

import cv2

ACTION_MODELS = ["qwen3-vl-plus", "qwen3-vl-flash"]
DEFAULT_ACTION_MODEL = "qwen3-vl-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

MAX_FRAMES_PER_REQUEST = 64
MAX_CHUNK_SECONDS = 30.0
CHUNK_OVERLAP = 3.0
MERGE_GAP = 1.0

PROMPT_TEMPLATE = """你是工业视频动作标注助手。这是一段工位操作视频, 时长 {duration:.1f} 秒。
传入的 {n_frames} 帧均匀采样, 相邻帧间隔 {interval:.2f} 秒, 第 i 帧(从1数)对应时间 (i-1)*{interval:.2f} 秒。

需要定位的动作列表:
{action_list}

要求:
1. 找出视频中上述动作每次出现的起止时间(秒), 同一动作出现多次就分别列出
2. 操作者是单人, 动作顺序发生、互不重叠
3. 列表之外的动作、以及未出现的动作, 不要输出, 不要编造
4. 只输出 JSON 数组, 不要任何其他文字, 时间用秒(可带小数), 格式:
[{{"action": "动作名", "start": 起始秒, "end": 结束秒}}]"""


def get_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if key:
        return key
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DASHSCOPE_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def sample_video_frames(video_path: Path, fps: float, max_side: int = 640):
    """均匀抽帧并即时压成 JPEG bytes(内存友好), 返回 (jpeg列表, 秒数列表, 视频时长)."""
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"无法读取视频: {video_path}")
    duration = total / native_fps

    want = set()
    for k in range(max(2, int(duration * fps))):
        want.add(min(int(round(k / fps * native_fps)), total - 1))

    jpegs, times = [], []
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
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2:
                jpegs.append(buf.tobytes())
                times.append(idx / native_fps)
        idx += 1
    cap.release()
    return jpegs, times, duration


def _chunk_windows(duration: float, chunk_seconds: float, overlap: float) -> list[tuple[float, float]]:
    if duration <= chunk_seconds * 1.2:
        return [(0.0, duration)]
    windows, start = [], 0.0
    while True:
        end = min(start + chunk_seconds, duration)
        windows.append((start, end))
        if end >= duration:
            return windows
        start = end - overlap


def _to_seconds(v) -> float:
    """兼容 12.5 / "12.5" / "00:12" / "00:00:12.5" 几种时间戳写法."""
    if isinstance(v, (int, float)):
        return float(v)
    sec = 0.0
    for part in str(v).strip().split(":"):
        sec = sec * 60 + float(part)
    return sec


def parse_reply(reply: str, valid_names: list[str], max_seconds: float) -> tuple[list[dict], list[str]]:
    """容错解析模型回复, 返回 (segments, 警告列表)."""
    warnings = []
    text = re.sub(r"```(?:json)?|```", "", reply).strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return [], [f"回复里找不到 JSON 数组: {reply[:150]}"]
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return [], [f"JSON 解析失败: {e}"]

    segments = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if action not in valid_names:
            warnings.append(f"丢弃列表外动作: {action}")
            continue
        try:
            start, end = _to_seconds(item["start"]), _to_seconds(item["end"])
        except (KeyError, TypeError, ValueError):
            warnings.append(f"丢弃无效时间戳: {item}")
            continue
        start, end = max(0.0, start), min(end, max_seconds)
        if end > start:
            segments.append({"action": action, "start": round(start, 2), "end": round(end, 2)})
    segments.sort(key=lambda s: s["start"])
    return segments, warnings


def merge_segments(segments: list[dict], merge_gap: float = MERGE_GAP) -> list[dict]:
    """跨切块合并: 同名相邻/重叠段取并集; 异名重叠(单人顺序假设)从中点切开."""
    segs = sorted((dict(s) for s in segments), key=lambda s: (s["start"], s["end"]))
    merged: list[dict] = []
    for s in segs:
        if merged and merged[-1]["action"] == s["action"] and s["start"] - merged[-1]["end"] <= merge_gap:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(s)
    for prev, nxt in zip(merged, merged[1:]):
        if nxt["start"] < prev["end"]:
            mid = round((prev["end"] + nxt["start"]) / 2, 2)
            prev["end"] = nxt["start"] = mid
    return [s for s in merged if s["end"] > s["start"]]


def _call_api(base_url: str, api_key: str, model: str, data_urls: list[str],
              fps: float, prompt: str) -> tuple[str, dict]:
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "video", "video": data_urls, "fps": round(fps, 2)},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    last_err = None
    for attempt in range(3):
        if attempt:
            time.sleep(5)
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload, timeout=(10, 300),
            )
        except requests.RequestException as e:
            last_err = f"网络错误: {e}"
            continue
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"], data.get("usage", {})
        if resp.status_code < 500:
            raise RuntimeError(f"API 调用失败 HTTP {resp.status_code}: {resp.text[:300]}")
        last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
    raise RuntimeError(f"API 重试 3 次后仍失败: {last_err}")


def detect_actions_video(video_path: Path, actions: list[str],
                         model: str = DEFAULT_ACTION_MODEL, fps: float = 2.0,
                         api_key: str | None = None, base_url: str | None = None):
    """生成器: 产出 ("log", 进度消息)..., 最后产出 ("done", 结果 dict)."""
    video_path = Path(video_path)
    api_key = api_key or get_api_key()
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY: 请设置环境变量, 或写入项目根目录 .env 文件")
    base_url = base_url or os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
    names = [a.strip() for a in actions if a.strip()]
    if not names:
        raise RuntimeError("动作列表为空")

    jpegs, times, duration = sample_video_frames(video_path, fps)
    chunk_seconds = min(MAX_CHUNK_SECONDS, (MAX_FRAMES_PER_REQUEST - 4) / fps)
    windows = _chunk_windows(duration, chunk_seconds, CHUNK_OVERLAP)
    if len(windows) > 1:
        yield "log", f"视频 {duration:.0f}s 超过单请求容量, 切成 {len(windows)} 段 (每段 ≤{chunk_seconds:.0f}s, 重叠 {CHUNK_OVERLAP:.0f}s)"

    all_segments = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for wi, (ws, we) in enumerate(windows, 1):
        sel = [i for i, t in enumerate(times) if ws <= t <= we]
        if not sel:
            continue
        t0 = times[sel[0]]
        data_urls = ["data:image/jpeg;base64," + base64.b64encode(jpegs[i]).decode() for i in sel]
        interval = 1.0 / fps
        prompt = PROMPT_TEMPLATE.format(
            duration=we - ws, n_frames=len(sel), interval=interval,
            action_list="\n".join(f"- {n}" for n in names))
        reply, usage = _call_api(base_url, api_key, model, data_urls, fps, prompt)
        for k in usage_total:
            usage_total[k] += usage.get(k, 0) or 0
        segments, warnings = parse_reply(reply, names, we - ws + 1.0)
        for w in warnings:
            yield "log", f"注意: {w}"
        for s in segments:
            s["start"] = round(min(s["start"] + t0, duration), 2)
            s["end"] = round(min(s["end"] + t0, duration), 2)
        all_segments.extend(segments)
        if len(windows) > 1:
            yield "log", f"片段 {wi}/{len(windows)} ({ws:.0f}-{we:.0f}s): {len(segments)} 段"

    yield "done", {
        "video": video_path.name,
        "duration": round(duration, 2),
        "model": model,
        "fps": fps,
        "actions": names,
        "segments": merge_segments(all_segments),
        "usage": usage_total,
    }


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], path: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}",
                  seg["action"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_action_outputs(result: dict, out_dir: Path) -> tuple[Path, Path]:
    """写 {stem}.json + {stem}.srt, 返回两个路径."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result["video"]).stem
    json_path = out_dir / f"{stem}.json"
    srt_path = out_dir / f"{stem}.srt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_srt(result["segments"], srt_path)
    return json_path, srt_path

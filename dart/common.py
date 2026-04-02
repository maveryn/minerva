from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import pandas as pd
import yaml


FINAL_MARKER_RE = re.compile(
    r"(assistantfinal|<\|channel\|>final<\|message\|>|<\|assistant\|>final)",
    flags=re.IGNORECASE,
)
ANALYSIS_MARKER_RE = re.compile(
    r"^(?:<\|channel\|>analysis<\|message\|>|analysis\s*[:\-]*\s*)",
    flags=re.IGNORECASE,
)
FINAL_PREFIX_RE = re.compile(
    r"^(?:<\|channel\|>final<\|message\|>|final\s*[:\-]*\s*)",
    flags=re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{[^}]*\}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def start_wall_clock() -> tuple[str, float]:
    return utc_now_iso(), time.perf_counter()


def finish_wall_clock(started_at_utc: str, started_perf: float) -> Dict[str, Any]:
    return {
        "started_at_utc": str(started_at_utc),
        "finished_at_utc": utc_now_iso(),
        "elapsed_seconds": round(time.perf_counter() - float(started_perf), 3),
    }


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]], mode: str = "w") -> None:
    path = ensure_parent(path)
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    write_jsonl(path, rows, mode="a")


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def normalize_messages(value: Any) -> List[Dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError(f"Expected list-like messages, got {type(value).__name__}")
    out: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "")
        if role and content:
            out.append({"role": role, "content": content})
    return out


def first_present_field(row: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list) and not value:
            continue
        return value
    return default


def split_reasoning_response(text: str | None) -> tuple[str, str]:
    if not isinstance(text, str):
        return "", ""
    raw = text.strip()
    if not raw:
        return "", ""

    analysis = ""
    final = raw
    match = FINAL_MARKER_RE.search(raw)
    if match:
        analysis = raw[: match.start()].strip()
        final = raw[match.end() :].strip()
    else:
        box_matches = list(BOXED_RE.finditer(raw))
        if box_matches:
            last = box_matches[-1]
            analysis = (raw[: last.start()] + raw[last.end() :]).strip()
            final = last.group(0).strip()

    if analysis:
        analysis = ANALYSIS_MARKER_RE.sub("", analysis).strip()
    if final:
        final = FINAL_PREFIX_RE.sub("", final).strip()
    return analysis, final


def compose_training_response(
    text: str | None = None,
    *,
    analysis: str | None = None,
    final: str | None = None,
) -> str:
    if analysis is None and final is None:
        analysis, final = split_reasoning_response(text)
    analysis = str(analysis or "").strip()
    final = str(final or "").strip()
    if analysis and final:
        last_box = None
        box_matches = list(BOXED_RE.finditer(analysis))
        if box_matches:
            last_box = box_matches[-1].group(0).strip()
        if analysis.rstrip().endswith(final) or last_box == final:
            return analysis
        return f"{analysis}\n\n{final}"
    return analysis or final


def get_system_prompt(messages: List[Dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "system":
            return message.get("content", "")
    return ""


def get_user_prompt(messages: List[Dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def stringify_ground_truth(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)
    return str(value)


def make_uid(row: Dict[str, Any], fallback_index: int) -> str:
    if row.get("uid"):
        return str(row["uid"])
    extra_info = row.get("extra_info") or {}
    source_file = str(extra_info.get("source_file") or row.get("source_file") or "unknown")
    source_index = extra_info.get("index", fallback_index)
    data_source = str(row.get("data_source") or "")
    return f"{source_file}:{source_index}:{data_source}"


def normalize_path_list(paths: str | Path | Sequence[str | Path]) -> List[str]:
    if isinstance(paths, (str, Path)):
        return [str(paths)]
    return [str(path) for path in paths]


def load_parquet_rows(paths: str | Path | Sequence[str | Path], limit: int | None = None) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    remaining = None if limit is None else max(0, int(limit))
    for path in normalize_path_list(paths):
        if remaining == 0:
            break
        df = pd.read_parquet(path)
        if remaining is not None:
            df = df.head(remaining)
        rows = df.to_dict(orient="records")
        all_rows.extend(rows)
        if remaining is not None:
            remaining -= len(rows)
    return all_rows


def is_athena_row(data_source: str) -> bool:
    return str(data_source or "").startswith("athena-cti-")


def is_seceval_row(data_source: str) -> bool:
    return str(data_source or "") in {"seceval", "seceval-mini"}


def summarize_counts(values: Iterable[int]) -> Dict[str, Any]:
    vals = [int(v) for v in values]
    if not vals:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": float(sum(vals) / len(vals)),
    }

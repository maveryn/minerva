"""SQL-R1 reward port for Minerva/VeRL custom reward loading.

This implementation mirrors the SynSQL reward shape from SQL-R1 while adapting
it to Minerva's response-only reward manager interface.
"""

from __future__ import annotations

import os
import re
import signal
import sqlite3
import threading
import time
from collections import defaultdict
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional
import unicodedata


class _RewardTimeout(Exception):
    pass


def _run_with_timeout(deadline_s: int, func, *args, **kwargs):
    deadline_s = max(1, int(deadline_s))
    if deadline_s <= 0:
        return func(*args, **kwargs)

    if threading.current_thread() is not threading.main_thread():
        return func(*args, **kwargs)

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _alarm_handler(_signum, _frame):
        raise _RewardTimeout()

    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, float(deadline_s))
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _strip_response_prefix(text: str) -> str:
    candidate = (text or "").strip()
    if not candidate:
        return ""

    if "<|im_start|>assistant" in candidate:
        candidate = candidate.split("<|im_start|>assistant", 1)[1]
    elif "Assistant:" in candidate:
        candidate = candidate.split("Assistant:", 1)[1]
    return candidate.strip()


def _normalize_response_for_parsing(solution_str: str) -> str:
    candidate = _strip_response_prefix(solution_str)
    if not candidate:
        return ""
    if "<think>" not in candidate and ("</think>" in candidate or "<answer>" in candidate):
        candidate = "<think>" + candidate
    return candidate


def _extract_solution_parts(solution_str: str) -> tuple[Optional[str], Optional[str], str]:
    processed = _normalize_response_for_parsing(solution_str)
    if not processed:
        return None, None, processed

    think_matches = list(re.finditer(r"<think>(.*?)</think>", processed, re.DOTALL | re.IGNORECASE))
    answer_matches = list(re.finditer(r"<answer>(.*?)</answer>", processed, re.DOTALL | re.IGNORECASE))

    think_text = think_matches[-1].group(1).strip() if think_matches else None
    answer_text = answer_matches[-1].group(1).strip() if answer_matches else None
    return answer_text, think_text, processed


def _parse_sql_from_answer(answer_text: Optional[str]) -> Optional[str]:
    if not answer_text:
        return None
    matches = list(re.finditer(r"```sql\s*(.*?)```", answer_text, re.DOTALL | re.IGNORECASE))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _validate_response_structure(answer_text: Optional[str], processed_str: str) -> tuple[Optional[str], bool]:
    if not processed_str:
        return None, False

    tags = ("<think>", "</think>", "<answer>", "</answer>")
    tag_counts = {tag: processed_str.count(tag) for tag in tags}
    if any(count != 1 for count in tag_counts.values()):
        return None, False

    think_start = processed_str.find("<think>")
    think_end = processed_str.find("</think>")
    answer_start = processed_str.find("<answer>")
    answer_end = processed_str.find("</answer>")
    if not (0 <= think_start < think_end < answer_start < answer_end):
        return None, False

    pred_sql = _parse_sql_from_answer(answer_text)
    if not pred_sql:
        return None, False
    return pred_sql, True


def _replace_cur_year(query: str) -> str:
    return re.sub(r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE)


def _postprocess_query(query: str) -> str:
    return query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")


def _remove_distinct(query: str) -> str:
    return re.sub(r"(?i)\bselect\s+distinct\b", "SELECT", query)


def _permute_tuple(element: tuple[Any, ...], perm: tuple[int, ...]) -> tuple[Any, ...]:
    return tuple(element[i] for i in perm)


def _unorder_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(sorted(row, key=lambda x: f"{x}{type(x)}"))


def _quick_rej(result1: list[tuple[Any, ...]], result2: list[tuple[Any, ...]], order_matters: bool) -> bool:
    s1 = [_unorder_row(row) for row in result1]
    s2 = [_unorder_row(row) for row in result2]
    return s1 == s2 if order_matters else set(s1) == set(s2)


def _multiset_eq(left: list[tuple[Any, ...]], right: list[tuple[Any, ...]]) -> bool:
    if len(left) != len(right):
        return False
    counts = defaultdict(int)
    for value in left:
        counts[value] += 1
    for value in right:
        counts[value] -= 1
        if counts[value] < 0:
            return False
    return True


def _get_constraint_permutation(
    table1_sets_by_column: list[set[Any]], result2: list[tuple[Any, ...]]
) -> Iterable[tuple[int, ...]]:
    num_cols = len(result2[0])
    constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*constraints)

    sample_rows = result2[: min(20, len(result2))]
    for row in sample_rows:
        for col_a in range(num_cols):
            for col_b in set(constraints[col_a]):
                if row[col_b] not in table1_sets_by_column[col_a]:
                    constraints[col_a].remove(col_b)
    return product(*constraints)


def _result_eq(result1: list[tuple[Any, ...]], result2: list[tuple[Any, ...]], order_matters: bool) -> bool:
    if not result1 and not result2:
        return True
    if len(result1) != len(result2):
        return False
    if not result1 or not result2:
        return False

    num_cols = len(result1[0])
    if len(result2[0]) != num_cols:
        return False
    if not _quick_rej(result1, result2, order_matters):
        return False

    table1_sets_by_column = [{row[i] for row in result1} for i in range(num_cols)]
    for perm in _get_constraint_permutation(table1_sets_by_column, result2):
        if len(perm) != len(set(perm)):
            continue
        result2_perm = result2 if num_cols == 1 else [_permute_tuple(row, perm) for row in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        elif set(result1) == set(result2_perm) and _multiset_eq(result1, result2_perm):
            return True
    return False


def _execute_on_db(sqlite_path: str, query: str, timeout_s: int) -> tuple[str, Any]:
    query = _replace_cur_year(query)
    connection = sqlite3.connect(sqlite_path)
    connection.text_factory = lambda b: b.decode(errors="ignore")
    deadline = time.monotonic() + max(1, int(timeout_s))

    def _progress_handler() -> int:
        return 1 if time.monotonic() >= deadline else 0

    connection.set_progress_handler(_progress_handler, 10_000)
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        return "result", cursor.fetchall()
    except Exception as exc:  # pragma: no cover - depends on sqlite execution path
        return "exception", exc
    finally:
        try:
            cursor.close()
        finally:
            connection.close()


def _eval_exec_match(db_path: str, pred_sql: str, gold_sql: str, keep_distinct: bool, timeout_s: int) -> str:
    pred_sql = _postprocess_query(pred_sql)
    gold_sql = _postprocess_query(gold_sql)
    if not keep_distinct:
        pred_sql = _remove_distinct(pred_sql)
        gold_sql = _remove_distinct(gold_sql)

    order_matters = "order by" in gold_sql.lower()
    db_dir = os.path.dirname(db_path)
    db_candidates = sorted(
        os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if basename.endswith(".sqlite")
    )
    if not db_candidates:
        db_candidates = [db_path]

    pred_passes = True
    for candidate in db_candidates:
        gold_flag, gold_denotation = _execute_on_db(candidate, gold_sql, timeout_s=timeout_s)
        pred_flag, pred_denotation = _execute_on_db(candidate, pred_sql, timeout_s=timeout_s)

        if pred_flag == "exception":
            return "Unexecutable"
        if gold_flag == "exception":
            return "Gold Error"
        if not _result_eq(gold_denotation, pred_denotation, order_matters=order_matters):
            pred_passes = False
            break

    return "Match" if pred_passes else "Mismatch"


def _resolve_db_root(extra_info: Any, db_root: Optional[str]) -> Optional[Path]:
    if db_root:
        return Path(db_root).expanduser().resolve()

    if isinstance(extra_info, dict):
        for key in ("db_root", "database_root", "sql_r1_db_root"):
            value = extra_info.get(key)
            if isinstance(value, str) and value.strip():
                return Path(value).expanduser().resolve()

    for env_name in ("SQL_R1_DB_ROOT", "SQLR1_DB_ROOT"):
        value = os.getenv(env_name)
        if value:
            return Path(value).expanduser().resolve()
    return None


def _normalize_db_key(value: str) -> str:
    candidate = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode().lower()
    candidate = candidate.replace("c++", "cplusplus")
    candidate = candidate.replace("'", "").replace("’", "").replace(".", "")
    candidate = re.sub(r"[^a-z0-9]+", "_", candidate)
    return re.sub(r"_+", "_", candidate).strip("_")


@lru_cache(maxsize=16)
def _build_db_alias_index(root_str: str) -> dict[str, str]:
    root = Path(root_str)
    aliases: dict[str, str] = {}
    if not root.is_dir():
        return aliases

    for entry in root.iterdir():
        if entry.is_dir():
            sqlite_files = sorted(entry.glob("*.sqlite"))
            if not sqlite_files:
                continue
            sqlite_path = sqlite_files[0]
            for alias_source in (entry.name, sqlite_path.stem):
                alias_key = _normalize_db_key(alias_source)
                if alias_key and alias_key not in aliases:
                    aliases[alias_key] = str(sqlite_path)
            continue

        if entry.is_file() and entry.suffix == ".sqlite":
            alias_key = _normalize_db_key(entry.stem)
            if alias_key and alias_key not in aliases:
                aliases[alias_key] = str(entry)

    return aliases


def _resolve_db_path(db_id: str, extra_info: Any, db_root: Optional[str]) -> Path:
    root = _resolve_db_root(extra_info, db_root)
    if root is None:
        raise RuntimeError(
            "SQL-R1 reward requires a database root. Set custom_reward_function.reward_kwargs.db_root "
            "or the SQL_R1_DB_ROOT environment variable."
        )

    db_dir = root / db_id
    if db_dir.is_dir():
        default_file = db_dir / f"{db_id}.sqlite"
        if default_file.is_file():
            return default_file
        sqlite_files = sorted(db_dir.glob("*.sqlite"))
        if sqlite_files:
            return sqlite_files[0]

    if db_dir.is_file():
        return db_dir

    alias_key = _normalize_db_key(db_id)
    alias_path = _build_db_alias_index(str(root)).get(alias_key)
    if alias_path:
        return Path(alias_path)

    raise FileNotFoundError(f"Could not resolve SQLite database for db_id={db_id!r} under {root}")


def reward_sql_r1(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info=None,
    return_dict: bool = False,
    db_root: Optional[str] = None,
    timeout_s: int = 30,
    format_reward: float = 1.0,
    exec_reward: float = 2.0,
    result_reward: float = 3.0,
    max_length: int = 2048,
    keep_distinct: bool = False,
    **_ignored: Any,
):
    if "synsql" not in str(data_source or "").lower():
        raise NotImplementedError(f"Unsupported SQL-R1 data_source: {data_source!r}")

    db_name = str((ground_truth or {}).get("db_id", "")).strip()
    gold_sql = re.sub(r"\s+", " ", str((ground_truth or {}).get("sql", "") or "")).strip()

    answer_text, think_text, processed_str = _extract_solution_parts(solution_str)
    pred_sql, format_correct = _validate_response_structure(answer_text, processed_str)
    format_score = float(format_reward if format_correct else -abs(format_reward))

    exec_status = "FormatError"
    exec_match = 0.0
    exec_success = 0.0
    exec_score_val = 0.0
    result_score_val = 0.0

    if format_correct and pred_sql and db_name and gold_sql:
        pred_sql_clean = re.sub(r"\s+", " ", pred_sql).strip()
        db_path = _resolve_db_path(db_name, extra_info, db_root)
        try:
            exec_status = _run_with_timeout(
                timeout_s,
                _eval_exec_match,
                str(db_path),
                pred_sql_clean,
                gold_sql,
                keep_distinct=keep_distinct,
                timeout_s=timeout_s,
            )
        except _RewardTimeout:
            exec_status = "Unexecutable"

        if exec_status == "Unexecutable":
            exec_success = 0.0
            exec_score_val = -abs(exec_reward)
        elif exec_status == "Gold Error":
            exec_success = 0.0
            exec_score_val = 0.0
        elif exec_status == "Mismatch":
            exec_success = 1.0
            exec_score_val = float(exec_reward)
            result_score_val = -abs(result_reward)
        elif exec_status == "Match":
            exec_success = 1.0
            exec_match = 1.0
            exec_score_val = float(exec_reward)
            result_score_val = float(result_reward)

    length_score = 0.0
    if exec_status == "Match" and answer_text and pred_sql and think_text is not None:
        total_length = len(think_text) + len(answer_text)
        answer_len = max(1, len(answer_text))
        sql_ratio = len(pred_sql) / answer_len
        if total_length <= max_length:
            length_score = (total_length / max_length) * 0.5 + sql_ratio
        else:
            length_score = 0.5 + sql_ratio

    total_score = float(format_score + exec_score_val + result_score_val + length_score)

    result = {
        "score": total_score,
        "format_correct": bool(format_correct),
        "format_score": float(format_score),
        "exec_status": exec_status,
        "exec_success": float(exec_success),
        "exec_match": float(exec_match),
        "exec_score": float(exec_score_val),
        "result_score": float(result_score_val),
        "length_score": float(length_score),
        "pred_sql": pred_sql or "",
        "db_id": db_name,
    }
    if return_dict:
        return result
    return total_score


__all__ = ["reward_sql_r1"]

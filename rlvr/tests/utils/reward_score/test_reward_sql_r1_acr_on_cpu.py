from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path


def _load_reward_sql_r1_acr():
    reward_dir = Path(__file__).resolve().parents[3] / "verl" / "utils" / "reward_score"

    base_spec = importlib.util.spec_from_file_location("reward_sql_r1", reward_dir / "reward_sql_r1.py")
    base_module = importlib.util.module_from_spec(base_spec)
    assert base_spec is not None and base_spec.loader is not None
    sys.modules["reward_sql_r1"] = base_module
    base_spec.loader.exec_module(base_module)

    spec = importlib.util.spec_from_file_location("reward_sql_r1_acr_module", reward_dir / "reward_sql_r1_acr.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.reward_sql_r1_acr


def _make_db(root: Path, db_id: str) -> None:
    db_dir = root / db_id
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{db_id}.sqlite"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    cur.executemany("INSERT INTO users (id, name) VALUES (?, ?)", [(1, "alice"), (2, "bob")])
    conn.commit()
    conn.close()


def _prompt_messages(question: str) -> list[dict[str, str]]:
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful SQL expert assistant.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>"
    )
    return [{"role": "user", "content": prompt}]


def test_reward_sql_r1_acr_match_without_leak(tmp_path: Path):
    reward_sql_r1_acr = _load_reward_sql_r1_acr()
    db_root = tmp_path / "dbs"
    _make_db(db_root, "toy")

    response = """
We need to count all rows in the users table and return a single aggregate value.
</think>
<answer>
```sql
SELECT COUNT(*) FROM users;
```
</answer>
"""
    result = reward_sql_r1_acr(
        data_source="synsql",
        solution_str=response,
        ground_truth={"db_id": "toy", "sql": "SELECT COUNT(*) FROM users;"},
        extra_info={"acr_orig_prompt": _prompt_messages("How many users are there?")},
        db_root=str(db_root),
        min_reasoning_chars=0,
    )

    assert result["acr_base_score"] == 1.0
    assert result["acr_is_correct"] is True
    assert result["acr_leak_hit"] is False
    assert result["score"] > 0.0


def test_reward_sql_r1_acr_banned_phrase_leak(tmp_path: Path):
    reward_sql_r1_acr = _load_reward_sql_r1_acr()
    db_root = tmp_path / "dbs"
    _make_db(db_root, "toy")

    response = """
Given SQL, I can simply use the provided query as the final answer.
</think>
<answer>
```sql
SELECT COUNT(*) FROM users;
```
</answer>
"""
    result = reward_sql_r1_acr(
        data_source="synsql",
        solution_str=response,
        ground_truth={"db_id": "toy", "sql": "SELECT COUNT(*) FROM users;"},
        extra_info={"acr_orig_prompt": _prompt_messages("How many users are there?")},
        db_root=str(db_root),
        min_reasoning_chars=0,
    )

    assert result["acr_base_score"] == 1.0
    assert result["acr_banned_phrase_hit"] is True
    assert result["acr_leak_hit"] is True
    assert result["score"] < 0.0


def test_reward_sql_r1_acr_query_copy_leak(tmp_path: Path):
    reward_sql_r1_acr = _load_reward_sql_r1_acr()
    db_root = tmp_path / "dbs"
    _make_db(db_root, "toy")

    response = """
I will use the exact query SELECT COUNT(*) FROM users because it directly answers the question.
</think>
<answer>
```sql
SELECT COUNT(*) FROM users;
```
</answer>
"""
    result = reward_sql_r1_acr(
        data_source="synsql",
        solution_str=response,
        ground_truth={"db_id": "toy", "sql": "SELECT COUNT(*) FROM users;"},
        extra_info={"acr_orig_prompt": _prompt_messages("How many users are there?")},
        db_root=str(db_root),
        min_reasoning_chars=0,
    )

    assert result["acr_base_score"] == 1.0
    assert result["acr_query_copy_hit"] is True
    assert result["acr_leak_hit"] is True

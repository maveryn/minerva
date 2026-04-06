from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_reward_fn():
    reward_path = Path(__file__).resolve().parents[3] / "verl" / "utils" / "reward_score" / "reward_sql_r1.py"
    spec = importlib.util.spec_from_file_location("reward_sql_r1_module", reward_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.reward_sql_r1


def _make_db(root: Path, db_id: str) -> Path:
    db_dir = root / db_id
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{db_id}.sqlite"

    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    cur.executemany("INSERT INTO users (id, name) VALUES (?, ?)", [(1, "alice"), (2, "bob")])
    conn.commit()
    conn.close()
    return db_path


def test_reward_sql_r1_response_only_match(tmp_path: Path):
    reward_sql_r1 = _load_reward_fn()
    db_root = tmp_path / "dbs"
    _make_db(db_root, "toy")

    response = """
count the rows in the users table.
</think>
<answer>
```sql
SELECT COUNT(*) FROM users;
```
</answer>
"""
    ground_truth = {"db_id": "toy", "sql": "SELECT COUNT(*) FROM users;"}

    score = reward_sql_r1(
        data_source="synsql",
        solution_str=response,
        ground_truth=ground_truth,
        return_dict=True,
        db_root=str(db_root),
    )

    assert score["format_correct"] is True
    assert score["exec_status"] == "Match"
    assert score["exec_match"] == 1.0
    assert score["score"] > 0.0


def test_reward_sql_r1_response_only_mismatch(tmp_path: Path):
    reward_sql_r1 = _load_reward_fn()
    db_root = tmp_path / "dbs"
    _make_db(db_root, "toy")

    response = """
count only one row.
</think>
<answer>
```sql
SELECT COUNT(*) FROM users WHERE id = 1;
```
</answer>
"""
    ground_truth = {"db_id": "toy", "sql": "SELECT COUNT(*) FROM users;"}

    score = reward_sql_r1(
        data_source="synsql",
        solution_str=response,
        ground_truth=ground_truth,
        return_dict=True,
        db_root=str(db_root),
    )

    assert score["format_correct"] is True
    assert score["exec_status"] == "Mismatch"
    assert score["exec_success"] == 1.0
    assert score["exec_match"] == 0.0


def test_reward_sql_r1_normalized_db_lookup(tmp_path: Path):
    reward_sql_r1 = _load_reward_fn()
    db_root = tmp_path / "dbs"
    _make_db(db_root, "pokemon_information_and_evolution_tracking")

    response = """
reason about pokemon info.
</think>
<answer>
```sql
SELECT COUNT(*) FROM users;
```
</answer>
"""
    ground_truth = {
        "db_id": "pokémon_information_and_evolution_tracking",
        "sql": "SELECT COUNT(*) FROM users;",
    }

    score = reward_sql_r1(
        data_source="synsql",
        solution_str=response,
        ground_truth=ground_truth,
        return_dict=True,
        db_root=str(db_root),
    )

    assert score["exec_status"] == "Match"
    assert score["exec_match"] == 1.0

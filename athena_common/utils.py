"""Shared utility helpers for Athena."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv
from datetime import datetime, timezone


def load_yaml(path: str) -> Dict:
    """Return the YAML content of *path* as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> List[Dict]:
    """Return a list of JSON objects loaded from a JSONL file."""
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_api_key(var_name: str) -> str:
    """Retrieve an API key from environment variables or a ``.env`` file.

    Parameters
    ----------
    var_name:
        Name of the environment variable containing the API key.

    Returns
    -------
    str
        The API key string.

    Raises
    ------
    ValueError
        If the variable is not set in the environment or ``.env`` file.
    """
    load_dotenv()
    key = os.getenv(var_name)
    if not key:
        raise ValueError(f"{var_name} not found in environment or .env file")
    return key


def parse_date(s: Optional[str]) -> Optional[datetime]:
    """Parse *s* into a :class:`~datetime.datetime` if possible."""

    if not s:
        return None
    s_str = str(s).strip()

    def _fix_z(val: str) -> str:
        return val.replace("Z", "+00:00") if val.endswith("Z") else val

    try:
        return datetime.fromisoformat(_fix_z(s_str))
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s_str, fmt)
        except Exception:
            continue
    return None


def within_inclusive(ts: str, start_dt: datetime, end_dt: datetime) -> bool:
    """Return ``True`` if ``ts`` is within the inclusive range."""

    d = parse_date(ts)
    if not d:
        return False
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return start_dt <= d <= end_dt

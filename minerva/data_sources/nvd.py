import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from minerva.logger import get_logger
from minerva.utils import parse_date, within_inclusive


NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def _is_nvd_source(src: str) -> bool:
    s = (src or "").lower()
    return ("nist" in s) or ("nvd" in s)


def _pick_better(existing: Optional[Dict[str, Any]], candidate: Dict[str, Any]) -> Dict[str, Any]:
    if existing is None:
        return candidate
    if (existing.get("type", "").lower() != "primary") and (candidate.get("type", "").lower() == "primary"):
        return candidate
    return existing


def _extract_version_metrics(metric_list: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    best: Dict[str, Optional[Dict[str, Any]]] = {"nvd": None, "cna": None}
    for m in metric_list or []:
        src = m.get("source", "")
        entry = {
            "type": m.get("type", ""),
            "source": src,
            "vector": m.get("cvssData", {}).get("vectorString", "") or "",
            "score": m.get("cvssData", {}).get("baseScore"),
            "severity": m.get("cvssData", {}).get("baseSeverity", "") or "",
        }
        key = "nvd" if _is_nvd_source(src) else "cna"
        best[key] = _pick_better(best[key], entry)

    def _simple(e: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not e:
            return {"vector": "", "score": None, "severity": ""}
        return {"vector": e["vector"], "score": e["score"], "severity": e["severity"]}

    return _simple(best["nvd"]), _simple(best["cna"])


def _english_description(cve_obj: Dict[str, Any]) -> str:
    for d in cve_obj.get("descriptions", []):
        if d.get("lang") == "en":
            return d.get("value", "")
    if cve_obj.get("descriptions"):
        return cve_obj["descriptions"][0].get("value", "")
    return ""


def _normalize_cwes(cve_obj: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for w in cve_obj.get("weaknesses", []):
        for d in w.get("description", []):
            val = d.get("value")
            if val:
                out.append(str(val))
    return out


def parse_cve_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cve = item.get("cve", {})
    cve_id = cve.get("id")
    if not cve_id:
        return None

    metrics = cve.get("metrics", {}) or {}
    nvd_v31, cna_v31 = _extract_version_metrics(metrics.get("cvssMetricV31") or [])
    nvd_v40, cna_v40 = _extract_version_metrics(metrics.get("cvssMetricV40") or [])

    return {
        "cve_id": cve_id,
        "published_date": cve.get("published"),
        "last_modified_date": cve.get("lastModified"),
        "description": _english_description(cve),
        "cwe_ids": _normalize_cwes(cve),
        "nvd_cvss_v31_vector": nvd_v31["vector"],
        "nvd_cvss_v31_score": nvd_v31["score"],
        "nvd_cvss_v31_severity": nvd_v31["severity"],
        "cna_cvss_v31_vector": cna_v31["vector"],
        "cna_cvss_v31_score": cna_v31["score"],
        "cna_cvss_v31_severity": cna_v31["severity"],
        "nvd_cvss_v4_vector": nvd_v40["vector"],
        "nvd_cvss_v4_score": nvd_v40["score"],
        "nvd_cvss_v4_severity": nvd_v40["severity"],
        "cna_cvss_v4_vector": cna_v40["vector"],
        "cna_cvss_v4_score": cna_v40["score"],
        "cna_cvss_v4_severity": cna_v40["severity"],
    }


def fetch_cves(start_dt: datetime, end_dt: datetime, api_key: Optional[str], delay_s: float, logger) -> List[Dict[str, Any]]:
    params = {
        "pubStartDate": start_dt.isoformat() + "Z",
        "pubEndDate": end_dt.isoformat() + "Z",
        "resultsPerPage": 2000,
        "startIndex": 0,
    }
    headers = {"apiKey": api_key} if api_key else {}
    results: List[Dict[str, Any]] = []

    while True:
        resp = requests.get(NVD_URL, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities", []) or []
        results.extend(vulns)

        total = data.get("totalResults", 0)
        got = len(vulns)
        if logger:
            logger.info("NVD page startIndex=%s got=%s/%s", params["startIndex"], got, total)

        if params["startIndex"] + got >= total:
            break
        params["startIndex"] += got
        time.sleep(delay_s)

    return results


def write_jsonl(records: List[Dict[str, Any]], outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_cached(nvd_dir: Path, start: datetime, end: datetime, logger) -> List[Dict[str, Any]]:
    paths = sorted(nvd_dir.glob("*.jsonl"))
    collected: List[Dict[str, Any]] = []
    for p in paths:
        for row in load_jsonl(p):
            if within_inclusive(row.get("published_date"), start, end):
                collected.append(row)
    if logger:
        logger.info("Loaded %d cached NVD rows from %d files", len(collected), len(paths))
    return collected


def ensure_records(cfg: Dict[str, Any], logger=None) -> List[Dict[str, Any]]:
    """
    Fetch CVE records for the configured window. Will read existing JSONL
    files under nvd_data_dir; if insufficient, fetches from NVD and saves.
    """
    common = cfg.get("COMMON", {})
    nvd_cfg = cfg.get("NVD", common) or {}
    start = parse_date(nvd_cfg.get("start_date") or common.get("nvd_start_date"))
    end = parse_date(nvd_cfg.get("end_date") or common.get("nvd_end_date"))
    if start is None or end is None:
        raise ValueError("Start and end dates must be configured for NVD fetch.")
    delay_s = float(os.getenv("NVD_DELAY_S", nvd_cfg.get("delay_s", 1.0)))
    api_key = os.getenv("NVD_API_KEY")
    nvd_dir = Path(nvd_cfg.get("data_dir") or common.get("nvd_data_dir") or "data/processed/nvd")
    nvd_dir.mkdir(parents=True, exist_ok=True)

    cached = load_cached(nvd_dir, start, end, logger)
    if cached:
        return cached

    if logger is None:
        logger = get_logger("nvd")
    day = start
    records: List[Dict[str, Any]] = []
    while day <= end:
        next_day = day + timedelta(days=1)
        outfile = nvd_dir / f"nvd_{day.date()}.jsonl"
        if outfile.exists():
            if logger:
                logger.info("Using cached %s", outfile)
            day = next_day
            continue
        try:
            raw_items = fetch_cves(day, next_day, api_key, delay_s, logger)
            parsed = [r for r in (parse_cve_item(it) for it in raw_items) if r]
            if parsed:
                write_jsonl(parsed, outfile)
            records.extend(parsed)
        except Exception as exc:
            if logger:
                logger.error("Failed to fetch %s: %s", day.date(), exc)
        time.sleep(delay_s)
        day = next_day

    if not records and cached:
        return cached
    return records


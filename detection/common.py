import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    tomllib = None  # type: ignore


TECHNIQUE_ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)


def load_yaml(path: Path) -> Dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_toml(path: Path) -> Dict:
    if tomllib is None:
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_repo_slug(repo_or_url: str) -> str:
    if "github.com" in repo_or_url:
        parts = repo_or_url.strip("/").split("/")
        if len(parts) >= 2:
            return "/".join(parts[-2:])
    return repo_or_url.strip("/")


def _download_zip(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    dest_path.write_bytes(data)


def download_github_repo(
    repo_or_url: str,
    cache_dir: Path,
    ref: Optional[str] = None,
    force: bool = False,
) -> Path:
    repo = parse_repo_slug(repo_or_url)
    refs = [ref] if ref else ["main", "master"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    for candidate_ref in refs:
        if not candidate_ref:
            continue
        repo_slug = repo.replace("/", "_")
        dest = cache_dir / f"{repo_slug}-{candidate_ref}"
        marker = dest / ".source.json"
        if dest.exists() and marker.exists() and not force:
            return dest

        zip_url = f"https://github.com/{repo}/archive/refs/heads/{candidate_ref}.zip"
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                zip_path = tmp_dir_path / "repo.zip"
                _download_zip(zip_url, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp_dir_path)
                extracted = next(tmp_dir_path.iterdir(), None)
                if extracted is None:
                    continue
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(extracted), dest)
                marker.write_text(
                    json.dumps({"repo": repo, "ref": candidate_ref}, ensure_ascii=True),
                    encoding="utf-8",
                )
                return dest
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue

    raise RuntimeError(f"Failed to download repo {repo} (refs tried: {refs})")


def iter_files(root: Path, suffixes: Sequence[str]) -> Iterable[Path]:
    for suffix in suffixes:
        for path in root.rglob(f"*{suffix}"):
            if path.is_file():
                yield path


def technique_id_name_map(mitre_path: Path) -> Dict[str, str]:
    if not mitre_path.exists():
        return {}
    try:
        payload = json.loads(mitre_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    mapping: Dict[str, str] = {}
    for obj in payload.get("objects", []) if isinstance(payload, dict) else []:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        name = obj.get("name")
        if not name:
            continue
        for ref in obj.get("external_references", []) or []:
            if ref.get("source_name") == "mitre-attack":
                tid = ref.get("external_id")
                if tid and isinstance(tid, str):
                    mapping[tid.upper()] = str(name)
                break
    return mapping


def normalize_technique_ids(raw_ids: Iterable[str]) -> List[str]:
    out: List[str] = []
    for raw in raw_ids or []:
        if not raw:
            continue
        for match in TECHNIQUE_ID_RE.findall(str(raw)):
            tid = match.upper()
            if tid not in out:
                out.append(tid)
    return out


def strip_technique_ids(text: str) -> str:
    return TECHNIQUE_ID_RE.sub("", text or "")


def strip_technique_names(text: str, names: Iterable[str]) -> str:
    cleaned = text or ""
    for name in names or []:
        if not name:
            continue
        cleaned = re.sub(re.escape(str(name)), "", cleaned, flags=re.IGNORECASE)
    return cleaned


def clean_snippet(text: str, technique_names: Iterable[str]) -> str:
    cleaned = strip_technique_ids(text)
    cleaned = strip_technique_names(cleaned, technique_names)
    lines = []
    for line in (cleaned or "").splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def split_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,\s]+", value.strip())
        return [p for p in parts if p]
    return []


def build_requirements(count: int, allow_subtechnique: bool = True) -> str:
    if count <= 1:
        lines = [
            "- Use MITRE ATT&CK Enterprise technique IDs only.",
            "- Return exactly ONE technique ID.",
        ]
        if allow_subtechnique:
            lines.append(
                "- Prefer a sub-technique ID (e.g., T1059.003) if it is more suitable; otherwise return the parent technique ID (e.g., T1059)."
            )
        else:
            lines.append("- Do not return sub-technique ID.")
    else:
        lines = [
            "- Use MITRE ATT&CK Enterprise technique IDs only.",
            f"- Return EXACTLY {count} technique ID(s).",
        ]
    return "\n".join(lines)


def token_set(text: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


def select_distractors(
    id_to_name: Dict[str, str],
    gold_ids: List[str],
    k: int,
    rng,
) -> List[str]:
    if k <= 0:
        return []
    gold_tokens = set()
    for tid in gold_ids:
        name = id_to_name.get(tid, "")
        gold_tokens |= token_set(name)
    scored: List[Tuple[float, str]] = []
    for tid, name in id_to_name.items():
        if tid in gold_ids:
            continue
        tokens = token_set(name)
        overlap = len(tokens & gold_tokens)
        if overlap > 0:
            scored.append((float(overlap), tid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    candidates = [tid for _, tid in scored[: max(k * 3, 10)]]
    if len(candidates) < k:
        extras = [tid for tid in id_to_name.keys() if tid not in gold_ids and tid not in candidates]
        rng.shuffle(extras)
        candidates.extend(extras[: (k - len(candidates))])
    rng.shuffle(candidates)
    return candidates[:k]


def format_candidate_block(
    ids: List[str],
    id_to_name: Optional[Dict[str, str]] = None,
    include_names: bool = True,
    count: int = 1,
) -> str:
    lines = [f"Candidate techniques (choose EXACTLY {count}):"]
    for idx, tid in enumerate(ids, 1):
        name = ""
        if include_names and id_to_name:
            name = id_to_name.get(tid, "")
        suffix = f" - {name}" if name else ""
        lines.append(f"{idx}) {tid}{suffix}")
    return "\n".join(lines)

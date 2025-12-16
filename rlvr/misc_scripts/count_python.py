#!/usr/bin/env python3
import re, json, sys, gzip, ast, warnings
from pathlib import Path
from typing import Optional, Tuple

PY_TAGS = {"python", "py", "ipython", "python3"}
# ```lang\n ... ```
FENCE_RE = re.compile(r"```([A-Za-z0-9_+\-]*)\s*\n(.*?)```", re.DOTALL | re.MULTILINE)
# also catch ~~~ fences (rare)
TILDE_RE = re.compile(r"~~~([A-Za-z0-9_+\-]*)\s*\n(.*?)~~~", re.DOTALL | re.MULTILINE)

# quick heuristic to avoid parsing obvious non-Python like our LIS markers
NON_PY_HINT_RE = re.compile(r"(\\ids\{|\\answer\{)")
PY_HINT_RE = re.compile(r"\b(def |class |import |from |for .* in |while |try:|except|with |print\()")

def lines(p):
    p = Path(p)
    openf = gzip.open if p.suffix == ".gz" else open
    with openf(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def extract_reply(obj: dict) -> Optional[str]:
    r = obj.get("model_reply")
    if isinstance(r, str) and r.strip():
        return r
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs:
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                return m["content"]
    return None

def extract_code_blocks(text: str):
    text = text or ""
    return FENCE_RE.findall(text) + TILDE_RE.findall(text)

def looks_like_python(code: str) -> bool:
    if NON_PY_HINT_RE.search(code):
        return False
    # light heuristic before parsing to avoid obvious non-Python blocks
    if not PY_HINT_RE.search(code) and ";" not in code and "=" not in code and "()" not in code:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)  # <-- suppress your '\i' warnings
            ast.parse(code)
        return True
    except Exception:
        return False

def has_python_code(reply: str) -> Tuple[bool, bool]:
    labeled = False
    inferred = False
    for lang, code in extract_code_blocks(reply or ""):
        lang = (lang or "").lower()
        if lang in PY_TAGS:
            labeled = True
        elif not lang and looks_like_python(code):
            inferred = True
    # fallback for dangling ```python without closing ```
    if not labeled and re.search(r"```(?:python|py|ipython|python3)\b", reply or "", re.IGNORECASE):
        labeled = True
    return (labeled or inferred), labeled

def open_writer(path: Optional[str]):
    if not path:
        return None
    return gzip.open(path, "wt", encoding="utf-8") if path.endswith(".gz") else open(path, "w", encoding="utf-8")

def main(in_path: str, out_path: Optional[str] = None):
    total = any_python = labeled_python = missing_reply = bad = 0
    wf = open_writer(out_path)
    try:
        for ln in lines(in_path):
            total += 1
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                bad += 1
                continue
            reply = extract_reply(obj)
            if not reply:
                missing_reply += 1
                continue
            any_py, labeled_py = has_python_code(reply)
            if any_py:
                any_python += 1
                if labeled_py:
                    labeled_python += 1
                if wf:
                    wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        if wf:
            wf.close()

    print(f"file: {in_path}")
    if out_path:
        print(f"saved rows-with-python to: {out_path}")
    print(f"total rows: {total}")
    print(f"rows with python code (any): {any_python}")
    print(f"rows with fenced '```python' code: {labeled_python}")
    pct = (100.0 * any_python / total) if total else 0.0
    print(f"fraction with python code: {any_python}/{total} = {pct:.4f}%")
    if missing_reply:
        print(f"(rows missing a reply: {missing_reply})")
    if bad:
        print(f"(malformed JSON lines skipped: {bad})")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python count_python.py INPUT.jsonl[.gz] [OUTPUT.jsonl]")
        sys.exit(2)
    in_path = args[0]
    out_path = args[1] if len(args) > 1 else None
    main(in_path, out_path)

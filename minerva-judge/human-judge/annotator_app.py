#!/usr/bin/env python3
"""Dependency-free local web app for pairwise and pointwise human annotation."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse


ROOT = Path(__file__).resolve().parent
ANNOTATIONS_DIR = ROOT / "annotations"
VALID_LABELS = ("A", "TIE", "B")
POINTWISE_SCORE_FIELDS = (
    (
        "writing_quality_score",
        "Writing Quality",
        "Is the response easy to read and efficiently structured?",
        (
            ("1", "Poor", "Provides no rationale at all, or is hard to follow because of rambling, disorganization, or awkward phrasing."),
            ("2", "Fair", "Understandable, but noticeably wordy, repetitive, or clunky."),
            ("3", "Good", "Clear and easy to follow, but still somewhat wordy, uneven, or less efficiently structured."),
            ("4", "Excellent", "Very clear, well-structured, concise, and easy to scan, with no noticeable writing issues."),
        ),
    ),
    (
        "evidence_use_score",
        "Evidence Use",
        "Does the response justify its answer using details from the prompt?",
        (
            ("1", "Poor", "Answer is mostly asserted, with little or no real prompt-based support."),
            ("2", "Fair", "Some prompt evidence is used, but the justification is weak, generic, or incomplete."),
            ("3", "Good", "The main answer is supported by prompt details, with only minor gaps."),
            ("4", "Excellent", "The answer is directly justified using strong, specific prompt evidence."),
        ),
    ),
    (
        "cti_concept_focus_score",
        "CTI Concept Use",
        "Does the response use the right CTI concepts correctly to support its answer?",
        (
            ("1", "Poor", "Uses no relevant CTI concepts, or uses CTI concepts not relevant to the question."),
            ("2", "Fair", "Uses some relevant CTI concepts, but with important mistakes or vague distinctions."),
            ("3", "Good", "Mostly uses the right CTI concepts correctly, with minor imprecision."),
            ("4", "Excellent", "Uses the right CTI concepts precisely and consistently to support the answer."),
        ),
    ),
)


def available_subsets() -> list[str]:
    names: list[str] = []
    for path in sorted(ROOT.iterdir()):
        if not path.is_dir():
            continue
        if (path / "blind_master.jsonl").exists() or (path / "blind_responses.jsonl").exists():
            names.append(path.name)
    return names


def subset_dir(subset: str) -> Path:
    path = ROOT / subset
    if subset not in available_subsets() or not path.is_dir():
        raise ValueError(f"Unknown subset: {subset}")
    return path


def subset_mode(subset: str) -> str:
    path = subset_dir(subset)
    if (path / "blind_master.jsonl").exists():
        return "pairwise"
    if (path / "blind_responses.jsonl").exists():
        return "pointwise"
    raise ValueError(f"Subset has no supported blind file: {subset}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_items(subset: str) -> list[dict[str, Any]]:
    filename = "blind_master.jsonl" if subset_mode(subset) == "pairwise" else "blind_responses.jsonl"
    return load_jsonl(subset_dir(subset) / filename)


def normalize_annotator(annotator: str) -> str:
    cleaned = annotator.strip()
    if not cleaned:
        raise ValueError("Annotator name is required.")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("Annotator name is invalid.")
    return cleaned[:80]


def normalize_label(label: str) -> str:
    cleaned = label.strip().upper()
    if cleaned not in VALID_LABELS:
        raise ValueError(f"Invalid label: {label}")
    return cleaned


def normalize_pointwise_score(score: str) -> int:
    cleaned = score.strip()
    if cleaned not in {"1", "2", "3", "4"}:
        raise ValueError(f"Invalid score: {score}")
    return int(cleaned)


def annotation_path(subset: str, annotator: str) -> Path:
    return ANNOTATIONS_DIR / subset / f"{annotator}.jsonl"


def load_annotations(subset: str, annotator: str) -> dict[str, dict[str, Any]]:
    path = annotation_path(subset, annotator)
    if not path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        item_id = str(row.get("item_id") or "")
        if item_id:
            records[item_id] = row
    return records


def save_annotations(
    subset: str,
    annotator: str,
    items: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> None:
    path = annotation_path(subset, annotator)
    path.parent.mkdir(parents=True, exist_ok=True)
    item_order = {item["item_id"]: index for index, item in enumerate(items)}
    ordered = sorted(records.values(), key=lambda row: item_order.get(row["item_id"], 10**9))
    with path.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def next_unlabeled_item(
    items: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for item in items:
        if item["item_id"] not in records:
            return item
    return None


def page_shell(title: str, body: str) -> str:
    css = """
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #f4efe3 0%, #efe5d4 100%);
      color: #1f1b16;
    }
    .page {
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }
    .card {
      background: rgba(255, 251, 244, 0.96);
      border: 1px solid #d8c9b2;
      border-radius: 18px;
      box-shadow: 0 18px 48px rgba(82, 53, 16, 0.10);
      padding: 22px 24px;
      margin-bottom: 18px;
    }
    .meta {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      font-size: 15px;
      color: #6a5842;
      margin-bottom: 10px;
    }
    .prompt {
      white-space: pre-wrap;
      line-height: 1.55;
      font-size: 17px;
    }
    .responses {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    .review-grid {
      display: grid;
      grid-template-columns: minmax(280px, 0.95fr) minmax(0, 1.25fr);
      gap: 18px;
      align-items: start;
    }
    .response-box {
      border: 1px solid #d8c9b2;
      border-radius: 16px;
      background: #fffaf0;
      min-height: 280px;
      padding: 18px;
    }
    .rubric-box {
      border: 1px solid #d8c9b2;
      border-radius: 16px;
      background: rgba(255, 250, 240, 0.98);
      padding: 18px;
      position: sticky;
      top: 18px;
    }
    .response-title {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #7f5b24;
      margin-bottom: 12px;
    }
    .response-text {
      white-space: pre-wrap;
      line-height: 1.55;
      font-size: 16px;
    }
    .rubric-summary {
      color: #6a5842;
      font-size: 14px;
      line-height: 1.5;
      margin-bottom: 12px;
    }
    .rubric-item {
      border-top: 1px solid #e2d2bc;
      padding-top: 14px;
      margin-top: 14px;
    }
    .rubric-item:first-of-type {
      border-top: none;
      margin-top: 0;
      padding-top: 0;
    }
    .rubric-title {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .panel-title {
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #7f5b24;
      margin-bottom: 12px;
    }
    .score-options {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .score-option {
      display: block;
      border: 1px solid #d8c9b2;
      border-radius: 12px;
      background: #fffdf7;
      padding: 10px 10px 12px;
      cursor: pointer;
      min-height: 94px;
    }
    .score-option input {
      margin-right: 8px;
    }
    .score-number {
      font-size: 16px;
      font-weight: 700;
      color: #224b3f;
    }
    .score-label {
      display: block;
      font-size: 13px;
      color: #8e6a32;
      margin-top: 4px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .score-anchor {
      display: block;
      font-size: 13px;
      line-height: 1.4;
      color: #5f4d39;
      margin-top: 8px;
    }
    .notes {
      width: 100%;
      min-height: 88px;
      border: 1px solid #cdbba1;
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
      background: #fffdf7;
      box-sizing: border-box;
      resize: vertical;
    }
    .actions {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      justify-content: center;
    }
    .actions button, .actions a, .start-form button {
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 14px 20px;
      background: #224b3f;
      color: #fffaf0;
      font-size: 16px;
      cursor: pointer;
      text-decoration: none;
    }
    .actions button.secondary {
      background: #8e6a32;
    }
    .actions button.tertiary {
      background: #6f3d2d;
    }
    .start-form {
      display: grid;
      gap: 14px;
      max-width: 520px;
    }
    .start-form input, .start-form select {
      border: 1px solid #cdbba1;
      border-radius: 12px;
      padding: 12px 14px;
      font-size: 16px;
      background: #fffdf7;
    }
    .helper {
      color: #6a5842;
      font-size: 15px;
      line-height: 1.5;
    }
    .helper-tight {
      color: #6a5842;
      font-size: 14px;
      line-height: 1.45;
    }
    .error {
      color: #6f1d1b;
      font-weight: 700;
    }
    code {
      background: rgba(34, 75, 63, 0.08);
      padding: 2px 6px;
      border-radius: 6px;
    }
    @media (max-width: 900px) {
      .responses {
        grid-template-columns: 1fr;
      }
      .review-grid {
        grid-template-columns: 1fr;
      }
      .rubric-box {
        position: static;
      }
      .score-options {
        grid-template-columns: 1fr 1fr;
      }
    }
    """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="page">
    {body}
  </div>
</body>
</html>"""


def render_home(error: str | None = None) -> str:
    subsets = available_subsets()
    error_block = f'<p class="helper error">{html.escape(error)}</p>' if error else ""
    if len(subsets) == 1:
        subset_field = f"""
        <label>
          <div>Dataset</div>
          <input value="{html.escape(subsets[0])}" readonly>
          <input type="hidden" name="subset" value="{html.escape(subsets[0])}">
        </label>
        """
    else:
        options = "\n".join(
            f'<option value="{html.escape(subset)}">{html.escape(subset)}</option>'
            for subset in subsets
        )
        subset_field = f"""
        <label>
          <div>Dataset</div>
          <select name="subset">{options}</select>
        </label>
        """
    body = f"""
    <div class="card">
      <h1>Human Judge</h1>
      <p class="helper">
        Pick a dataset and enter your annotator ID. Pairwise datasets show
        Response A vs Response B. Pointwise datasets show one blind response
        with the 3-criterion rubric. Progress is saved locally under
        <code>annotations/&lt;subset&gt;/&lt;annotator&gt;.jsonl</code>.
      </p>
      {error_block}
      <form class="start-form" action="/" method="get">
        {subset_field}
        <label>
          <div>Annotator ID</div>
          <input name="annotator" placeholder="annotator_1" required>
        </label>
        <button type="submit">Start Annotating</button>
      </form>
    </div>
    """
    return page_shell("Human Judge", body)


def render_done(subset: str, annotator: str, completed: int) -> str:
    subset_q = quote_plus(subset)
    annotator_q = quote_plus(annotator)
    body = f"""
    <div class="card">
      <h1>Annotation Complete</h1>
      <p class="helper">
        Annotator <strong>{html.escape(annotator)}</strong> finished
        <strong>{completed}</strong> items in <strong>{html.escape(subset)}</strong>.
      </p>
      <div class="actions">
        <a href="/?subset={subset_q}&annotator={annotator_q}">Refresh Progress</a>
        <a href="/">Back to Start</a>
      </div>
    </div>
    """
    return page_shell("Annotation Complete", body)


def render_pairwise_item_page(
    subset: str,
    annotator: str,
    item: dict[str, Any],
    completed: int,
    total: int,
) -> str:
    subset_q = quote_plus(subset)
    annotator_q = quote_plus(annotator)
    body = f"""
    <div class="card">
      <div class="meta">
        <div><strong>Dataset:</strong> {html.escape(subset)}</div>
        <div><strong>Annotator:</strong> {html.escape(annotator)}</div>
        <div><strong>Progress:</strong> {completed} / {total}</div>
        <div><strong>Item:</strong> {html.escape(str(item["item_id"]))}</div>
      </div>
      <h2>Prompt</h2>
      <div class="prompt">{html.escape(str(item["prompt"]))}</div>
    </div>
    <div class="responses">
      <div class="response-box">
        <div class="response-title">Response A</div>
        <div class="response-text">{html.escape(str(item["response_a"]))}</div>
      </div>
      <div class="response-box">
        <div class="response-title">Response B</div>
        <div class="response-text">{html.escape(str(item["response_b"]))}</div>
      </div>
    </div>
    <div class="card">
      <form action="/submit" method="post">
        <input type="hidden" name="subset" value="{html.escape(subset)}">
        <input type="hidden" name="annotator" value="{html.escape(annotator)}">
        <input type="hidden" name="item_id" value="{html.escape(str(item["item_id"]))}">
        <div class="actions">
          <button type="submit" name="label" value="A">Response A Better</button>
          <button type="submit" name="label" value="TIE" class="secondary">Tie / Same Quality</button>
          <button type="submit" name="label" value="B" class="tertiary">Response B Better</button>
        </div>
      </form>
    </div>
    <div class="card">
      <p class="helper">
        Use the same URL to resume later:
        <code>/?subset={html.escape(subset_q)}&amp;annotator={html.escape(annotator_q)}</code>
      </p>
    </div>
    """
    return page_shell(f"{subset} - {annotator}", body)


def render_pointwise_item_page(
    subset: str,
    annotator: str,
    item: dict[str, Any],
    completed: int,
    total: int,
) -> str:
    subset_q = quote_plus(subset)
    annotator_q = quote_plus(annotator)
    rubric_sections = []
    for index, (field_name, title, question, anchors) in enumerate(POINTWISE_SCORE_FIELDS, start=1):
        options_html = "\n".join(
            f"""
            <label class="score-option">
              <input type="radio" name="{html.escape(field_name)}" value="{score}" required>
              <span class="score-number">{score}</span>
              <span class="score-label">{html.escape(label)}</span>
              <span class="score-anchor">{html.escape(anchor)}</span>
            </label>
            """
            for score, label, anchor in anchors
        )
        rubric_sections.append(
            f"""
            <div class="rubric-item">
              <div class="rubric-title">{index}. {html.escape(title)}</div>
              <div class="helper-tight">{html.escape(question)}</div>
              <div class="score-options">{options_html}</div>
            </div>
            """
        )

    body = f"""
    <div class="card">
      <div class="meta">
        <div><strong>Dataset:</strong> {html.escape(subset)}</div>
        <div><strong>Annotator:</strong> {html.escape(annotator)}</div>
        <div><strong>Progress:</strong> {completed} / {total}</div>
        <div><strong>Item:</strong> {html.escape(str(item["item_id"]))}</div>
        <div><strong>Task:</strong> {html.escape(str(item.get("task") or ""))}</div>
      </div>
      <h2>Prompt</h2>
      <div class="prompt">{html.escape(str(item["prompt"]))}</div>
    </div>
    <div class="review-grid">
      <div class="response-box">
        <div class="response-title">Model Response</div>
        <div class="response-text">{html.escape(str(item["response"]))}</div>
      </div>
      <div class="rubric-box">
        <div class="panel-title">Rubric</div>
        <form action="/submit-pointwise" method="post">
          <input type="hidden" name="subset" value="{html.escape(subset)}">
          <input type="hidden" name="annotator" value="{html.escape(annotator)}">
          <input type="hidden" name="item_id" value="{html.escape(str(item["item_id"]))}">
          {''.join(rubric_sections)}
          <div class="actions" style="margin-top: 18px;">
            <button type="submit">Save Scores And Next</button>
          </div>
        </form>
      </div>
    </div>
    """
    return page_shell(f"{subset} - {annotator}", body)


def render_for_request(subset: str | None, annotator: str | None) -> str:
    if subset is None or annotator is None:
        return render_home()

    normalized_annotator = normalize_annotator(annotator)
    items = load_items(subset)
    records = load_annotations(subset, normalized_annotator)
    current_item = next_unlabeled_item(items, records)
    if current_item is None:
        return render_done(subset, normalized_annotator, len(records))

    if subset_mode(subset) == "pairwise":
        return render_pairwise_item_page(
            subset=subset,
            annotator=normalized_annotator,
            item=current_item,
            completed=len(records),
            total=len(items),
        )
    return render_pointwise_item_page(
        subset=subset,
        annotator=normalized_annotator,
        item=current_item,
        completed=len(records),
        total=len(items),
    )


class AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "HumanJudgeHTTP/2.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        params = parse_qs(parsed.query)
        subset = params.get("subset", [None])[0]
        annotator = params.get("annotator", [None])[0]
        try:
            body = render_for_request(subset, annotator)
            self._send_html(HTTPStatus.OK, body)
        except ValueError as exc:
            self._send_html(HTTPStatus.BAD_REQUEST, render_home(str(exc)))

    def do_POST(self) -> None:
        if self.path not in {"/submit", "/submit-pointwise"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        params = parse_qs(body)

        try:
            subset = params["subset"][0]
            annotator = normalize_annotator(params["annotator"][0])
            item_id = params["item_id"][0]
            items = load_items(subset)
            item_lookup = {item["item_id"]: item for item in items}
            if item_id not in item_lookup:
                raise ValueError(f"Unknown item_id: {item_id}")

            records = load_annotations(subset, annotator)
            mode = subset_mode(subset)
            if self.path == "/submit":
                if mode != "pairwise":
                    raise ValueError(f"Subset is not pairwise: {subset}")
                label = normalize_label(params["label"][0])
                records[item_id] = {
                    "item_id": item_id,
                    "label": label,
                    "subset": subset,
                    "annotator": annotator,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }
            else:
                if mode != "pointwise":
                    raise ValueError(f"Subset is not pointwise: {subset}")
                writing_quality_score = normalize_pointwise_score(params["writing_quality_score"][0])
                evidence_use_score = normalize_pointwise_score(params["evidence_use_score"][0])
                cti_concept_focus_score = normalize_pointwise_score(params["cti_concept_focus_score"][0])
                records[item_id] = {
                    "item_id": item_id,
                    "subset": subset,
                    "annotator": annotator,
                    "writing_quality_score": writing_quality_score,
                    "evidence_use_score": evidence_use_score,
                    "cti_concept_focus_score": cti_concept_focus_score,
                    "total_score": writing_quality_score + evidence_use_score + cti_concept_focus_score,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }

            save_annotations(subset, annotator, items, records)
            location = f"/?subset={quote_plus(subset)}&annotator={quote_plus(annotator)}"
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", location)
            self.end_headers()
        except (KeyError, ValueError) as exc:
            self._send_html(HTTPStatus.BAD_REQUEST, render_home(str(exc)))

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send_html(self, status: HTTPStatus, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    print(f"Serving Human Judge at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

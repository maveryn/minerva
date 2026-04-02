#!/usr/bin/env python3
"""Simple local web app for pairwise and pointwise human annotation."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse


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

app = FastAPI(title="Human Judge")


def available_subsets() -> list[str]:
    names: list[str] = []
    for path in sorted(ROOT.iterdir()):
        if not path.is_dir():
            continue
        if (path / "blind_master.jsonl").exists() or (path / "blind_responses.jsonl").exists():
            names.append(path.name)
    return names


def subset_dir(subset: str) -> Path:
    if subset not in available_subsets():
        raise HTTPException(status_code=404, detail=f"Unknown subset: {subset}")
    return ROOT / subset


def subset_mode(subset: str) -> str:
    path = subset_dir(subset)
    if (path / "blind_master.jsonl").exists():
        return "pairwise"
    if (path / "blind_responses.jsonl").exists():
        return "pointwise"
    raise HTTPException(status_code=500, detail=f"Subset has no supported blind file: {subset}")


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
    path = subset_dir(subset) / (
        "blind_master.jsonl" if subset_mode(subset) == "pairwise" else "blind_responses.jsonl"
    )
    return load_jsonl(path)


def normalize_annotator(annotator: str) -> str:
    cleaned = annotator.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Annotator name is required.")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Annotator name is invalid.")
    return cleaned[:80]


def normalize_label(label: str) -> str:
    cleaned = label.strip().upper()
    if cleaned in VALID_LABELS:
        return cleaned
    raise HTTPException(status_code=400, detail=f"Invalid label: {label}")


def normalize_pointwise_score(score: str) -> int:
    cleaned = score.strip()
    if cleaned not in {"1", "2", "3", "4"}:
        raise HTTPException(status_code=400, detail=f"Invalid score: {score}")
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


def page_shell(title: str, body: str) -> HTMLResponse:
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
      background: rgba(255, 251, 244, 0.95);
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
    .rubric-anchors {
      display: grid;
      gap: 8px;
      margin: 10px 0 12px;
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
    page = f"""<!doctype html>
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
    return HTMLResponse(page)


def render_home(subsets: list[str]) -> HTMLResponse:
    if len(subsets) == 1:
        subset_field = f"""
        <label>
          <div>Subset</div>
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
          <div>Subset</div>
          <select name="subset">{options}</select>
        </label>
        """
    body = f"""
    <div class="card">
      <h1>Human Judge</h1>
      <p class="helper">
        Pick a subset and enter an annotator ID. Pairwise subsets show Response A
        vs Response B. Pointwise subsets show one blind response with the 4-part
        rubric. Annotations are saved under
        <code>human-judge/annotations/&lt;subset&gt;/&lt;annotator&gt;.jsonl</code>.
      </p>
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


def render_done(subset: str, annotator: str, completed: int) -> HTMLResponse:
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
) -> HTMLResponse:
    subset_q = quote_plus(subset)
    annotator_q = quote_plus(annotator)
    body = f"""
    <div class="card">
      <div class="meta">
        <div><strong>Subset:</strong> {html.escape(subset)}</div>
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
      <form action="/submit-pairwise" method="post">
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
) -> HTMLResponse:
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
        <div><strong>Subset:</strong> {html.escape(subset)}</div>
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


@app.get("/", response_class=HTMLResponse)
def home(subset: str | None = None, annotator: str | None = None) -> HTMLResponse:
    subsets = available_subsets()
    if not subsets:
        raise HTTPException(status_code=500, detail="No subsets found.")
    if subset is None or annotator is None:
        return render_home(subsets)

    annotator_id = normalize_annotator(annotator)
    items = load_items(subset)
    records = load_annotations(subset, annotator_id)
    current_item = next_unlabeled_item(items, records)
    if current_item is None:
        return render_done(subset, annotator_id, len(records))

    mode = subset_mode(subset)
    if mode == "pairwise":
        return render_pairwise_item_page(
            subset=subset,
            annotator=annotator_id,
            item=current_item,
            completed=len(records),
            total=len(items),
        )
    return render_pointwise_item_page(
        subset=subset,
        annotator=annotator_id,
        item=current_item,
        completed=len(records),
        total=len(items),
    )


@app.post("/submit")
@app.post("/submit-pairwise")
def submit_pairwise(
    subset: str = Form(...),
    annotator: str = Form(...),
    item_id: str = Form(...),
    label: str = Form(...),
) -> RedirectResponse:
    if subset_mode(subset) != "pairwise":
        raise HTTPException(status_code=400, detail=f"Subset is not pairwise: {subset}")
    annotator_id = normalize_annotator(annotator)
    normalized_label = normalize_label(label)
    items = load_items(subset)
    item_lookup = {item["item_id"]: item for item in items}
    if item_id not in item_lookup:
        raise HTTPException(status_code=404, detail=f"Unknown item_id: {item_id}")

    records = load_annotations(subset, annotator_id)
    records[item_id] = {
        "item_id": item_id,
        "label": normalized_label,
        "subset": subset,
        "annotator": annotator_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    save_annotations(subset, annotator_id, items, records)

    subset_q = quote_plus(subset)
    annotator_q = quote_plus(annotator_id)
    return RedirectResponse(url=f"/?subset={subset_q}&annotator={annotator_q}", status_code=303)


@app.post("/submit-pointwise")
def submit_pointwise(
    subset: str = Form(...),
    annotator: str = Form(...),
    item_id: str = Form(...),
    writing_quality_score: str = Form(...),
    evidence_use_score: str = Form(...),
    cti_concept_focus_score: str = Form(...),
) -> RedirectResponse:
    if subset_mode(subset) != "pointwise":
        raise HTTPException(status_code=400, detail=f"Subset is not pointwise: {subset}")
    annotator_id = normalize_annotator(annotator)
    scores = {
        "writing_quality_score": normalize_pointwise_score(writing_quality_score),
        "evidence_use_score": normalize_pointwise_score(evidence_use_score),
        "cti_concept_focus_score": normalize_pointwise_score(cti_concept_focus_score),
    }
    items = load_items(subset)
    item_lookup = {item["item_id"]: item for item in items}
    if item_id not in item_lookup:
        raise HTTPException(status_code=404, detail=f"Unknown item_id: {item_id}")

    records = load_annotations(subset, annotator_id)
    records[item_id] = {
        "item_id": item_id,
        "subset": subset,
        "annotator": annotator_id,
        **scores,
        "total_score": sum(scores.values()),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    save_annotations(subset, annotator_id, items, records)

    subset_q = quote_plus(subset)
    annotator_q = quote_plus(annotator_id)
    return RedirectResponse(url=f"/?subset={subset_q}&annotator={annotator_q}", status_code=303)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8787, reload=False)

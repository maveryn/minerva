#!/usr/bin/env python3
"""Generate a simplified MinervaRL diagram as Draw.io, SVG, and PDF."""

from __future__ import annotations

import html
import xml.etree.ElementTree as ET
import base64
from pathlib import Path
from urllib.parse import quote
import zlib

import cairosvg


ROOT = Path(__file__).resolve().parents[1]
DRAWIO_OUT = ROOT / "minerva_simplified.drawio"
SVG_OUT = ROOT / "latex" / "figures" / "minerva_simplified.svg"
PDF_OUT = ROOT / "latex" / "figures" / "minerva_simplified.pdf"


W, H = 1050, 560


def style(**kwargs: str) -> str:
    return ";".join(f"{k}={v}" for k, v in kwargs.items()) + ";"


def drawio_label(label: str) -> str:
    return "<br>".join(html.escape(part) for part in label.split("\n"))


def add_cell(parent_elem: ET.Element, cell_id: str, value: str = "", **attrs: str) -> ET.Element:
    cell = ET.SubElement(parent_elem, "mxCell", id=cell_id, value=value, **attrs)
    return cell


def add_vertex(
    parent: ET.Element,
    cell_id: str,
    label: str,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str,
    rounded: str = "1",
    font_size: str = "14",
    text_color: str = "#111111",
) -> None:
    cell = add_cell(
        parent,
        cell_id,
        drawio_label(label),
        parent="1",
        vertex="1",
        style=style(
            rounded=rounded,
            whiteSpace="wrap",
            html="1",
            fillColor=fill,
            strokeColor=stroke,
            fontSize=font_size,
            fontColor=text_color,
            spacing="8",
        ),
    )
    ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), as_="geometry")


def add_text(parent: ET.Element, cell_id: str, label: str, x: float, y: float, w: float, h: float, size: str = "14") -> None:
    cell = add_cell(
        parent,
        cell_id,
        html.escape(label),
        parent="1",
        vertex="1",
        style=style(
            text="",
            html="1",
            whiteSpace="wrap",
            strokeColor="none",
            fillColor="none",
            align="center",
            verticalAlign="middle",
            rounded="0",
            fontSize=size,
            fontColor="#111111",
        ),
    )
    ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), as_="geometry")


def add_edge(
    parent: ET.Element,
    cell_id: str,
    source: str,
    target: str,
    dashed: bool = False,
    label: str = "",
    stroke: str = "#111111",
) -> None:
    edge_style = style(
        edgeStyle="orthogonalEdgeStyle",
        rounded="0",
        orthogonalLoop="1",
        jettySize="auto",
        html="1",
        endArrow="block",
        endFill="1",
        strokeWidth="2",
        strokeColor=stroke,
        dashed="1" if dashed else "0",
        fontSize="13",
    )
    cell = add_cell(
        parent,
        cell_id,
        html.escape(label),
        parent="1",
        edge="1",
        source=source,
        target=target,
        style=edge_style,
    )
    ET.SubElement(cell, "mxGeometry", relative="1", as_="geometry")


def build_drawio() -> str:
    graph = ET.Element(
        "mxGraphModel",
        dx="1200",
        dy="720",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth=str(W),
        pageHeight=str(H),
        math="0",
        shadow="0",
    )
    root = ET.SubElement(graph, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    add_vertex(root, "lane_grpo", "", 25, 35, 1000, 170, "#EEF6FF", "#5B8BD6", rounded="1")
    add_vertex(root, "lane_minerva", "", 25, 240, 1000, 275, "#FFF5E8", "#D9902F", rounded="1")
    add_text(root, "lbl_grpo", "Standard GRPO / RLVR update (every step)", 50, 45, 360, 30, "16")
    add_text(root, "lbl_minerva", "MinervaRL auxiliary ACR distillation (hard prompts, every I steps)", 55, 250, 520, 30, "16")

    add_vertex(root, "rl_batch", "RL batch\noriginal prompt x", 55, 105, 125, 60, "#FFFFFF", "#333333")
    add_vertex(root, "actor", "Actor policy", 230, 105, 125, 60, "#FFFFFF", "#333333")
    add_vertex(root, "verifier", "Verifier", 420, 105, 125, 60, "#FFFFFF", "#333333")
    add_vertex(root, "grpo", "GRPO update", 610, 105, 135, 60, "#FFFFFF", "#333333")

    add_vertex(root, "hard_gate", "Hard-prompt gate\nmax reward < 1", 420, 295, 150, 70, "#FFFFFF", "#333333")
    add_vertex(root, "label_ref", "Label reference\ny*", 60, 355, 125, 60, "#FFFFFF", "#333333")
    add_vertex(root, "cond_prompt", "Label-conditioned\nprompt", 230, 305, 145, 65, "#FFFFFF", "#333333")
    add_vertex(root, "acr_buffer", "ACR prompt\nbuffer", 230, 420, 145, 60, "#FFFFFF", "#333333")
    add_vertex(root, "ema", "EMA teacher", 610, 295, 135, 60, "#FFFFFF", "#333333")
    add_vertex(root, "filter", "Verifier + filters", 815, 295, 145, 60, "#FFFFFF", "#333333")
    add_vertex(root, "distill_buffer", "Distillation buffer\n(x, y_acr)", 815, 420, 145, 60, "#FFFFFF", "#333333")
    add_vertex(root, "sample", "Sample M pairs", 610, 420, 135, 60, "#FFFFFF", "#333333")
    add_vertex(root, "sft", "Auxiliary SFT\nupdate", 420, 420, 145, 60, "#FFFFFF", "#333333")

    blue = "#2D6FB7"
    orange = "#C87800"
    add_edge(root, "e1", "rl_batch", "actor", label="sample", stroke=blue)
    add_edge(root, "e2", "actor", "verifier", label="rollouts", stroke=blue)
    add_edge(root, "e3", "verifier", "grpo", label="rewards", stroke=blue)
    add_edge(root, "e4", "grpo", "actor", dashed=True, label="policy update", stroke=blue)

    add_edge(root, "e5", "verifier", "hard_gate", dashed=True, label="hard samples", stroke=orange)
    add_edge(root, "e6", "hard_gate", "cond_prompt", stroke=orange)
    add_edge(root, "e7", "label_ref", "cond_prompt", stroke=orange)
    add_edge(root, "e8", "cond_prompt", "acr_buffer", stroke=orange)
    add_edge(root, "e9", "acr_buffer", "ema", dashed=True, label="every I steps", stroke=orange)
    add_edge(root, "e10", "ema", "filter", stroke=orange, label="ACR rollouts")
    add_edge(root, "e11", "filter", "distill_buffer", stroke=orange)
    add_edge(root, "e12", "distill_buffer", "sample", stroke=orange)
    add_edge(root, "e13", "sample", "sft", stroke=orange)
    add_edge(root, "e14", "sft", "actor", dashed=True, label="distill to actor", stroke=orange)
    add_edge(root, "e15", "grpo", "ema", dashed=True, label="EMA update", stroke=orange)

    add_text(root, "legend", "Solid arrows: per-step computation.  Dashed arrows: conditional or periodic updates.", 525, 525, 450, 25, "13")

    graph_xml = ET.tostring(graph, encoding="unicode").replace("as_=", "as=")
    payload = quote(graph_xml, safe="~()*!.'")
    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(payload.encode("utf-8")) + compressor.flush()
    encoded = base64.b64encode(compressed).decode("ascii")
    return (
        '<mxfile host="app.diagrams.net" '
        'agent="Mozilla/5.0" version="24.7.17">\n'
        f'  <diagram name="Page-1" id="minerva_simplified">{encoded}</diagram>\n'
        '</mxfile>\n'
    )


def svg_text(label: str, x: float, y: float, size: int = 16, weight: str = "400", anchor: str = "middle") -> str:
    label = label.replace("\\n", "\n")
    lines = label.split("\n")
    out = []
    start = y - (len(lines) - 1) * size * 0.6
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x}" y="{start + i * size * 1.2}" text-anchor="{anchor}" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="#111">{html.escape(line)}</text>'
        )
    return "\n".join(out)


def svg_box(x: float, y: float, w: float, h: float, label: str, fill: str = "#fff", stroke: str = "#333", radius: int = 8) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="2"/>\n'
        f'{svg_text(label, x + w / 2, y + h / 2 + 5, 16)}'
    )


def svg_arrow(x1: float, y1: float, x2: float, y2: float, color: str = "#111", dashed: bool = False, label: str = "") -> str:
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    midx, midy = (x1 + x2) / 2, (y1 + y2) / 2 - 8
    marker = "arrow_orange" if color == "#C87800" else "arrow_blue"
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="2.4"{dash} marker-end="url(#{marker})"/>\n'
        + (svg_text(label, midx, midy, 13) if label else "")
    )


def svg_polyline(points: list[tuple[float, float]], color: str = "#111", dashed: bool = False, label: str = "") -> str:
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    text = ""
    if label:
        x, y = points[len(points) // 2]
        text = svg_text(label, x, y - 10, 13)
    marker = "arrow_orange" if color == "#C87800" else "arrow_blue"
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"{dash} marker-end="url(#{marker})"/>\n{text}'


def build_svg() -> str:
    blue = "#2D6FB7"
    orange = "#C87800"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <marker id="arrow_blue" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto">
    <path d="M0,0 L10,4 L0,8 z" fill="{blue}"/>
  </marker>
  <marker id="arrow_orange" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto">
    <path d="M0,0 L10,4 L0,8 z" fill="{orange}"/>
  </marker>
</defs>
<rect width="100%" height="100%" fill="#fff"/>
<rect x="25" y="35" width="1000" height="170" rx="14" fill="#EEF6FF" stroke="#5B8BD6" stroke-width="2"/>
<rect x="25" y="240" width="1000" height="275" rx="14" fill="#FFF5E8" stroke="#D9902F" stroke-width="2"/>
{svg_text("Standard GRPO / RLVR update (every step)", 240, 65, 17, "700")}
{svg_text("MinervaRL auxiliary ACR distillation (hard prompts, every I steps)", 330, 270, 17, "700")}

{svg_box(55, 105, 125, 60, "RL batch\noriginal prompt x")}
{svg_box(230, 105, 125, 60, "Actor policy")}
{svg_box(420, 105, 125, 60, "Verifier")}
{svg_box(610, 105, 135, 60, "GRPO update")}

{svg_box(420, 295, 150, 70, "Hard-prompt gate\nmax reward < 1")}
{svg_box(60, 355, 125, 60, "Label reference\ny*")}
{svg_box(230, 305, 145, 65, "Label-conditioned\nprompt")}
{svg_box(230, 420, 145, 60, "ACR prompt\nbuffer")}
{svg_box(610, 295, 135, 60, "EMA teacher")}
{svg_box(815, 295, 145, 60, "Verifier + filters")}
{svg_box(815, 420, 145, 60, "Distillation buffer\n(x, y_acr)")}
{svg_box(610, 420, 135, 60, "Sample M pairs")}
{svg_box(420, 420, 145, 60, "Auxiliary SFT\nupdate")}

{svg_arrow(180, 135, 230, 135, blue, label="sample")}
{svg_arrow(355, 135, 420, 135, blue, label="rollouts")}
{svg_arrow(545, 135, 610, 135, blue, label="rewards")}
{svg_polyline([(745, 115), (790, 115), (790, 80), (292, 80), (292, 105)], blue, dashed=True, label="policy update")}

{svg_arrow(495, 165, 495, 295, orange, dashed=True, label="hard prompts")}
{svg_arrow(420, 330, 375, 330, orange)}
{svg_arrow(185, 385, 230, 340, orange)}
{svg_arrow(302, 370, 302, 420, orange)}
{svg_polyline([(302, 420), (302, 390), (678, 390), (678, 355)], orange, dashed=True)}
{svg_arrow(745, 325, 815, 325, orange, label="ACR rollouts")}
{svg_arrow(887, 355, 887, 420, orange)}
{svg_arrow(815, 450, 745, 450, orange)}
{svg_arrow(610, 450, 565, 450, orange)}
{svg_polyline([(420, 450), (390, 450), (390, 220), (292, 220), (292, 165)], orange, dashed=True, label="distill to actor")}
{svg_polyline([(678, 165), (678, 230), (678, 295)], orange, dashed=True, label="EMA update")}

{svg_text("Solid arrows: per-step computation.  Dashed arrows: conditional or periodic updates.", 750, 535, 13)}
</svg>
'''


def main() -> None:
    DRAWIO_OUT.write_text(build_drawio())
    SVG_OUT.write_text(build_svg())
    cairosvg.svg2pdf(bytestring=SVG_OUT.read_bytes(), write_to=str(PDF_OUT))
    print(f"Wrote {DRAWIO_OUT}")
    print(f"Wrote {SVG_OUT}")
    print(f"Wrote {PDF_OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert scientific_report.md to PDF using markdown + weasyprint.
Resolves relative image paths from docs/reports/ directory.
Renders LaTeX math ($..$ and $$..$$) as inline PNG images via matplotlib.
"""

import os
import re
import sys
import base64
import io
import markdown
from weasyprint import HTML
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
REPORTS_DIR = DOCS_DIR / "reports"
MD_FILE = REPORTS_DIR / "scientific_report.md"
PDF_FILE = REPORTS_DIR / "scientific_report.pdf"

# Cache rendered math to avoid duplicates
_math_cache: dict = {}


def latex_to_base64_img(latex: str, display: bool = False) -> str:
    """Render a LaTeX string to a base64-encoded PNG data URI."""
    cache_key = (latex, display)
    if cache_key in _math_cache:
        return _math_cache[cache_key]

    fontsize = 14 if display else 12
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis("off")
    fig.patch.set_alpha(0)

    text = ax.text(
        0.5, 0.5,
        f"${latex}$",
        fontsize=fontsize,
        ha="center", va="center",
        transform=ax.transAxes,
        color="#1a1a1a",
    )

    # Render to determine bounding box
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = text.get_window_extent(renderer=renderer)
    dpi = fig.dpi
    pad = 4
    width_in = (bbox.width + 2 * pad) / dpi
    height_in = (bbox.height + 2 * pad) / dpi
    fig.set_size_inches(max(width_in, 0.3), max(height_in, 0.2))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.03,
                transparent=True)
    plt.close(fig)

    b64 = base64.b64encode(buf.getvalue()).decode()
    data_uri = f"data:image/png;base64,{b64}"
    _math_cache[cache_key] = data_uri
    return data_uri


def replace_math_with_images(md_text: str) -> str:
    """Replace $$...$$ (display) and $...$ (inline) with <img> tags."""

    # Display math first ($$...$$)
    def _display_repl(m):
        latex = m.group(1).strip()
        uri = latex_to_base64_img(latex, display=True)
        return f'<div style="text-align:center;margin:0.4cm 0;"><img src="{uri}" style="border:none;max-height:60px;vertical-align:middle;" alt="equation"/></div>'

    md_text = re.sub(r'\$\$(.+?)\$\$', _display_repl, md_text, flags=re.DOTALL)

    # Inline math ($...$) — avoid matching $$
    def _inline_repl(m):
        latex = m.group(1).strip()
        uri = latex_to_base64_img(latex, display=False)
        return f'<img src="{uri}" style="border:none;display:inline;vertical-align:middle;max-height:18px;" alt="{latex}"/>'

    md_text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', _inline_repl, md_text)

    return md_text


def convert_md_to_pdf():
    # Read markdown
    md_text = MD_FILE.read_text(encoding="utf-8")

    # Render math to images before markdown conversion
    print("Rendering LaTeX math equations ...")
    md_text = replace_math_with_images(md_text)
    print(f"  Rendered {len(_math_cache)} unique equations")

    # Convert markdown to HTML
    extensions = [
        "tables",
        "fenced_code",
        "codehilite",
        "toc",
        "attr_list",
        "md_in_html",
    ]
    html_body = markdown.markdown(md_text, extensions=extensions)

    # Full HTML with CSS styling for a professional scientific report
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
@page {{
    size: A4;
    margin: 2cm 2.2cm;
    @bottom-center {{
        content: counter(page);
        font-size: 9pt;
        color: #666;
    }}
}}

body {{
    font-family: 'DejaVu Serif', 'Times New Roman', Georgia, serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1a1a1a;
    text-align: justify;
    hyphens: auto;
}}

h1 {{
    font-size: 20pt;
    text-align: center;
    margin-top: 0.5cm;
    margin-bottom: 0.3cm;
    color: #111;
    line-height: 1.3;
    page-break-after: avoid;
}}

h2 {{
    font-size: 15pt;
    color: #1a3a5c;
    border-bottom: 1.5pt solid #1a3a5c;
    padding-bottom: 4pt;
    margin-top: 1.2cm;
    margin-bottom: 0.4cm;
    page-break-after: avoid;
}}

h3 {{
    font-size: 12.5pt;
    color: #2a4a6c;
    margin-top: 0.7cm;
    margin-bottom: 0.3cm;
    page-break-after: avoid;
}}

h4 {{
    font-size: 11.5pt;
    color: #333;
    margin-top: 0.5cm;
    margin-bottom: 0.2cm;
    page-break-after: avoid;
}}

p strong {{ font-size: 11pt; }}
h2 + p {{ text-indent: 0; }}
p {{ margin-bottom: 0.35cm; orphans: 3; widows: 3; }}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.5cm 0;
    font-size: 10pt;
    page-break-inside: auto;
}}

thead {{ background-color: #1a3a5c; color: white; }}
th {{ padding: 6pt 8pt; text-align: left; font-weight: bold; border: 0.5pt solid #1a3a5c; }}
td {{ padding: 5pt 8pt; border: 0.5pt solid #ccc; vertical-align: top; }}
tr:nth-child(even) {{ background-color: #f5f7fa; }}
tr {{ page-break-inside: avoid; }}

code {{
    font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
    font-size: 9pt;
    background-color: #f0f2f5;
    padding: 1pt 3pt;
    border-radius: 2pt;
}}

pre {{
    background-color: #f0f2f5;
    border: 0.5pt solid #ddd;
    border-radius: 3pt;
    padding: 8pt 10pt;
    font-size: 8.5pt;
    line-height: 1.4;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    page-break-inside: avoid;
}}

pre code {{ background: none; padding: 0; }}

img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0.4cm auto;
    border: 0.5pt solid #ddd;
    border-radius: 2pt;
}}

blockquote {{
    margin: 0.4cm 0;
    padding: 6pt 12pt;
    border-left: 3pt solid #1a3a5c;
    background-color: #f5f7fa;
    font-size: 10pt;
    color: #333;
    page-break-inside: avoid;
}}

blockquote p {{ margin-bottom: 0.15cm; }}
hr {{ border: none; border-top: 1pt solid #ccc; margin: 0.6cm 0; }}
ul, ol {{ margin-bottom: 0.3cm; padding-left: 1.2cm; }}
li {{ margin-bottom: 0.1cm; }}
em {{ font-style: italic; color: #333; }}
table img {{ max-width: 100%; border: none; margin: 2pt auto; }}

h2:last-of-type ~ p {{
    text-indent: -1.5cm;
    padding-left: 1.5cm;
    margin-bottom: 0.2cm;
}}

hr + p {{ text-align: center; margin-bottom: 0.1cm; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    # Write PDF — use reports/ dir as base_url so relative image paths resolve
    print(f"Converting {MD_FILE.name} → {PDF_FILE.name} ...")
    HTML(
        string=full_html,
        base_url=str(REPORTS_DIR),
    ).write_pdf(str(PDF_FILE))

    size_mb = PDF_FILE.stat().st_size / (1024 * 1024)
    print(f"Done: {PDF_FILE}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    convert_md_to_pdf()

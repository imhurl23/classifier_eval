#!/usr/bin/env python3
"""Render REVIEW_CHECKLIST.md as a printable PDF (and the HTML it came from).

Usage:
    python3 review/build_printable.py            # writes REVIEW_CHECKLIST.pdf
    python3 review/build_printable.py --html     # also keep REVIEW_CHECKLIST.print.html

Requires pandoc and Google Chrome (or set CHROME to another Chromium binary).
Paper is US Letter; pass --a4 for A4.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "REVIEW_CHECKLIST.md"
PDF_OUT = ROOT / "REVIEW_CHECKLIST.pdf"
HTML_OUT = ROOT / "REVIEW_CHECKLIST.print.html"

CHROME_CANDIDATES = [
    os.environ.get("CHROME"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome"),
    shutil.which("chromium"),
    shutil.which("chromium-browser"),
]

CSS = """
@page {
  size: PAGE_SIZE;
  margin: 0.65in 0.7in 0.75in 0.7in;
  @top-left { content: "Safety and risk decision evaluation — review packet"; font: 8pt Helvetica, Arial, sans-serif; color: #555; }
  @top-right { content: "As of 2026-10-02 · development preparation, not frozen"; font: 8pt Helvetica, Arial, sans-serif; color: #555; }
  @bottom-center { content: "Page " counter(page) " of " counter(pages); font: 8pt Helvetica, Arial, sans-serif; color: #555; }
}
@page :first { @top-left { content: none; } @top-right { content: none; } }

html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font: 10pt/1.35 Helvetica, Arial, sans-serif;
  color: #000;
  margin: 0;
  max-width: none;
}
h1 { font-size: 17pt; margin: 0 0 8pt; line-height: 1.2; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; padding-bottom: 2pt; border-bottom: 1.5pt solid #000; break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
h1, h2, h3 { page-break-after: avoid; }
p { margin: 4pt 0 6pt; orphans: 3; widows: 3; }
blockquote { margin: 6pt 0 6pt 10pt; padding: 4pt 8pt; border-left: 3pt solid #888; background: #f3f3f3; }
blockquote p { margin: 0; }
hr { border: 0; border-top: 1pt solid #000; margin: 14pt 0; }

/* Lists and checklist items */
ul, ol { margin: 4pt 0 8pt; padding-left: 18pt; }
li { margin: 2pt 0; break-inside: avoid; page-break-inside: avoid; }
li.task { list-style: none; position: relative; padding-left: 4pt; }
li.task label { display: block; }
li.task input[type=checkbox] {
  appearance: none; -webkit-appearance: none;
  position: absolute; left: -16pt; top: 2pt;
  width: 10pt; height: 10pt; margin: 0;
  border: 1pt solid #000; border-radius: 1.5pt; background: #fff;
}
li.task input[type=checkbox]:checked { background: #000; }
li.task input[type=checkbox]:checked::after {
  content: ""; position: absolute; left: 3pt; top: 0.5pt; width: 2.5pt; height: 5.5pt;
  border: solid #fff; border-width: 0 1.6pt 1.6pt 0; transform: rotate(45deg);
}
li.task.done label { color: #333; }

/* Fill-in lines for decision / note fields */
ul.fields li { list-style: none; }
ul.fields li:not(.fill) { margin: 7pt 0; }
.yn { margin-left: 6pt; }
.yn .box { display: inline-block; width: 9pt; height: 9pt; border: 1pt solid #000; vertical-align: -1pt; margin: 0 3pt 0 6pt; }
li.fill { list-style: none; margin: 7pt 0; display: flex; align-items: baseline; gap: 6pt; }
li.fill .line { flex: 1; border-bottom: 0.8pt solid #000; min-width: 2in; height: 1em; }
.notes-box { margin: 2pt 0 12pt; break-inside: avoid; page-break-inside: avoid; }
.notes-box .rule { height: 20pt; border-bottom: 0.8pt solid #888; }

/* Tables */
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; line-height: 1.3; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th, td { border: 0.7pt solid #444; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #e6e6e6; font-weight: bold; }
td code, th code { white-space: nowrap; overflow-wrap: normal; }

/* Code */
code { font: 8.5pt/1.3 Menlo, Consolas, "Courier New", monospace; background: #f0f0f0; padding: 0 2pt; border-radius: 2pt; overflow-wrap: anywhere; }
pre { background: #f6f6f6; border: 0.7pt solid #bbb; padding: 6pt 8pt; margin: 4pt 0 10pt;
  white-space: pre-wrap; overflow-wrap: anywhere; break-inside: auto; }
pre code { background: none; padding: 0; font-size: 8.5pt; }

/* Links: show where they go, since paper has no hover */
a { color: #000; text-decoration: none; }
a[href^="http"]::after { content: " <" attr(href) ">"; font-size: 7.5pt; color: #444; overflow-wrap: anywhere; }
a[href]:not([href^="http"]):not([href^="#"])::after { content: " [" attr(href) "]"; font-size: 7.5pt; color: #444; overflow-wrap: anywhere; }

/* Page structure */
.page-break { break-before: page; page-break-before: always; }
.keep { break-inside: avoid; page-break-inside: avoid; }
.legend { font-size: 8.5pt; color: #333; margin: 2pt 0 10pt; }
.legend .box { display: inline-block; width: 8pt; height: 8pt; border: 1pt solid #000; vertical-align: -1pt; margin-right: 3pt; }
.legend .box.filled { background: #000; }
"""


def find_chrome() -> str:
    for cand in CHROME_CANDIDATES:
        if cand and Path(cand).exists():
            return cand
    sys.exit("Chrome/Chromium not found; set CHROME=/path/to/binary")


def run_pandoc() -> str:
    if not shutil.which("pandoc"):
        sys.exit("pandoc not found on PATH")
    result = subprocess.run(
        ["pandoc", "-f", "gfm", "-t", "html5", "--columns=10000", "--wrap=none", str(SOURCE)],
        check=True, capture_output=True, text=True,
    )
    return result.stdout


def post_process(body: str) -> str:
    # Checklist items: tag the <li> so CSS can draw print-friendly boxes.
    body = re.sub(
        r'<li><label><input type="checkbox" checked="" />',
        '<li class="task done"><label><input type="checkbox" checked="" disabled="" />',
        body,
    )
    body = re.sub(
        r'<li><label><input type="checkbox" />',
        '<li class="task"><label><input type="checkbox" disabled="" />',
        body,
    )

    # Fill-in fields: list items that end with a colon get a writing line.
    body = re.sub(
        r"<li>([^<\n]*?:)</li>",
        r'<li class="fill">\1<span class="line"></span></li>',
        body,
    )

    # Any list that contains fill-in fields is a form block: drop bullets on all its items.
    body = re.sub(
        r"<ul>\n((?:<li.*?</li>\n)+?)</ul>",
        lambda m: ('<ul class="fields">\n' if 'class="fill"' in m.group(1) else "<ul>\n") + m.group(1) + "</ul>",
        body,
        flags=re.DOTALL,
    )
    # "yes / no" at the end of a field becomes two tick boxes.
    body = re.sub(
        r"(<li>[^<\n]*?:) yes / no</li>",
        r'\1 <span class="yn"><span class="box"></span>yes <span class="box"></span>no</span></li>',
        body,
    )

    # "Notes / changes:" has no content after it; give reviewers ruled space.
    body = body.replace(
        "<p><strong>Notes / changes:</strong></p>",
        '<p><strong>Notes / changes:</strong></p>\n<div class="notes-box">' + '<div class="rule"></div>' * 6 + '</div>',
    )

    # Legend under the title block, before section 1.
    legend = (
        '<p class="legend"><span class="box filled"></span>decision recorded or work complete'
        '&nbsp;&nbsp;&nbsp;<span class="box"></span>still needs a decision, implementation, or verification'
        '&nbsp;&nbsp;&nbsp;Write directly on the lines and in the ruled boxes.</p>\n'
    )
    body = re.sub(r'(<h2 id="[^"]*start-here">)', lambda m: legend + m.group(1), body, count=1)

    # Start the appendix and the sign-off sheet on fresh pages.
    body = re.sub(r'<h1 id="(appendix[^"]*)">', r'<h1 class="page-break" id="\1">', body, count=1)
    body = re.sub(
        r'<h2 id="reviewer-sign-off">(.*)$',
        lambda m: '<section class="keep"><h2 id="reviewer-sign-off">' + m.group(1) + "</section>",
        body,
        flags=re.DOTALL,
    )
    # The <hr> before the appendix is redundant once there is a page break.
    body = re.sub(r"<hr />\s*(<h1 class=\"page-break\")", r"\1", body)
    return body


def build_html(page_size: str) -> str:
    body = post_process(run_pandoc())
    css = CSS.replace("PAGE_SIZE", page_size)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
        "<title>Safety and risk decision evaluation — review packet</title>\n"
        f"<style>{css}</style>\n</head>\n<body>\n{body}\n</body></html>\n"
    )


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = find_chrome()
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=2000",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--html", action="store_true", help="keep the intermediate HTML next to the PDF")
    parser.add_argument("--a4", action="store_true", help="use A4 paper instead of US Letter")
    args = parser.parse_args()

    html = build_html("A4" if args.a4 else "Letter")
    HTML_OUT.write_text(html, encoding="utf-8")
    try:
        render_pdf(HTML_OUT, PDF_OUT)
    finally:
        if not args.html:
            HTML_OUT.unlink(missing_ok=True)
    print(f"wrote {PDF_OUT.relative_to(ROOT)}" + (f" and {HTML_OUT.relative_to(ROOT)}" if args.html else ""))


if __name__ == "__main__":
    main()

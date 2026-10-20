"""Render the failure report with a full page for each chart."""

import html
import json
import subprocess
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/output/failures-2026-10-12/report.md"
OUTPUT = ROOT / "output/pdf/safety-classification-errors.pdf"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for name, filename in [
        ("Arial", "Arial.ttf"),
        ("Arial-Bold", "Arial Bold.ttf"),
        ("Arial-Italic", "Arial Italic.ttf"),
        ("Arial-BoldItalic", "Arial Bold Italic.ttf"),
    ]:
        pdfmetrics.registerFont(
            TTFont(name, "/System/Library/Fonts/Supplemental/" + filename)
        )
    pdfmetrics.registerFontFamily(
        "Arial",
        normal="Arial",
        bold="Arial-Bold",
        italic="Arial-Italic",
        boldItalic="Arial-BoldItalic",
    )
    body = ParagraphStyle(
        "body",
        fontName="Arial",
        fontSize=9.5,
        leading=12.5,
        spaceAfter=6,
        textColor=colors.HexColor("#18181B"),
    )
    styles = {
        1: ParagraphStyle(
            "title",
            parent=body,
            fontName="Arial-Bold",
            fontSize=22,
            leading=26,
            spaceAfter=12,
            keepWithNext=True,
        ),
        2: ParagraphStyle(
            "section",
            parent=body,
            fontName="Arial-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        3: ParagraphStyle(
            "chart",
            parent=body,
            fontName="Arial-Bold",
            fontSize=14,
            leading=18,
            spaceAfter=10,
            keepWithNext=True,
        ),
    }
    cell_style = ParagraphStyle(
        "cell", parent=body, fontSize=8.5, leading=11, spaceAfter=0
    )
    sizes = {"text": letter, "chart": landscape(letter), "matrix": (792, 1224)}
    margin = 38

    def normalize(text):
        for dash in ["\u2011", "\u2013", "\u2014", "\u2212"]:
            text = text.replace(dash, "-")
        return text

    def inline(nodes):
        chunks = []
        for node in nodes:
            t, c = node["t"], node.get("c")
            if t == "Str":
                chunks.append(html.escape(normalize(c)))
            elif t in ["Space", "SoftBreak"]:
                chunks.append(" ")
            elif t == "LineBreak":
                chunks.append("<br/>")
            elif t in ["Strong", "Emph"]:
                tag = "b" if t == "Strong" else "i"
                chunks.append(f"<{tag}>{inline(c)}</{tag}>")
            elif t == "Code":
                chunks.append(html.escape(normalize(c[1])))
            elif t == "Link":
                target = c[2][0]
                if not target.startswith(("https://", "http://")):
                    path, _, fragment = target.partition("#")
                    target = (SOURCE.parent / path).resolve().as_uri() + (
                        "#" + fragment if fragment else ""
                    )
                chunks.append(
                    f'<a href="{html.escape(target, quote=True)}" color="#2C1FEB">{inline(c[1])}</a>'
                )
            elif t == "Quoted":
                chunks.append("“" + inline(c[1]) + "”")
            else:
                raise ValueError("Unsupported inline type: " + t)
        return "".join(chunks)

    def block_text(blocks):
        return "<br/>".join(
            inline(b["c"]) for b in blocks if b["t"] in ["Para", "Plain"]
        )

    def footer(canvas, doc):
        width, height = canvas._pagesize
        canvas.setFont("Arial", 8)
        canvas.setFillColor(colors.HexColor("#6B6B76"))
        canvas.drawString(margin, 20, "Safety classification errors")
        canvas.drawRightString(width - margin, 20, str(doc.page))

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title="Safety classification errors",
        author="",
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id=name,
                pagesize=size,
                onPage=footer,
                frames=[
                    Frame(
                        margin,
                        margin,
                        size[0] - 2 * margin,
                        size[1] - 2 * margin,
                        leftPadding=0,
                        rightPadding=0,
                        topPadding=0,
                        bottomPadding=0,
                    )
                ],
            )
            for name, size in sizes.items()
        ]
    )
    ast = json.loads(subprocess.check_output(["pandoc", str(SOURCE), "-t", "json"]))
    story = []
    in_charts = False
    chart_title = ""
    for block in ast["blocks"]:
        t, c = block["t"], block.get("c")
        if t == "Header":
            level, _, content = c
            title = inline(content)
            if title == "Charts":
                in_charts = True
                continue
            if in_charts and level == 3:
                chart_title = title
                continue
            if in_charts:
                story += [NextPageTemplate("text"), PageBreak()]
                in_charts = False
            story.append(Paragraph(title, styles.get(level, styles[3])))
        elif t in ["Para", "Plain"]:
            story.append(Paragraph(inline(c), body))
        elif t in ["OrderedList", "BulletList"]:
            items = c[1] if t == "OrderedList" else c
            start = c[0][0] if t == "OrderedList" else 1
            for i, item in enumerate(items, start):
                prefix = f"{i}. " if t == "OrderedList" else "• "
                story.append(Paragraph(prefix + block_text(item), body))
        elif t == "Table":
            head = c[3][1]
            data_rows = head + [row for section in c[4] for row in section[3]]
            table_data = [
                [Paragraph(block_text(cell[4]), cell_style) for cell in row[1]]
                for row in data_rows
            ]
            ncols = len(table_data[0])
            width = letter[0] - 2 * margin
            widths = (
                [82, 98, 65, 54, 193, 44]
                if ncols == 6
                else [100, 218, 218]
                if ncols == 3
                else [220, 316]
            )
            assert sum(widths) == width
            table = Table(
                table_data, colWidths=widths, repeatRows=len(head), hAlign="LEFT"
            )
            table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, len(head) - 1),
                            colors.HexColor("#EEF0FF"),
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        (
                            "LINEBELOW",
                            (0, 0),
                            (-1, -1),
                            0.3,
                            colors.HexColor("#E8E8E8"),
                        ),
                    ]
                )
            )
            story += [table, Spacer(1, 10)]
        elif t == "Figure":
            image = c[2][0]["c"][0]
            assert image["t"] == "Image"
            path = SOURCE.parent / image["c"][2][0]
            template = (
                "matrix" if path.name == "06_user_assistant_safety.png" else "chart"
            )
            story += [
                NextPageTemplate(template),
                PageBreak(),
                Paragraph(chart_title, styles[3]),
            ]
            w, h = sizes[template]
            available_w = w - 2 * margin
            available_h = h - 2 * margin - (115 if template == "matrix" else 38)
            with PILImage.open(path) as im:
                iw, ih = im.size
            scale = min(available_w / iw, available_h / ih)
            story.append(
                Image(str(path), width=iw * scale, height=ih * scale, hAlign="CENTER")
            )
            story.append(Spacer(1, 8))
        else:
            raise ValueError("Unsupported block type: " + t)
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()

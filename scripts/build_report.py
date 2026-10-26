"""Render reports/report.md as a printable English PDF.

Install the optional report dependencies with: pip install -e '.[report]'
"""

from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/report.md"
DESTINATION = ROOT / "reports/customer_churn_report.pdf"
INK = colors.HexColor("#263942")
ACCENT = colors.HexColor("#275971")
MUTED = colors.HexColor("#65747b")


def register_fonts():
    """Embed fonts when available; retain portable standard-font fallbacks."""
    fonts = Path("/usr/share/fonts/truetype/liberation2")
    if not fonts.exists():
        fonts = Path("/usr/share/fonts/truetype/liberation")
    if (fonts / "LiberationSans-Regular.ttf").exists():
        pdfmetrics.registerFont(TTFont("Body", str(fonts / "LiberationSans-Regular.ttf")))
        pdfmetrics.registerFont(TTFont("Body-Bold", str(fonts / "LiberationSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")
        return "Body", "Body-Bold"
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    if (fonts / "DejaVuSans.ttf").exists():
        pdfmetrics.registerFont(TTFont("Body", str(fonts / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("Body-Bold", str(fonts / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("Mono", str(fonts / "DejaVuSansMono.ttf")))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")
        return "Body", "Body-Bold"
    return "Helvetica", "Helvetica-Bold"


def inline(text):
    text = escape(text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2" color="#275971">\1</a>', text)
    mono = "Mono" if "Mono" in pdfmetrics.getRegisteredFontNames() else "Courier"
    text = re.sub(r"`([^`]+)`", rf'<font name="{mono}" size="8.1">\1</font>', text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def styles():
    regular, bold = register_fonts()
    result = getSampleStyleSheet()
    result.add(ParagraphStyle("BodyText2", fontName=regular, fontSize=9.5,
                              leading=13.5, textColor=INK, spaceAfter=8, alignment=TA_LEFT))
    result.add(ParagraphStyle("ReportTitle", fontName=bold, fontSize=24,
                              leading=28, textColor=ACCENT, spaceAfter=18))
    result.add(ParagraphStyle("Section2", fontName=bold, fontSize=16,
                              leading=20, textColor=ACCENT, spaceAfter=12, keepWithNext=True))
    result.add(ParagraphStyle("Subsection2", fontName=bold, fontSize=11,
                              leading=15, textColor=INK, spaceBefore=8, spaceAfter=7, keepWithNext=True))
    result.add(ParagraphStyle("Cell2", fontName=regular, fontSize=8.3, leading=11.4,
                              textColor=INK, spaceAfter=0))
    return result


def footer(canvas, doc):
    canvas.saveState()
    page_width, page_height = A4
    canvas.setStrokeColor(colors.HexColor("#cfd9de"))
    canvas.line(48, page_height - 36, page_width - 48, page_height - 36)
    canvas.setFont("Body" if "Body" in pdfmetrics.getRegisteredFontNames() else "Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(48, page_height - 28, "CUSTOMER CHURN PREDICTION")
    canvas.drawRightString(page_width - 48, page_height - 28, "Machine Learning course project")
    canvas.drawString(48, 27, "Sharif University of Technology")
    canvas.drawRightString(page_width - 48, 27, str(doc.page))
    canvas.restoreState()


def render():
    style = styles()
    width = A4[0] - 96
    lines = SOURCE.read_text().splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), style["ReportTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), style["Section2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), style["Subsection2"]))
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
            image = Image(str(SOURCE.parent / match.group(2)))
            scale = min(width / image.imageWidth, 265 / image.imageHeight)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            story.extend([Spacer(1, 4), image, Spacer(1, 9)])
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph(inline(line), style["BodyText2"]))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                items = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", x) for x in items):
                    rows.append([Paragraph(inline(x), style["Cell2"]) for x in items])
                i += 1
            columns = len(rows[0])
            if columns == 5:
                widths = [width * .44] + [width * .14] * 4
            elif columns == 3:
                widths = [width * .42, width * .29, width * .29]
            else:
                widths = [width * .33, width * .67]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5eef2")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7f8")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#b8ccd6")),
            ]))
            story.extend([table, Spacer(1, 10)])
            continue
        else:
            paragraph = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(("#", "|", "![", "<!--")):
                i += 1
                paragraph.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(paragraph)), style["BodyText2"]))
        i += 1
    document = SimpleDocTemplate(
        str(DESTINATION), pagesize=A4, rightMargin=48, leftMargin=48,
        topMargin=52, bottomMargin=46, title="Customer Churn Prediction and Retention Analysis",
        author="Setayesh Heydari; Amir Hossein Shahrabi; Amir Abbas Donyadideh; Bardia Soltani Moakhar",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(DESTINATION)


if __name__ == "__main__":
    render()

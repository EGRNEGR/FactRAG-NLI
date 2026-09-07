"""Render the slide appendix of our conference report as a Cyrillic PDF."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


def read_slides(report: Path) -> list[tuple[str, list[str]]]:
    text = report.read_text(encoding="utf-8").split("<!-- SLIDES -->", 1)
    if len(text) != 2:
        raise ValueError("Мы не нашли раздел слайдов в докладе")
    slides: list[tuple[str, list[str]]] = []
    for section in text[1].split("### ")[1:]:
        lines = section.strip().splitlines()
        slides.append((lines[0], [line[2:] for line in lines[1:] if line.startswith("- ")]))
    if not slides:
        raise ValueError("Мы не нашли тезисы презентации")
    return slides


def generate(report: Path, output: Path, font: Path, bold: Path) -> int:
    slides = read_slides(report)
    pdfmetrics.registerFont(TTFont("RAG", str(font)))
    pdfmetrics.registerFont(TTFont("RAGBold", str(bold)))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(output), pagesize=(960, 540), invariant=1)
    canvas.setTitle("Наша локальная RAG-система: архитектура и проверка")
    canvas.setAuthor("РУДН, Факультет ИИ, ЗПИбд-03-25")
    title_style = ParagraphStyle("title", fontName="RAGBold", fontSize=30, leading=36,
                                 textColor=HexColor("#133744"))
    body_style = ParagraphStyle("body", fontName="RAG", fontSize=22, leading=31,
                                textColor=HexColor("#203641"))
    for number, (title, bullets) in enumerate(slides, 1):
        canvas.setFillColor(HexColor("#FAFBF8"))
        canvas.rect(0, 0, 960, 540, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#147D80"))
        canvas.rect(54, 474, 60, 5, fill=1, stroke=0)
        heading = Paragraph(escape(title), title_style)
        _, height = heading.wrap(850, 100)
        heading.drawOn(canvas, 54, 458 - height)
        y = 430 - height
        for bullet in bullets:
            paragraph = Paragraph(escape(bullet), body_style)
            _, height = paragraph.wrap(838, 420)
            if y - height < 65:
                raise ValueError(f"Мы обнаружили переполнение слайда {number}")
            paragraph.drawOn(canvas, 64, y - height)
            y -= height + 19
        canvas.setFont("RAG", 11)
        canvas.setFillColor(HexColor("#617780"))
        canvas.drawString(54, 27, "РУДН • ЗПИбд-03-25 • Наш локальный исследовательский стенд")
        canvas.drawRightString(906, 27, f"{number} / {len(slides)}")
        canvas.showPage()
    canvas.save()
    return len(slides)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    windows = Path("C:/Windows/Fonts")
    linux = Path("/usr/share/fonts/truetype/dejavu")
    parser = argparse.ArgumentParser(description="Мы создаём PDF из тезисов доклада")
    parser.add_argument("--report", type=Path, default=root / "docs/CONFERENCE_REPORT.md")
    parser.add_argument("--output", type=Path, default=root / "docs/presentation.pdf")
    parser.add_argument("--font", type=Path,
                        default=windows / "arial.ttf" if windows.exists()
                        else linux / "DejaVuSans.ttf")
    parser.add_argument("--bold-font", type=Path,
                        default=windows / "arialbd.ttf" if windows.exists()
                        else linux / "DejaVuSans-Bold.ttf")
    args = parser.parse_args()
    count = generate(args.report, args.output, args.font, args.bold_font)
    print(f"Мы подготовили {count} слайдов: {args.output}")


if __name__ == "__main__":
    main()

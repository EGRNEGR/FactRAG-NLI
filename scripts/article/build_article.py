"""Build the article (reports/article/FactRAG-NLI_article.md) into .docx and LaTeX.

Source conventions (one source for both outputs):
  @key value             front matter (type, udk, title, author, affil, email, cite_ru, ...)
  @abstract, @en_abstract, @author_info   the following lines up to the next @ marker
  @references            the numbered bibliography (reports/article/references.md)
  # / ##                 section / subsection
  $$ latex $$ (n)        display formula n; a following paragraph starting with "где" explains it
  ![Рисунок N – text](figN)   figure (reports/article/figures/figN.png / .pdf)
  Таблица N – text + pipe table
  - item                 list item
  inline: **bold**, $latex$, [@key; @key]  (citations numbered by first appearance)

The .docx follows the manuscript layout of the previous version: A4, margins 2 cm, Times New
Roman 14 pt, 1.5 spacing, first-line indent 1.25 cm; front matter, captions, tables and the
bibliography 12 pt. Formulas are native Word equations (LaTeX -> MathML by latex2mathml ->
OMML by the MML2OMML.XSL of Microsoft Office). The LaTeX source compiles with XeLaTeX.

Usage:  python scripts/article/build_article.py [--docx-copy PATH]
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from latex2mathml.converter import convert as latex_to_mathml
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "reports" / "article"
SOURCE = ART / "FactRAG-NLI_article.md"
REFS = ART / "references.md"
FIGS = ART / "figures"
DOCX = ART / "FactRAG-NLI_article.docx"
TEX = ART / "FactRAG-NLI_article.tex"
MML2OMML = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
FONT = "Times New Roman"

# ------------------------------------------------------------------ parsing


@dataclass
class Block:
    kind: str  # meta, abstract, heading, sub, para, where, eq, fig, table, item, refs, info
    text: str = ""
    number: str = ""
    rows: list[list[str]] = field(default_factory=list)


def parse(source: str) -> tuple[dict[str, str], list[Block]]:
    meta: dict[str, str] = {}
    blocks: list[Block] = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("@"):
            key, _, value = line[1:].partition(" ")
            if key in ("abstract", "en_abstract", "author_info"):
                body = []
                i += 1
                while i < len(lines) and not lines[i].startswith("@") and not lines[i].startswith("# "):
                    if lines[i].strip():
                        body.append(lines[i].strip())
                    i += 1
                blocks.append(Block(key, "\n".join(body)))
                continue
            if key == "references":
                blocks.append(Block("refs"))
            else:
                meta[key] = value.strip()
                if key in ("cite_ru", "cite_en", "keywords_ru", "keywords_en", "en_type"):
                    blocks.append(Block("meta", key))
            i += 1
            continue
        if line.startswith("## "):
            blocks.append(Block("sub", line[3:]))
        elif line.startswith("# "):
            blocks.append(Block("heading", line[2:]))
        elif m := re.match(r"^\$\$ (.+) \$\$ \((\d+)\)$", line):
            blocks.append(Block("eq", m[1], m[2]))
        elif m := re.match(r"^!\[(.+)\]\((fig\d+)\)$", line):
            blocks.append(Block("fig", m[1], m[2]))
        elif line.startswith("Таблица ") and (
            nxt := next((x for x in lines[i + 1 :] if x.strip()), "")
        ).startswith("|"):
            rows = []
            i += 1
            while not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"-{3,}", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(Block("table", line, rows=rows))
            continue
        elif line.startswith("- "):
            blocks.append(Block("item", line[2:]))
        elif line.startswith("где "):
            blocks.append(Block("where", line))
        else:
            blocks.append(Block("para", line))
        i += 1
    return meta, blocks


def references(blocks: list[Block], meta: dict[str, str]) -> tuple[dict[str, int], list[str]]:
    """Number the bibliography by first citation."""
    entries = {}
    for line in REFS.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, _, text = line.partition(": ")
            entries[key.strip()] = text.strip()
    order: dict[str, int] = {}
    texts = [b.text for b in blocks] + [c for b in blocks for r in b.rows for c in r]
    for text in texts:
        for group in re.findall(r"\[(@[^\]]+)\]", text):
            for key in re.findall(r"@([\w]+)", group):
                if key not in entries:
                    raise KeyError(f"unknown reference {key}")
                order.setdefault(key, len(order) + 1)
    unused = set(entries) - set(order)
    if unused:
        raise ValueError(f"references never cited: {sorted(unused)}")
    return order, [entries[k] for k in sorted(order, key=order.get)]


def cite(text: str, order: dict[str, int]) -> str:
    def repl(m: re.Match[str]) -> str:
        nums = [order[k] for k in re.findall(r"@([\w]+)", m[1])]
        return "[" + "; ".join(str(n) for n in nums) + "]"

    text = re.sub(r"\[(@[^\]]+)\]", repl, text)
    # Case ids (V3-TC-022) never break at a line end: non-breaking hyphen U+2011.
    return _CASE_ID.sub(lambda m: m[0].replace("-", "‑"), text)


_CASE_ID = re.compile(r"\bV3-[A-Z]+-\d+\b")


_TOKEN = re.compile(r"(\*\*[^*]+\*\*|\$[^$]+\$)")

# ------------------------------------------------------------------ docx helpers

XSL = etree.XSLT(etree.parse(str(MML2OMML)))


def omml(tex: str, size: float | None = None):
    # \text{a\_b} keeps the escaped underscore in MathML; LaTeX needs it, Word does not.
    tex = re.sub(r"\\text\{([^}]*)\}", lambda t: r"\text{" + t[1].replace(r"\_", "_") + "}", tex)
    mml = latex_to_mathml(tex)
    # Function names as upright identifiers (an operator would merge with its neighbours).
    mml = re.sub(r"<mo>(max|min|exp|log)</mo>", r'<mi mathvariant="normal">\1</mi>', mml)
    # Decimal comma: latex2mathml renders "0{,}3" as number, operator, number.
    mml = re.sub(r"<mn>(\d+)</mn>\s*<mrow>\s*<mo>&#x0002C;</mo>\s*</mrow>\s*<mn>(\d+)</mn>",
                 r"<mn>\1,\2</mn>", mml)
    result = XSL(etree.fromstring(mml.encode("utf-8")))
    element = etree.fromstring(etree.tostring(result.getroot()))
    m = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    # MML2OMML leaves the operand of a big operator (sum) outside it, with an empty base:
    # move the following element into the base.
    for nary in element.iter(f"{m}nary"):
        base = nary.find(f"{m}e")
        if base is not None and len(base) == 0 and nary.getnext() is not None:
            base.append(nary.getnext())
    if size is not None:
        # Word sizes a formula by the w:rPr of its runs (tables use a smaller font).
        w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        for run in element.iter(f"{m}r"):
            rpr = run.find(f"{w}rPr")
            if rpr is None:
                rpr = etree.Element(f"{w}rPr")
                text = run.find(f"{m}t")
                run.insert(list(run).index(text) if text is not None else len(run), rpr)
            for tag in ("sz", "szCs"):
                el = etree.SubElement(rpr, f"{w}{tag}")
                el.set(f"{w}val", str(int(size * 2)))
    return element


def set_font(run, size: float, *, bold=None, italic=None, name: str = FONT) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)


def fmt(p, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent: float | None = 1.25, spacing=1.5,
        before=0.0, after=0.0, keep_next=False) -> None:
    pf = p.paragraph_format
    p.alignment = align
    pf.first_line_indent = Cm(indent) if indent else None
    pf.line_spacing = spacing
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.keep_with_next = keep_next
    pf.widow_control = True


def inline(p, text: str, size: float, *, bold=False, italic=False) -> None:
    """Runs of a paragraph: **bold**, $math$ (native equations; a lone sub/superscript is a
    formatted run), plain text."""
    for part in _TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2])
            set_font(run, size, bold=True, italic=italic)
        elif part.startswith("$") and part.endswith("$"):
            tex = part[1:-1]
            if m := re.fullmatch(r"([_^])\{?(\w+)\}?", tex):
                run = p.add_run(m[2])
                set_font(run, size, bold=bold, italic=italic)
                if m[1] == "_":
                    run.font.subscript = True
                else:
                    run.font.superscript = True
            else:
                p._p.append(omml(tex, None if size == 14 else size))
        else:
            run = p.add_run(part)
            set_font(run, size, bold=bold, italic=italic)


def page_number(section) -> None:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            fld = OxmlElement("w:fldChar")
            fld.set(qn("w:fldCharType"), tag)
            run._r.append(fld)
        else:
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = text
            run._r.append(instr)
    set_font(run, 12)


def cell_borders(table) -> None:
    tbl = table._tbl
    pr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    pr.append(borders)


def repeat_header(row) -> None:
    trpr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trpr.append(el)


# ------------------------------------------------------------------ docx


def build_docx(meta: dict[str, str], blocks: list[Block], order: dict[str, int],
               bibliography: list[str]) -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2))
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(14)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    page_number(sec)

    def para(text, size=14, **kw):
        p = doc.add_paragraph()
        style = {k: kw.pop(k) for k in ("bold", "italic") if k in kw}
        fmt(p, **kw)
        inline(p, cite(text, order), size, **style)
        return p

    def front(text, *, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=0, before=0, size=12):
        p = doc.add_paragraph()
        fmt(p, align=align, indent=None, spacing=1.0, after=after, before=before)
        inline(p, text, size, bold=bold)
        return p

    # Russian front matter
    front(meta["type"], align=WD_ALIGN_PARAGRAPH.LEFT)
    front(meta["udk"], align=WD_ALIGN_PARAGRAPH.LEFT, after=12)
    front(meta["title"].upper(), bold=True, size=14, after=12)
    front(meta["author"], bold=True)
    front(meta["affil"])
    front(meta["email"], after=12)
    for b in blocks:
        if b.kind == "abstract":
            p = doc.add_paragraph()
            fmt(p, spacing=1.0, after=4)
            inline(p, b.text, 12)
        elif b.kind == "meta" and b.text == "keywords_ru":
            p = doc.add_paragraph()
            fmt(p, spacing=1.0, after=4)
            inline(p, "**Ключевые слова:** " + meta["keywords_ru"], 12)
        elif b.kind == "meta" and b.text == "cite_ru":
            p = doc.add_paragraph()
            fmt(p, spacing=1.0, after=12)
            inline(p, "**Для цитирования:** " + meta["cite_ru"], 12)
        elif b.kind == "meta" and b.text == "en_type":
            front(meta["en_type"], align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=6)
            front(meta["en_title"], bold=True, size=14, after=12)
            front(meta["en_author"], bold=True)
            front(meta["en_affil"])
            front(meta["en_email"], after=12)
        elif b.kind == "en_abstract":
            p = doc.add_paragraph()
            fmt(p, spacing=1.0, after=4)
            inline(p, b.text, 12)
        elif b.kind == "meta" and b.text == "keywords_en":
            p = doc.add_paragraph()
            fmt(p, spacing=1.0, after=4)
            inline(p, "**Keywords:** " + meta["keywords_en"], 12)
        elif b.kind == "meta" and b.text == "cite_en":
            p = doc.add_paragraph()
            fmt(p, spacing=1.0, after=12)
            inline(p, "**For citation:** " + meta["cite_en"], 12)
        elif b.kind == "heading":
            para(b.text, bold=True, before=12, after=6, keep_next=True,
                 align=WD_ALIGN_PARAGRAPH.LEFT)
        elif b.kind == "sub":
            para(b.text, bold=True, italic=True, before=8, after=4, keep_next=True,
                 align=WD_ALIGN_PARAGRAPH.LEFT)
        elif b.kind == "para":
            para(b.text)
        elif b.kind == "where":
            para(b.text, indent=None)
        elif b.kind == "item":
            para("– " + b.text)
        elif b.kind == "eq":
            p = doc.add_paragraph()
            fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT, indent=None, spacing=1.0, before=6, after=6)
            stops = p.paragraph_format.tab_stops
            stops.add_tab_stop(Cm(8.5), WD_TAB_ALIGNMENT.CENTER)
            stops.add_tab_stop(Cm(17), WD_TAB_ALIGNMENT.RIGHT)
            set_font(p.add_run("\t"), 14)
            p._p.append(omml(b.text))
            set_font(p.add_run(f"\t({b.number})"), 14)
        elif b.kind == "fig":
            p = doc.add_paragraph()
            fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, indent=None, spacing=1.0, before=6,
                keep_next=True)
            p.add_run().add_picture(str(FIGS / f"{b.number}.png"), width=Cm(17))
            cap = doc.add_paragraph()
            fmt(cap, align=WD_ALIGN_PARAGRAPH.CENTER, indent=None, spacing=1.0, after=10)
            inline(cap, b.text, 12)
        elif b.kind == "table":
            cap = doc.add_paragraph()
            fmt(cap, align=WD_ALIGN_PARAGRAPH.LEFT, indent=None, spacing=1.0, before=8, after=4,
                keep_next=True)
            inline(cap, b.text, 12)
            rows = b.rows
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            cell_borders(table)
            size = 11 if len(rows[0]) <= 4 else 10
            for r, row in enumerate(rows):
                for c, text in enumerate(row):
                    cell = table.cell(r, c)
                    cp = cell.paragraphs[0]
                    fmt(cp, align=WD_ALIGN_PARAGRAPH.LEFT if c == 0 or len(text) > 25
                        else WD_ALIGN_PARAGRAPH.CENTER, indent=None, spacing=1.0)
                    inline(cp, cite(text, order), size, bold=(r == 0))
            repeat_header(table.rows[0])
            # Keep a table on one page when it fits: rows never split, each row keeps with the next.
            for r, row in enumerate(table.rows):
                trpr = row._tr.get_or_add_trPr()
                trpr.append(OxmlElement("w:cantSplit"))
                if r < len(table.rows) - 1:
                    for cell in row.cells:
                        for cp in cell.paragraphs:
                            cp.paragraph_format.keep_with_next = True
            spacer = doc.add_paragraph()
            fmt(spacer, indent=None, spacing=1.0, after=4)
        elif b.kind == "refs":
            for n, text in enumerate(bibliography, start=1):
                p = doc.add_paragraph()
                fmt(p, spacing=1.0, after=3)
                inline(p, f"{n}. {text}", 12)
        elif b.kind == "author_info":
            for line in b.text.split("\n"):
                p = doc.add_paragraph()
                fmt(p, spacing=1.0, before=6 if line.startswith("**") else 0, after=3,
                    indent=None if line.startswith("**") else 1.25)
                inline(p, line, 12)
    doc.core_properties.title = meta["title"]
    doc.core_properties.author = "Мясников Е. Д."
    doc.save(DOCX)


# ------------------------------------------------------------------ LaTeX

_TEX_ESC = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
# URLs go to \url (xurl breaks them anywhere); a trailing dot or bracket is not part of the URL.
_URL = re.compile(r"(https?://[^\s()]+?)(?=[.,;)]?(?:\s|$))")


def tex_inline(text: str) -> str:
    out = []
    for part in _TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            out.append(r"\textbf{" + tex_inline(part[2:-2]) + "}")
        elif part.startswith("$") and part.endswith("$"):
            out.append(tex_math(part))
        else:
            for k, piece in enumerate(_URL.split(part)):
                if k % 2:
                    out.append(r"\url{" + piece + "}")
                else:
                    out.append("".join(_TEX_ESC.get(ch, ch) for ch in piece))
    return "".join(out)


def tex_math(tex: str) -> str:
    """unicode-math: bold symbols by \\symbf."""
    return tex.replace(r"\boldsymbol", r"\symbf")


def build_tex(meta: dict[str, str], blocks: list[Block], order: dict[str, int],
              bibliography: list[str]) -> None:
    t = tex_inline
    # Same layout as the .docx: body 14 pt (\normalsize of extarticle), front matter, captions
    # and the bibliography 12 pt (\small), tables 11/10 pt in a full grid, headings 14 pt bold.
    L = [
        r"\documentclass[14pt,a4paper]{extarticle}",
        r"\usepackage[a4paper,margin=2cm]{geometry}",
        r"\usepackage{fontspec}",
        r"\setmainfont{Times New Roman}",
        r"\usepackage{polyglossia}",
        r"\setdefaultlanguage{russian}",
        r"\setotherlanguage{english}",
        r"\usepackage{amsmath}",
        r"\usepackage{unicode-math}",
        r"\setmathfont{STIXTwoMath-Regular.otf}",
        r"\usepackage{graphicx,tabularx,array,caption,setspace,indentfirst,float}",
        r"\usepackage{xurl}",
        r"\usepackage[protrusion=true,expansion=false]{microtype}",  # XeTeX: protrusion only
        r"\usepackage[hidelinks]{hyperref}",
        r"\urlstyle{same}",
        r"\Urlmuskip=0mu",  # no letter spacing inside URLs of a justified line
        r"\captionsetup{labelformat=empty,font=small,justification=centering,skip=6pt}",
        r"\setlength{\parindent}{1.25cm}",
        r"\setlength{\emergencystretch}{3em}",
        r"\onehalfspacing",
        r"\tolerance=400",  # long Latin terms (mDeBERTa-v3) do not hyphenate in Russian text
        # Figures and tables stand where the source puts them, between paragraphs (as in the
        # .docx): [H] never lets a float cut a paragraph; a page before a large float may end short.
        r"\raggedbottom",
        r"\renewcommand{\arraystretch}{1.15}",
        # \hspace{0pt}: the first word of a cell may be hyphenated too
        r"\newcolumntype{Y}{>{\raggedright\arraybackslash\hspace{0pt}}X}",
        r"\makeatletter",
        r"\renewcommand\section{\@startsection{section}{1}{\z@}{12pt plus 2pt}{6pt}"
        r"{\normalfont\normalsize\bfseries\raggedright}}",
        r"\renewcommand\subsection{\@startsection{subsection}{2}{\z@}{8pt plus 2pt}{4pt}"
        r"{\normalfont\normalsize\bfseries\itshape\raggedright}}",
        r"\makeatother",
        r"\begin{document}",
        r"\hyphenation{RAG-Response}",  # Russian patterns do not split Latin identifiers
        r"\begin{singlespace*}\small",
        r"\noindent " + t(meta["type"]) + r"\\",
        r"\noindent " + t(meta["udk"]),
        r"\begin{center}\normalsize\textbf{" + t(meta["title"].upper()) + r"}\\[12pt]",
        r"\small\textbf{" + t(meta["author"]) + r"}\\",
        t(meta["affil"]) + r"\\" + t(meta["email"]) + r"\end{center}",
    ]
    for b in blocks:
        if b.kind == "abstract":
            L += [t(b.text), ""]
        elif b.kind == "meta" and b.text == "keywords_ru":
            L += [r"\textbf{Ключевые слова:} " + t(meta["keywords_ru"]), ""]
        elif b.kind == "meta" and b.text == "cite_ru":
            L += [r"\textbf{Для цитирования:} " + t(meta["cite_ru"]), ""]
        elif b.kind == "meta" and b.text == "en_type":
            L += [r"\begin{otherlanguage}{english}", r"\vspace{6pt}\noindent " + t(meta["en_type"]),
                  r"\begin{center}\normalsize\textbf{" + t(meta["en_title"]) + r"}\\[12pt]",
                  r"\small\textbf{" + t(meta["en_author"]) + r"}\\",
                  t(meta["en_affil"]) + r"\\" + t(meta["en_email"]) + r"\end{center}"]
        elif b.kind == "en_abstract":
            L += [t(b.text), ""]
        elif b.kind == "meta" and b.text == "keywords_en":
            L += [r"\textbf{Keywords:} " + t(meta["keywords_en"]), ""]
        elif b.kind == "meta" and b.text == "cite_en":
            L += [r"\textbf{For citation:} " + t(meta["cite_en"]), r"\end{otherlanguage}",
                  r"\end{singlespace*}", ""]
        elif b.kind == "heading":
            L += [r"\section*{" + t(b.text) + "}"]
        elif b.kind == "sub":
            L += [r"\subsection*{" + t(cite(b.text, order)) + "}"]
        elif b.kind == "para":
            L += [t(cite(b.text, order)), ""]
        elif b.kind == "where":
            L += [r"\noindent " + t(cite(b.text, order)), ""]
        elif b.kind == "item":
            L += [r"-- " + t(cite(b.text, order)) + r"\par"]
        elif b.kind == "eq":
            L += [r"\begin{equation}", tex_math(b.text), r"\tag{" + b.number + "}",
                  r"\end{equation}"]
        elif b.kind == "fig":
            L += [r"\begin{figure}[H]\centering",
                  r"\includegraphics[width=\textwidth]{figures/" + b.number + ".pdf}",
                  r"\caption{" + t(b.text) + "}", r"\end{figure}", ""]
        elif b.kind == "table":
            cols = len(b.rows[0])
            spec = "|" + "Y|" * cols
            size = r"\fontsize{11}{13}" if cols <= 4 else r"\fontsize{10}{12}"
            L += [r"\begin{table}[H]", r"\setstretch{1}",
                  r"\noindent{\small " + t(b.text) + r"}\par\vspace{4pt}",
                  size + r"\selectfont",
                  r"\noindent\begin{tabularx}{\textwidth}{" + spec + "}", r"\hline"]
            for r, row in enumerate(b.rows):
                cells = [t(cite(c, order)) for c in row]
                if r == 0:
                    cells = [r"\textbf{" + c + "}" for c in cells]
                L += [" & ".join(cells) + r" \\", r"\hline"]
            L += [r"\end{tabularx}", r"\end{table}", ""]
        elif b.kind == "refs":
            L += [r"\begin{singlespace*}\small\setlength{\topsep}{0pt}\setlength{\partopsep}{0pt}",
                  r"\begin{enumerate}\setlength{\itemsep}{3pt}\setlength{\parsep}{0pt}"]
            # an entry without Cyrillic is hyphenated by English patterns
            L += [r"\item " + (t(text) if re.search("[А-яЁё]", text)
                               else r"\foreignlanguage{english}{" + t(text) + "}")
                  for text in bibliography]
            L += [r"\end{enumerate}", r"\end{singlespace*}"]
        elif b.kind == "author_info":
            # as in the .docx: bold heading lines flush left, text lines with the paragraph indent
            L += [r"\par\begingroup\setstretch{1}\small"]
            for line in b.text.split("\n"):
                if line.startswith("**"):
                    L += [r"\vspace{6pt}\noindent " + t(line), ""]
                else:
                    L += [t(line), ""]
            L += [r"\par\endgroup"]
    L.append(r"\end{document}")
    TEX.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx-copy", type=Path, help="also copy the .docx there")
    args = parser.parse_args()
    meta, blocks = parse(SOURCE.read_text(encoding="utf-8"))
    order, bibliography = references(blocks, meta)
    build_docx(meta, blocks, order, bibliography)
    build_tex(meta, blocks, order, bibliography)
    print("docx:", DOCX)
    print("tex:", TEX)
    print("references:", len(bibliography))
    if args.docx_copy:
        shutil.copyfile(DOCX, args.docx_copy)
        print("copy:", args.docx_copy)


if __name__ == "__main__":
    main()

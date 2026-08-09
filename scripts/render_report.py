"""Render REPORT.md to REPORT.pdf with the figures appended.

reportlab rather than pandoc/LaTeX: neither is installed on this host and
weasyprint fails on a missing system library (libpango). Documented fallback
under the brief's dependency time-box -- the deliverable is a rendered PDF,
not a particular toolchain.

Paragraphs are JOINED before inline formatting is applied. Formatting
line-by-line breaks bold and italics that span a line break, which rendered
literal asterisks in the first version of this file.
"""
import html, re, sys, pathlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as RLImage, PageBreak, KeepTogether)
from PIL import Image as PILImage

ss = getSampleStyleSheet()
S = {
 "h1": ParagraphStyle("h1", ss["Title"], fontName="Helvetica-Bold", fontSize=15, leading=19, spaceAfter=10, alignment=0),
 "h2": ParagraphStyle("h2", ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, spaceBefore=13, spaceAfter=5),
 "h3": ParagraphStyle("h3", ss["Heading3"], fontName="Helvetica-Bold", fontSize=9.8, leading=12.5, spaceBefore=9, spaceAfter=3, textColor=colors.HexColor("#333333")),
 "p":  ParagraphStyle("p",  ss["BodyText"], fontName="Helvetica", fontSize=8.6, leading=11.8, spaceAfter=5),
 "li": ParagraphStyle("li", ss["BodyText"], fontName="Helvetica", fontSize=8.6, leading=11.6, leftIndent=9, bulletIndent=2, spaceAfter=2.5),
 "cap":ParagraphStyle("cap",ss["BodyText"], fontName="Helvetica-Oblique", fontSize=7.2, leading=9, textColor=colors.HexColor("#555555"), spaceBefore=2, spaceAfter=9),
}
META = colors.HexColor("#555555")
GLYPH = {"\u2074": "^4", "\u00b3": "^3", "\u2264": "<=", "\u2265": ">=", "\u00d7": "x"}

FIGS = [
    ("figures/F1_performance_vs_budget.png", "Figure 1. Macro-F1 against label budget under three shift regimes. Mean over 13 datasets; bands are 95% CI."),
    ("figures/F4_effect_sizes.png", "Figure 4. Paired effect sizes versus HVG+PCA with bootstrap 95% CI. Filled = significant after BH-FDR; grey band = pre-specified negligible effect."),
    ("figures/F2_embedding_structure.png", "Figure 2. Signal versus nuisance structure in each representation, and its relation to the transfer penalty. Descriptive: n=5 representations."),
    ("figures/F3_rarity.png", "Figure 3. Per-class effect sizes by cell-type prevalence stratum, at unrestricted labels."),
    ("figures/F5_gene_overlap.png", "Figure 5. Foundation-model advantage against train-test gene-space overlap (H4, null)."),
]


def inline(t):
    for k, v in GLYPH.items():
        t = t.replace(k, v)
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.S)
    t = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`([^`]+?)`", r'<font face="Courier" size="7.8">\1</font>', t)
    return t


def build_story(md):
    story, i, lines, buf = [], 0, md.split("\n"), []

    def flush():
        if buf:
            story.append(Paragraph(inline(" ".join(buf)), S["p"]))
            buf.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("| ") and i + 1 < len(lines) and set(lines[i+1].replace("|", "").strip()) <= set("-: "):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "):
                    rows.append(cells)
                i += 1
            w = (A4[0] - 40*mm) / len(rows[0])
            data = [[Paragraph(inline(c), ParagraphStyle("tc", S["p"], fontSize=7.4, leading=9.4,
                    fontName="Helvetica-Bold" if r == 0 else "Helvetica")) for c in row]
                    for r, row in enumerate(rows)]
            tb = Table(data, colWidths=[w]*len(rows[0]))
            tb.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EFEFEF")),
                ("LINEBELOW", (0,0), (-1,0), 0.6, colors.HexColor("#666666")),
                ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#CCCCCC")),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("TOPPADDING", (0,0), (-1,-1), 2.5), ("BOTTOMPADDING", (0,0), (-1,-1), 2.5)]))
            story += [Spacer(1, 3), tb, Spacer(1, 8)]
            continue
        if ln.startswith("#"):
            flush()
            lvl = len(ln) - len(ln.lstrip("#"))
            story.append(Paragraph(inline(ln[lvl:].strip()), S[f"h{min(lvl,3)}"]))
        elif ln.startswith("- ") or re.match(r"^\d+\. ", ln):
            flush()
            body = ln[2:] if ln.startswith("- ") else re.sub(r"^\d+\. ", "", ln)
            bullet = "\u2022" if ln.startswith("- ") else ln.split(".")[0] + "."
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and lines[j].strip():
                body += " " + lines[j].strip()
                j += 1
            i = j - 1
            story.append(Paragraph(inline(body), S["li"], bulletText=bullet))
        elif ln.strip() == "---":
            flush(); story.append(Spacer(1, 6))
        elif not ln.strip():
            flush()
        else:
            buf.append(ln.strip())
        i += 1
    flush()
    return story


def footer(cv, doc):
    cv.saveState(); cv.setFont("Helvetica", 6.5); cv.setFillColor(colors.HexColor("#888888"))
    cv.drawString(20*mm, 12*mm, "scfm-tme-benchmark - github.com/cbostwick87/scfm-tme-benchmark")
    cv.drawRightString(A4[0] - 20*mm, 12*mm, f"page {doc.page}")
    cv.restoreState()


def main(src="REPORT.md", dst="REPORT.pdf"):
    story = build_story(pathlib.Path(src).read_text())
    story += [PageBreak(), Paragraph("Figures", S["h2"])]
    for fn, cap in FIGS:
        if not pathlib.Path(fn).exists():
            raise FileNotFoundError(f"figure missing: {fn}")
        w0, h0 = PILImage.open(fn).size
        w = A4[0] - 40*mm
        h = w * h0 / w0
        if h > 95*mm:
            h = 95*mm; w = h * w0 / h0
        story.append(KeepTogether([RLImage(fn, width=w, height=h), Paragraph(cap, S["cap"])]))
    SimpleDocTemplate(dst, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                      topMargin=18*mm, bottomMargin=18*mm,
                      title="When do zero-shot scFM embeddings beat classical representations?",
                      author="Caleb Bostwick").build(story, onFirstPage=footer, onLaterPages=footer)
    # verify by reading the rendered file back, not by trusting the exit code
    from pypdf import PdfReader
    txt = "\n".join(pg.extract_text() for pg in PdfReader(dst).pages)
    assert "**" not in txt, "literal markdown bold survived into the PDF"
    assert "\ufffd" not in txt, "undefined glyph in the PDF"
    print(f"{dst}: {len(PdfReader(dst).pages)} pages")
    return txt


if __name__ == "__main__":
    main(*sys.argv[1:])

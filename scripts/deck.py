"""Piezas comunes de las dos presentaciones (36_build_presentations.py): diapositivas 16:9
con título-hallazgo, figura de views/, tablas editables, notas del orador y exportación a
PDF con PowerPoint cuando está instalado."""

import shutil
import subprocess
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt
from PIL import Image

from utils import VIEWS_DIR

WIDTH, HEIGHT = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.5)
TITLE_HEIGHT = Inches(1.1)
# Inicio del contenido: el título reserva dos líneas; con subtítulo, una más.
CONTENT_TOP = Inches(1.5)
CONTENT_TOP_SUB = Inches(1.95)
# Caracteres que caben en una línea de título a 26 puntos (aproximado, Calibri).
TITLE_CHARS_PER_LINE = 78
FONT = "Calibri"
DARK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x66, 0x66, 0x66)
ACCENT = RGBColor(0xD5, 0x5E, 0x00)   # el color del boosting en plot_style


def new_deck(title, author):
    """Presentación vacía 16:9 con título y autor en las propiedades del archivo."""
    deck = Presentation()
    deck.slide_width, deck.slide_height = WIDTH, HEIGHT
    properties = deck.core_properties
    properties.title, properties.author, properties.last_modified_by = title, author, author
    properties.comments = ""
    return deck


def _text(slide, text, left, top, width, height, size, bold=False, color=DARK,
          align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    lines = text if isinstance(text, list) else [text]
    for k, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if k == 0 else frame.add_paragraph()
        paragraph.alignment = align
        run = paragraph.add_run()
        run.text = line
        run.font.size, run.font.bold, run.font.name = Pt(size), bold, FONT
        run.font.color.rgb = color
        paragraph.space_after = Pt(size * 0.5)
    return box


def slide(deck, title, notes, subtitle=None):
    """Diapositiva en blanco con el título (el hallazgo), un subtítulo opcional y las
    notas del orador. Guarda en 'content_top' dónde empieza el contenido, según si el
    título ocupa una o dos líneas y si hay subtítulo."""
    page = deck.slides.add_slide(deck.slide_layouts[6])
    _text(page, title, MARGIN, Inches(0.25), WIDTH - 2 * MARGIN, Inches(1.0), 26, bold=True)
    shift = 0 if len(title) > TITLE_CHARS_PER_LINE else Inches(0.45)
    if subtitle:
        _text(page, subtitle, MARGIN, Inches(1.35) - shift, WIDTH - 2 * MARGIN, Inches(0.5),
              16, color=MUTED)
    page.content_top = (CONTENT_TOP_SUB if subtitle else CONTENT_TOP) - shift
    page.notes_slide.notes_text_frame.text = notes
    return page


def top(page):
    return page.content_top


def figure(page, name, left=None, top=None, max_width=None, max_height=None):
    """Inserta views/<name> escalada para caber en la caja, centrada si no hay 'left'."""
    path = VIEWS_DIR / name
    max_width = max_width or WIDTH - 2 * MARGIN
    max_height = max_height or HEIGHT - (top or TITLE_HEIGHT) - Inches(0.3)
    with Image.open(path) as image:
        ratio = image.width / image.height
    width = min(max_width, int(max_height * ratio))
    height = int(width / ratio)
    left = left if left is not None else int((WIDTH - width) / 2)
    top = top if top is not None else TITLE_HEIGHT + Inches(0.2)
    page.shapes.add_picture(str(path), Emu(left), Emu(top), Emu(width), Emu(height))
    return width, height


def bullets(page, lines, left, top, width, height, size=18):
    return _text(page, [f"• {line}" for line in lines], left, top, width, height, size)


def big_number(page, value, label, left, top, width, color=ACCENT):
    _text(page, value, left, top, width, Inches(1.1), 54, bold=True, color=color,
          align=PP_ALIGN.CENTER)
    _text(page, label, left, top + Inches(1.1), width, Inches(0.9), 16, color=MUTED,
          align=PP_ALIGN.CENTER)


def table(page, rows, left, top, width, col_widths=None, size=13, highlight_rows=(),
          text_cols=(0,)):
    """Tabla editable: la primera fila es el encabezado."""
    n_rows, n_cols = len(rows), len(rows[0])
    height = Inches(0.4) * n_rows
    shape = page.shapes.add_table(n_rows, n_cols, left, top, width, height)
    grid = shape.table
    if col_widths:
        total = sum(col_widths)
        for k, share in enumerate(col_widths):
            grid.columns[k].width = int(width * share / total)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = grid.cell(i, j)
            cell.text = str(value)
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if j in text_cols else PP_ALIGN.RIGHT
            run = paragraph.runs[0]
            run.font.size, run.font.name = Pt(size), FONT
            run.font.bold = i == 0 or i in highlight_rows
    return shape


def export_pdf(pptx_path, preview_dir=None):
    """Exporta a PDF con PowerPoint (Windows, por COM desde PowerShell) y, si se pide, cada
    diapositiva a PNG para revisarla. Devuelve la ruta del PDF o None si no hay PowerPoint."""
    if shutil.which("powershell") is None:
        return None
    pptx_path = Path(pptx_path).resolve()
    pdf_path = pptx_path.with_suffix(".pdf")
    preview = ""
    if preview_dir:
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
        preview = (f"$i=1; foreach ($s in $p.Slides) {{ $s.Export('{Path(preview_dir).resolve()}"
                   f"\\slide_' + $i + '.png', 'PNG', 1600, 900); $i++ }};")
    # Si PowerPoint ya estaba abierto, COM se engancha a esa instancia: solo se cierra la
    # presentación, no el programa.
    script = (
        "$ErrorActionPreference='Stop';"
        "$running = Get-Process POWERPNT -ErrorAction SilentlyContinue;"
        "try { $app = New-Object -ComObject PowerPoint.Application } catch { exit 3 };"
        f"$p = $app.Presentations.Open('{pptx_path}', $true, $false, $false);"
        f"$p.SaveAs('{pdf_path}', 32);" + preview +
        "$p.Close(); if (-not $running) { $app.Quit() }")
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                            capture_output=True, text=True)
    return pdf_path if result.returncode == 0 and pdf_path.exists() else None

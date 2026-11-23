"""Estilo común de las figuras de las presentaciones Beamer (plantilla Uniandes): un color
fijo por familia, la fuente de la plantilla, etiquetas en español, precios en millones de
pesos y tamaños pensados para el área útil de una diapositiva 16:9 (14 cm de ancho).

Cada figura se guarda en views/ en PDF (vectorial, para LaTeX) y en PNG (para el README).
"""

import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from utils import VIEWS_DIR  # noqa: E402

DPI = 300
MILLION = 1e6
CM = 1 / 2.54

# Área útil de la diapositiva (\textwidth de la plantilla = 14,0 cm). Alto pensado para un
# título de una línea y, como mucho, una frase de cierre debajo de la figura.
FULL = (14.0 * CM, 5.6 * CM)
FULL_TALL = (14.0 * CM, 6.3 * CM)
HALF = (6.8 * CM, 5.8 * CM)

# Colores de la plantilla (beamerthemeUniandes.sty).
ANDES = {
    "black": "#1A1A1A",
    "gray": "#4D4D4D",
    "mid": "#B3B3B3",
    "light": "#F2F2F2",
    "yellow": "#FFD100",
    "accent": "#C79A00",
    "red": "#B3261E",
}
GRID = "#E3E3E3"

# Un color por familia en todas las figuras. La mezcla, el boosting y la red usan los colores
# de la plantilla (negro, amarillo oscuro y rojo); las demás familias toman tonos que pasan
# la validación de daltonismo de la skill de gráficas junto a esos dos. El ingenuo, el kNN y
# el híbrido son referencias y van en grises, siempre con etiqueta directa.
FAMILY_COLORS = {
    "naive": ANDES["mid"],
    "linear": "#9085E9",
    "elastic_net": "#1C5CAB",
    "cart": "#E87BA4",
    "random_forest": "#1BAF7A",
    "boosting": ANDES["accent"],
    "neural_net": ANDES["red"],
    "ensemble": ANDES["black"],
    "knn": ANDES["gray"],
    "en_boosting": "#8C8C8C",
}
FAMILY_MARKERS = {
    "naive": "X", "linear": "v", "elastic_net": "^", "cart": "s", "random_forest": "D",
    "boosting": "o", "neural_net": "P", "ensemble": "*", "knn": "h", "en_boosting": "<",
}
FAMILY_LABELS = {
    "naive": "Ingenuo",
    "linear": "Regresión lineal",
    "elastic_net": "Elastic net",
    "cart": "CART",
    "random_forest": "Random forest",
    "boosting": "Boosting",
    "neural_net": "Red neuronal",
    "ensemble": "Mezcla",
    "knn": "kNN",
    "en_boosting": "Híbrido lineal + boosting",
}
# Familia de cada valor de la columna 'family' de cv_results.csv.
FAMILY_ALIASES = {"quantile_net": "neural_net", "catboost": "boosting",
                  "xgboost": "boosting"}

# Sobre y subpredicción: polos de la plantilla (rojo = pagar de más) y un azul frío.
OVER_COLOR = ANDES["red"]
UNDER_COLOR = "#1C5CAB"


def family_of(raw):
    """Familia para colorear a partir de la columna 'family' (o del nombre del modelo)."""
    return FAMILY_ALIASES.get(raw, raw)


def _template_font():
    """Source Sans 3, la fuente de la plantilla, si la distribución de LaTeX la trae."""
    if shutil.which("kpsewhich") is None:
        return None
    for weight in ("Regular", "RegularIt", "Semibold", "Bold"):
        found = subprocess.run(["kpsewhich", f"SourceSans3-{weight}.otf"],
                               capture_output=True, text=True).stdout.strip()
        if not found:
            return None
        font_manager.fontManager.addfont(found)
    return "Source Sans 3"


def setup():
    """Parámetros de matplotlib comunes a todas las figuras (texto de 8 pt a tamaño real)."""
    family = _template_font()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [f for f in (family, "Arial", "DejaVu Sans") if f],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 8,
        "axes.labelcolor": ANDES["gray"],
        "axes.edgecolor": ANDES["mid"],
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.color": ANDES["gray"],
        "ytick.color": ANDES["gray"],
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "text.color": ANDES["black"],
        "figure.dpi": 100,
        "figure.constrained_layout.use": True,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 3,
        "axes.formatter.use_locale": False,
        "axes.unicode_minus": True,
    })


def es_number(value, decimals=1):
    """Número con coma decimal y punto de miles, como en el resto del proyecto."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def es_formatter(decimals=0, scale=1.0, suffix=""):
    """Formateador de ejes en español (escala opcional, p. ej. pesos a millones)."""
    return FuncFormatter(lambda value, _: es_number(value / scale, decimals) + suffix)


def millions(ax, axis="y", decimals=0):
    """Eje en millones de pesos con separador de miles '.' y decimal ','."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(es_formatter(decimals, MILLION))


def save(fig, name):
    """Guarda la figura en views/ como PDF (para LaTeX) y PNG, y la cierra. Devuelve el PDF."""
    VIEWS_DIR.mkdir(exist_ok=True)
    stem = name.rsplit(".", 1)[0]
    pdf = VIEWS_DIR / f"{stem}.pdf"
    fig.savefig(pdf)
    fig.savefig(VIEWS_DIR / f"{stem}.png", dpi=DPI)
    plt.close(fig)
    return pdf

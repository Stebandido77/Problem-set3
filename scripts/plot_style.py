"""Estilo común de las figuras de las presentaciones: un color fijo por familia, etiquetas
en español, 300 dpi y precios en millones de pesos."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from utils import VIEWS_DIR  # noqa: E402

DPI = 300
MILLION = 1e6

# Paleta Okabe-Ito (apta para daltonismo): el mismo color para cada familia en todas las
# figuras. La red cuantílica y el MLP de la ronda 1 son la misma familia.
FAMILY_COLORS = {
    "naive": "#9E9E9E",
    "linear": "#56B4E9",
    "elastic_net": "#0072B2",
    "cart": "#E69F00",
    "random_forest": "#009E73",
    "boosting": "#D55E00",
    "neural_net": "#CC79A7",
    "ensemble": "#000000",
    "knn": "#F0E442",
    "en_boosting": "#8C564B",
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


def family_of(raw):
    """Familia para colorear a partir de la columna 'family' (o del nombre del modelo)."""
    return FAMILY_ALIASES.get(raw, raw)


def setup():
    """Parámetros de matplotlib comunes a todas las figuras."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "axes.formatter.use_locale": False,
    })


def millions(ax, axis="y", decimals=0):
    """Eje en millones de pesos con separador de miles '.' y decimal ','."""
    def fmt(value, _):
        text = f"{value / MILLION:,.{decimals}f}"
        return text.replace(",", "X").replace(".", ",").replace("X", ".")
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(FuncFormatter(fmt))


def es_number(value, decimals=1):
    """Número con coma decimal y punto de miles, como en el resto del proyecto."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def save(fig, name):
    """Guarda la figura en views/ a 300 dpi y la cierra."""
    VIEWS_DIR.mkdir(exist_ok=True)
    path = VIEWS_DIR / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path

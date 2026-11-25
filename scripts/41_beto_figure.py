"""Figura de la ronda de BETO para las presentaciones: error y sesgo de BETO y del ridge
sobre TF-IDF por decil del precio pedido, como lectores del texto por sí solos.

No entrena nada: lee views/beto_deciles.csv y views/beto_check.csv (38_beto_check.py), que
comparan las dos predicciones fuera de muestra con las mismas particiones.

Salida: views/beto_deciles.pdf y .png

Uso (desde scripts/):
  python 41_beto_figure.py
"""

import numpy as np
import pandas as pd

import plot_style as ps
from plot_style import ANDES, plt
from utils import VIEWS_DIR

SERIES = [  # columna, etiqueta, color, estilo de línea
    ("beto", "BETO", ANDES["black"], "-"),
    ("ridge", "Ridge sobre TF-IDF", ANDES["gray"], (0, (3, 1.5))),
]


def total_mae():
    """MAE total de cada lector del texto, ponderado por las filas de cada partición."""
    check = pd.read_csv(VIEWS_DIR / "beto_check.csv")
    return {name: np.average(check[f"mae_{name}"], weights=check["n"])
            for name, *_ in SERIES}


def plot_deciles(deciles, totals):
    """MAE (izquierda) y sesgo (derecha) por decil del precio pedido."""
    fig, axes = plt.subplots(1, 2, figsize=ps.FULL, sharex=True)
    x = deciles["decile"]
    for ax, metric, label in [(axes[0], "mae", "Error absoluto medio (M)"),
                              (axes[1], "bias", "Sesgo medio (M; + sobrepredice)")]:
        for name, series_label, color, style in SERIES:
            values = deciles[f"{metric}_{name}"]
            legend = f"{series_label} (MAE {ps.es_number(totals[name])} M)"
            ax.plot(x, values, color=color, ls=style, marker="o", ms=3, lw=1.3,
                    markeredgewidth=0, label=legend)
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xlabel("Decil del precio pedido")
        ax.yaxis.set_major_formatter(ps.es_formatter())
    axes[1].axhline(0, color=ANDES["gray"], lw=0.6, zorder=1)
    edges = (f"decil 1: {ps.es_number(deciles['price_min'].iloc[0], 0)}–"
             f"{ps.es_number(deciles['price_max'].iloc[0], 0)} M; decil 10: "
             f"{ps.es_number(deciles['price_min'].iloc[-1], 0)}–"
             f"{ps.es_number(deciles['price_max'].iloc[-1], 0)} M")
    axes[0].text(0.02, 0.97, edges, transform=axes[0].transAxes, va="top", fontsize=7,
                 color=ANDES["gray"])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2,
               handlelength=2.2, columnspacing=1.5)
    return ps.save(fig, "beto_deciles.png")


def main():
    ps.setup()
    deciles = pd.read_csv(VIEWS_DIR / "beto_deciles.csv")
    print(plot_deciles(deciles, total_mae()))


if __name__ == "__main__":
    main()

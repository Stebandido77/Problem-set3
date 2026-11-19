"""Tabla de resultados en Markdown para el README, a partir de views/cv_results.csv.

Salida: views/cv_results_table.md (se copia tal cual en la sección de resultados del README).

Uso: python scripts/07_results_table.py
"""

import pandas as pd

from utils import VIEWS_DIR


def millions(value):
    """Pesos -> millones con un decimal."""
    return f"{value / 1e6:.1f}"


def main():
    results = pd.read_csv(VIEWS_DIR / "cv_results.csv").sort_values("model")
    table = pd.DataFrame({
        "Modelo": "`" + results["model"] + "`",
        "Predictores": results["n_features"].map(lambda v: "–" if pd.isna(v) else f"{v:.0f}"),
        "MAE aleatoria": results["mae_random"].map(millions),
        "MAE espacial": results["mae_spatial"].map(millions),
        "MAE tipo Chapinero": results["mae_chapinero"].map(millions),
        "Sesgo tipo Chapinero": results["bias_chapinero"].map(lambda v: f"{v / 1e6:+.1f}"),
        "% sobrepredicho (Chapinero)": results["share_over_chapinero"].map(lambda v: f"{v:.0%}"),
    })
    header = "| " + " | ".join(table.columns) + " |"
    align = "|---|" + "---:|" * (len(table.columns) - 1)
    lines = [header, align] + ["| " + " | ".join(row) + " |" for row in table.to_numpy()]
    text = "Cifras en millones de pesos.\n\n" + "\n".join(lines) + "\n"
    (VIEWS_DIR / "cv_results_table.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

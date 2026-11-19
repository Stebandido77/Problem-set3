"""Tabla de resultados en Markdown para el README, a partir de views/cv_results.csv.

Salidas:
  views/cv_results_table.md     se copia tal cual en la sección de resultados del README
  views/northeast_deciles.csv   error y sesgo por decil de precio en el holdout nororiental,
                                más la media, el p95 y el máximo de lo predicho en el test

Uso: python scripts/07_results_table.py
"""

import pandas as pd

from utils import NORTHEAST_PRICE_CAP, PROCESSED_DIR, TARGET, VIEWS_DIR


def millions(value):
    """Pesos -> millones con un decimal."""
    return f"{value / 1e6:.1f}"


def or_dash(formatter):
    """Formato que deja un guion donde no hay dato (esquema no corrido)."""
    return lambda v: "–" if pd.isna(v) else formatter(v)


def decile_table(results):
    """Error y sesgo por decil de precio en el holdout nororiental para cada modelo."""
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    tests = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    rows = []
    for model in results["model"]:
        column = f"{model}__northeast"
        if column not in oof:
            continue
        valid = oof[column].notna()
        y, pred = oof.loc[valid, TARGET], oof.loc[valid, column]
        decile = pd.qcut(y, 10, labels=False) + 1
        err = pred - y
        row = {"model": model, "mae": err.abs().mean(), "bias": err.mean()}
        row.update({f"mae_d{d}": v for d, v in err.abs().groupby(decile).mean().items()})
        row.update({f"bias_d{d}": v for d, v in err.groupby(decile).mean().items()})
        # Extrapolación: en el holdout el ajuste solo ve precios hasta el tope (1.000 M);
        # en el test, hasta 1.650 M.
        row["share_val_over_cap"] = float((y > NORTHEAST_PRICE_CAP).mean())
        row["share_pred_over_cap"] = float((pred > NORTHEAST_PRICE_CAP).mean())
        row["max_pred_val"] = pred.max()
        if model in tests:
            test = tests[model]
            row.update(test_mean=test.mean(), test_p95=test.quantile(0.95),
                       test_max=test.max(), n_test_over_1650=int((test > 1.65e9).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    results = pd.read_csv(VIEWS_DIR / "cv_results.csv").sort_values("model")
    decile_table(results).to_csv(VIEWS_DIR / "northeast_deciles.csv", index=False)
    table = pd.DataFrame({
        "Modelo": "`" + results["model"] + "`",
        "Predictores": results["n_features"].map(or_dash(lambda v: f"{v:.0f}")),
        "MAE aleatoria": results["mae_random"].map(or_dash(millions)),
        "MAE espacial": results["mae_spatial"].map(or_dash(millions)),
        "MAE tipo Chapinero": results["mae_chapinero"].map(or_dash(millions)),
        "MAE nororiental": results["mae_northeast"].map(or_dash(millions)),
        "Sesgo nororiental": results["bias_northeast"].map(
            or_dash(lambda v: f"{v / 1e6:+.1f}")),
        "% sobrepredicho (nororiental)": results["share_over_northeast"].map(
            or_dash(lambda v: f"{v:.0%}")),
    })
    header = "| " + " | ".join(table.columns) + " |"
    align = "|---|" + "---:|" * (len(table.columns) - 1)
    lines = [header, align] + ["| " + " | ".join(row) + " |" for row in table.to_numpy()]
    text = "Cifras en millones de pesos.\n\n" + "\n".join(lines) + "\n"
    (VIEWS_DIR / "cv_results_table.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

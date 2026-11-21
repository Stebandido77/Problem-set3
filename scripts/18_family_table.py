"""Tabla de la ronda 5: por familia, la versión anterior, la versión con la receta de la 83
y el boosting 83, con la media, la mediana y el p95 de las predicciones del test (del
archivo de envío) y el MAE ponderado y espacial (de views/cv_results.csv).

Salida: views/round5_families.csv y views/round5_families.md
Uso: python 18_family_table.py
"""

import pandas as pd

from utils import TARGET, VIEWS_DIR, find_submission

BOOSTING = "83_boosting_ext_core_coords_by_type_level_q70_ridge_s10"
# Familia -> (versión anterior, versiones nuevas). La red no se reentrena: la 78 es la
# mejor red cuantílica por tipo que ya existe.
FAMILIES = {
    "Regresión lineal": ("03_linear_full", [
        "101_linear_ext_core_coords_splines_by_type_level_q70_ridge"]),
    "Elastic net": ("04_elastic_net_full", [
        "100_elastic_net_ext_core_coords_splines_by_type_level_q70_ridge",
        "104_elastic_net_ext_core_coords_splines_by_type_level_q70_ridge_tuned"]),
    "CART": ("21_cart_full_tuned", ["102_cart_ext_core_coords_by_type_level_q70_ridge",
                                    "105_cart_ext_core_coords_by_type_level_q70_ridge_tuned"]),
    "Random forest": ("24_random_forest_full_by_type_per_m2_tuned", [
        "99_random_forest_ext_core_coords_by_type_level_q70_ridge",
        "103_random_forest_ext_core_coords_by_type_level_q70_ridge_tuned"]),
    "Red neuronal": ("11_neural_net_full", ["78_quantile_net_core_coords_by_type_q70"]),
}


def test_stats(file_name):
    """Media, mediana y p95 de un envío en millones (vacío si no hay envío)."""
    if not isinstance(file_name, str):
        return {}
    price = pd.read_csv(find_submission(file_name))[TARGET] / 1e6
    return {"test_mean": price.mean(), "test_median": price.median(),
            "test_p95": price.quantile(0.95)}


def main():
    results = pd.read_csv(VIEWS_DIR / "cv_results.csv").set_index("model")
    rows = []
    for family, (old, new) in FAMILIES.items():
        versions = [("anterior", old)] + [("nueva", m) for m in new] + [("boosting 83", BOOSTING)]
        for version, model in versions:
            if model not in results.index:
                continue
            row = results.loc[model]
            rows.append({"family": family, "version": version, "model": model,
                         **test_stats(row.get("submission")),
                         "mae_weighted": row["mae_weighted"] / 1e6,
                         "mae_spatial": row["mae_spatial"] / 1e6})
    table = pd.DataFrame(rows).round(1)
    table.to_csv(VIEWS_DIR / "round5_families.csv", index=False)
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |"
              for row in table.itertuples(index=False)]
    (VIEWS_DIR / "round5_families.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table.to_string())


if __name__ == "__main__":
    main()

"""Regla de compra (etapa 4, pieza 4) con las predicciones espaciales fuera de muestra.

Simulación: la empresa ofrece su predicción × (1 − descuento) por cada inmueble anunciado.
La compra se cierra si la oferta es al menos el precio pedido (el precio del anuncio); si se
cierra, el sobrepago es la oferta menos ese precio. Es el peor caso: supone que la empresa
paga lo que ofrece y que el precio pedido es el valor del inmueble.

Reglas comparadas: la receta de la 125 con cuantil 0,20, 0,30 y 0,50 (filas 190 a 192) y
0,70 (la 127), todas con la regla 'mix' de gemelos de 20_twin_rules.py como la 127, y la
mezcla 152 (el mejor modelo en Kaggle). Descuentos de 0 % a 20 %, de 1 en 1. Las columnas
que falten se omiten con aviso.

Salidas:
  views/purchase_rule.csv            cierre y sobrepago por regla y descuento
  views/purchase_rule_segments.csv   lo mismo por tipo y área para la regla recomendada
  views/purchase_rule_northeast.csv  127 y 152 en el holdout nororiental (cambio de zona)
  views/purchase_rule_frontier.png   frontera entre compras cerradas y sobrepago

Uso:
  python scripts/35_purchase_rule.py
  python scripts/35_purchase_rule.py --recommended "q0,30" --discount 0.05   (otra regla por segmento)
"""

import argparse
import importlib

import numpy as np
import pandas as pd

import plot_style
from matplotlib.ticker import FuncFormatter, MaxNLocator
from plot_style import FAMILY_COLORS, es_number, plt
from utils import TARGET, VIEWS_DIR, load_processed

errors = importlib.import_module("34_best_model_errors")

MILLION = 1e6
DISCOUNTS = np.round(np.arange(0, 0.201, 0.01), 2)
MARKED_DISCOUNTS = (0.0, 0.05, 0.10, 0.15, 0.20)
KAGGLE_RULE = "152 (Kaggle, q0,70)"
# Tonos del color del boosting, del cuantil más bajo (claro) al más alto (oscuro).
QUANTILE_COLORS = {"q0,20": "#E8C866", "q0,30": "#C79A00", "q0,50": "#8F6F00",
                   "q0,70 (127)": "#574300"}
RULE_LABELS = {"q0,20": "cuantil 0,20", "q0,30": "cuantil 0,30", "q0,50": "cuantil 0,50",
               "q0,70 (127)": "cuantil 0,70 (127)", KAGGLE_RULE: "mezcla 152 (Kaggle)"}


def candidate_rules(train, oof):
    """Predicciones espaciales de cada regla; las que falten se omiten con aviso."""
    rules = {}
    for number, q in errors.LOW_QUANTILES.items():
        name = errors.low_quantile_name(number)
        pred = errors.twin_mix_spatial(name, train, oof)
        if pred is None:
            print(f"Aviso: falta {name}__spatial; se omite el cuantil 0,{q}")
            continue
        rules[f"q0,{q}"] = pred
    check_twin_rule(train, oof)
    for label, name in [("q0,70 (127)", "127_boosting_pt_core_coords_by_type_level_q70_ridge_"
                                        "nocoords_s10_twin_mix"),
                        (KAGGLE_RULE, errors.BEST)]:
        rules[label] = oof[f"{name}__spatial"].to_numpy(float)
    return rules


def check_twin_rule(train, oof):
    """La regla aplicada aquí a la 125 debe reproducir la 127 guardada."""
    rebuilt = errors.twin_mix_spatial(errors.BASE_125, train, oof)
    saved = oof["127_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_mix"
                "__spatial"].to_numpy(float)
    gap = np.abs(rebuilt - saved).max()
    assert gap < 1, f"la regla de gemelos no reproduce la 127 (diferencia {gap:.0f})"


def simulate(price, pred, discount):
    """Compras cerradas y sobrepago cuando se ofrece la predicción con descuento."""
    offer = pred * (1 - discount)
    closed = offer >= price
    overpay = (offer - price)[closed]
    capital = offer[closed].sum()
    return {"n_listings": len(price), "n_closed": int(closed.sum()),
            "share_closed": closed.mean(),
            "mean_overpay": overpay.mean() if closed.any() else 0.0,
            "mean_overpay_pct_price": (overpay / price[closed]).mean() if closed.any() else 0.0,
            "total_overpay": overpay.sum(), "capital": capital,
            "overpay_share_capital": overpay.sum() / capital if capital else 0.0}


def rule_table(price, rules, mask=None):
    rows = []
    for label, pred in rules.items():
        keep = np.ones(len(price), bool) if mask is None else mask
        for d in DISCOUNTS:
            rows.append({"rule": label, "discount": d, **simulate(price[keep], pred[keep], d)})
    return pd.DataFrame(rows)


def northeast_table(price, oof):
    """Sensibilidad al cambio de zona: la 127 y la 152 en el holdout nororiental, el vecino
    del test. Ese holdout ajusta solo con precios hasta 1.000 M, así que subpredice los
    caros por construcción: su sobrepago es un piso, no una estimación central."""
    rules = {}
    for label, name in [("q0,70 (127)", "127_boosting_pt_core_coords_by_type_level_q70_ridge_"
                                        "nocoords_s10_twin_mix"), (KAGGLE_RULE, errors.BEST)]:
        rules[label] = oof[f"{name}__northeast"].to_numpy(float)
    mask = ~np.isnan(rules[KAGGLE_RULE])
    return rule_table(price, rules, mask).assign(scheme="northeast")


def segment_table(train, price, pred, label, discount):
    """La regla recomendada por tipo y por disponibilidad del área efectiva."""
    kind = np.where(train["is_house"] == 1, "casa", "apartamento")
    area = np.where(train["surface_pt"].isna(), "sin área", "con área")
    rows = []
    for segment, mask in [("todas", np.ones(len(price), bool)),
                          *[(k, kind == k) for k in ("apartamento", "casa")],
                          *[(a, area == a) for a in ("con área", "sin área")],
                          *[(f"{k}, {a}", (kind == k) & (area == a))
                            for k in ("apartamento", "casa") for a in ("con área", "sin área")]]:
        rows.append({"rule": label, "discount": discount, "segment": segment,
                     **simulate(price[mask], pred[mask], discount)})
    return pd.DataFrame(rows)


def plot_frontier(table):
    """Frontera entre compras cerradas y sobrepago; cada línea es una regla y cada punto
    marcado un descuento (0 % a la derecha, 20 % a la izquierda)."""
    fig, axes = plt.subplots(1, 2, figsize=plot_style.FULL)
    metrics = [("mean_overpay", "Sobrepago medio por compra (M)", MILLION),
               ("overpay_share_capital", "Sobrepago (% del capital)", 0.01)]
    kaggle = table[(table["rule"] == KAGGLE_RULE) & (table["discount"] == 0)].iloc[0]
    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        for rule, part in table.groupby("rule", sort=False):
            color = QUANTILE_COLORS.get(rule, FAMILY_COLORS["ensemble"])
            style = (0, (3, 1.5)) if rule == KAGGLE_RULE else "-"
            x, y = part["share_closed"] * 100, part[metric] / scale
            ax.plot(x, y, color=color, ls=style, lw=1.2, label=RULE_LABELS[rule])
            marked = part["discount"].isin(MARKED_DISCOUNTS)
            ax.plot(x[marked], y[marked], "o", color=color, ms=2.5, markeredgewidth=0)
        kx, ky = kaggle["share_closed"] * 100, kaggle[metric] / scale
        ax.plot(kx, ky, "*", color=FAMILY_COLORS["ensemble"], ms=8, markeredgewidth=0,
                zorder=5)
        ax.annotate("152 sin\ndescuento", (kx, ky), xytext=(-5, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7, color=plot_style.ANDES["black"])
        ax.set_xlabel("Compras cerradas (% de los anuncios)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, (table["share_closed"] * 100).max() * 1.05)
        ax.set_ylim(0, (table[metric] / scale).max() * 1.15)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: es_number(v, 0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: es_number(v, 0)))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5,
               handlelength=1.8, columnspacing=1.2)
    return plot_style.save(fig, "purchase_rule_frontier.png")


def show(table, columns):
    out = table[columns].copy()
    for c in ["mean_overpay", "total_overpay", "capital"]:
        out[c] = out[c] / MILLION
    return out.round({"share_closed": 3, "mean_overpay": 1, "mean_overpay_pct_price": 3,
                      "total_overpay": 0, "capital": 0, "overpay_share_capital": 3})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommended", default="q0,20",
                        help="regla para la tabla por segmento")
    parser.add_argument("--discount", type=float, default=0.0)
    args = parser.parse_args()
    plot_style.setup()
    train, _ = load_processed()
    oof = errors.load_oof()
    price = train[TARGET].to_numpy(float)
    rules = candidate_rules(train, oof)

    table = rule_table(price, rules)
    table.to_csv(VIEWS_DIR / "purchase_rule.csv", index=False)
    print(plot_frontier(table))
    columns = ["rule", "discount", "n_closed", "share_closed", "mean_overpay",
               "mean_overpay_pct_price", "total_overpay", "capital", "overpay_share_capital"]
    print(show(table[table["discount"].isin(MARKED_DISCOUNTS)], columns).to_string(index=False))

    northeast = northeast_table(price, oof)
    northeast.to_csv(VIEWS_DIR / "purchase_rule_northeast.csv", index=False)
    print(show(northeast[northeast["discount"].isin(MARKED_DISCOUNTS)], columns)
          .to_string(index=False))

    label = args.recommended if args.recommended in rules else KAGGLE_RULE
    segments = pd.concat([segment_table(train, price, rules[label], label, args.discount),
                          segment_table(train, price, rules[KAGGLE_RULE], KAGGLE_RULE, 0.0)])
    segments.to_csv(VIEWS_DIR / "purchase_rule_segments.csv", index=False)
    print(show(segments, ["rule", "discount", "segment"] + columns[2:]).to_string(index=False))


if __name__ == "__main__":
    main()

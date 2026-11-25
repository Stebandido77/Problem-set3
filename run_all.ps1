# Ruta principal para Windows (PowerShell): el mismo flujo de run_all.sh.
# Uso:  .\run_all.ps1               # usa las descargas externas ya versionadas
#       .\run_all.ps1 -Download     # vuelve a descargar OSM y los datos de Bogotá
# Usa el Python del entorno virtual .venv si existe; si no, el 'python' del sistema.
param([switch]$Download)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
Set-Location (Join-Path $root "scripts")

function Run {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Falló: python $args" }
}

# 1. Fuentes externas.
if ($Download) {
    Run 01_download_osm.py
    Run 11_download_bogota.py
    Run 16_download_lots.py
} else {
    Run 01_download_osm.py --cache-only
}

# 2. Predictores, modelos ingenuos y embeddings del texto (los usa el bosque 114).
Run 02_build_features.py
Run 03_naive_baselines.py
Run 13_text_embeddings.py

# 3. Mejor especificación de cada familia, la 127 y la mezcla final 152.
Run 04_train_models.py --specs 104 105 108 114 83 125 148 --schemes random random_twins spatial northeast
Run 20_twin_rules.py --rules mix --numbers 127 --model 125_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10
Run 06_ensemble.py --number 152 --how weighted --weights 0.7 0.3 --models `
    127_boosting_pt_core_coords_by_type_level_q70_ridge_nocoords_s10_twin_mix `
    148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords

# 3b. El kNN 185 y la mezcla 186 (181 = 177 con la regla de gemelos, más la red 148).
Run 04_train_models.py --specs 177 185 --schemes spatial northeast
Run 20_twin_rules.py --rules mix --numbers 181 --model 177_boosting_pt_core_coords_by_type_pinball_ridge_nocoords_s10
Run 06_ensemble.py --number 186 --how weighted --weights 0.7 0.3 --models `
    181_boosting_pt_core_coords_by_type_pinball_ridge_nocoords_s10_twin_mix `
    148_quantile_net_ext_core_coords_by_type_q70_ridge_nocoords

# 3c. Mejor modelo en Kaggle: la 203 (boosting 198 y red 196, con BETO y ridge).
Run 38_beto_check.py
Run 04_train_models.py --specs 194 196 --schemes spatial northeast
Run 20_twin_rules.py --rules mix --numbers 198 `
    --model 194_boosting_pt_core_coords_by_type_level_q70_ridge_beto_nocoords_s10
Run 06_ensemble.py --number 203 --how weighted --weights 0.7 0.3 --models `
    198_boosting_pt_core_coords_by_type_level_q70_ridge_beto_nocoords_s10_twin_mix `
    196_quantile_net_ext_core_coords_by_type_q70_ridge_beto_nocoords

# 4. Cuantiles 0,20, 0,30 y 0,50 para la regla de compra.
Run 04_train_models.py --specs 190 191 192 --schemes spatial --no-submission
Run 07_results_table.py

# 5. Figuras, tablas y presentaciones.
Run 31_data_overview.py
Run 33_validation_schemes.py
Run 32_algorithm_comparison.py
Run 34_best_model_errors.py
Run 35_purchase_rule.py
Run 41_beto_figure.py
Run 42_deck_checks.py
Run 36_build_presentations.py

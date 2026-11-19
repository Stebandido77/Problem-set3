Cifras en millones de pesos.

| Modelo | Predictores | MAE aleatoria | MAE espacial | MAE tipo Chapinero | Sesgo tipo Chapinero | % sobrepredicho (Chapinero) |
|---|---:|---:|---:|---:|---:|---:|
| `01_linear_base` | 8 | 187.4 | 202.2 | 234.8 | -183.0 | 26% |
| `02_linear_base_text` | 26 | 173.3 | 186.0 | 211.3 | -156.5 | 30% |
| `03_linear_full` | 46 | 155.8 | 168.1 | 183.3 | -55.6 | 47% |
| `04_elastic_net_full` | 46 | 132.8 | 159.7 | 172.0 | -61.5 | 42% |
| `05_cart_full` | 46 | 130.6 | 169.9 | 195.6 | -103.6 | 36% |
| `06_random_forest_full` | 46 | 93.9 | 147.6 | 180.2 | -110.4 | 35% |
| `07_random_forest_full_coords` | 48 | 93.0 | 146.6 | 181.2 | -111.6 | 35% |
| `08_boosting_base_text` | 26 | 124.4 | 150.7 | 181.2 | -138.4 | 24% |
| `09_boosting_full` | 46 | 96.5 | 145.8 | 182.3 | -120.8 | 30% |
| `10_boosting_full_coords` | 48 | 95.6 | 145.4 | 179.2 | -115.8 | 31% |
| `11_neural_net_full` | 46 | 146.8 | 196.6 | 237.2 | +56.6 | 57% |
| `12_boosting_full_coords_weighted` | 48 | 112.7 | 150.1 | 176.5 | -117.3 | 31% |
| `13_random_forest_full_weighted` | 46 | 107.1 | 151.2 | 179.2 | -108.6 | 36% |
| `14_boosting_full_coords_by_type` | 48 | 90.5 | 142.7 | 177.6 | -114.2 | 31% |
| `15_random_forest_full_by_type` | 46 | 91.5 | 146.8 | 177.5 | -106.5 | 35% |
| `16_boosting_full_coords_by_type_level` | 48 | 90.8 | 144.5 | 171.2 | -97.3 | 34% |
| `17_boosting_full_coords_by_type_per_m2` | 48 | 90.7 | 142.6 | 169.9 | -102.4 | 32% |
| `18_random_forest_full_by_type_per_m2` | 46 | 91.7 | 146.8 | 172.2 | -98.9 | 37% |
| `19_boosting_rich_coords_by_type_per_m2` | 72 | 89.4 | 144.1 | 174.5 | -111.2 | 30% |
| `20_random_forest_rich_by_type_per_m2` | 70 | 89.7 | 150.4 | 177.3 | -110.9 | 33% |
| `21_cart_full_tuned` | 46 | 137.6 | 165.9 | 193.2 | -116.1 | 34% |
| `22_neural_net_full_tuned` | 46 | 129.8 | 170.6 | 212.0 | -62.2 | 38% |
| `23_boosting_full_coords_by_type_per_m2_tuned` | 48 | 95.7 | 143.8 | 174.0 | -108.2 | 31% |
| `24_random_forest_full_by_type_per_m2_tuned` | 46 | 96.0 | 147.2 | 172.7 | -102.3 | 35% |
| `30_ensemble_median_3` | – | 93.0 | 142.2 | 167.3 | -94.6 | 35% |
| `31_ensemble_mean_3` | – | 99.9 | 142.9 | 164.6 | -87.6 | 36% |

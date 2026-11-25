# Presentaciones

- `best_equipo_06`: el mejor modelo (mezcla 203, primer puesto en Kaggle) y si se puede confiar en él para comprar.
- `compare_equipo_06`: comparación de familias y de esquemas de validación.

Están hechas en Beamer con la plantilla de la Universidad de los Andes
(`beamerthemeUniandes.sty`, `preamble.tex` y `figuras/logo-uniandes.png`, sin cambios).
Las notas del orador están aparte, en `notas_best.md` y `notas_compare.md`.

`scripts/36_build_presentations.py` genera `generated/cifras.tex` (una macro por cifra),
las tablas `generated/tabla_*.tex` y las notas a partir de `views/`, y compila los dos PDF.
Para compilar a mano, desde `document/`:

```
latexmk -lualatex best_equipo_06.tex
latexmk -lualatex compare_equipo_06.tex
```

Se usa LuaLaTeX porque la plantilla carga Source Sans 3 y, en MiKTeX, pdfLaTeX no la
encuentra si los mapas de fuentes no están actualizados (`initexmf --mkmaps`).

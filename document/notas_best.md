# Notas del orador: best_equipo_06

El mejor modelo y la decisión de compra. Neislen García y Esteban Labastidas, Equipo 06. Una sección por diapositiva de contenido, en orden. Generado por `scripts/36_build_presentations.py`; las cifras salen de `views/`.

## 1. Resultado

Empezamos por el resultado. La mezcla 203 quedó primera en los dos tableros de Kaggle. El MAE es el error promedio, en pesos, entre el precio predicho y el observado. En el tablero público, que usa el 20 % del test, fue de 145,7 millones; en el privado, el 80 % restante y el que define la competencia, fue de 148,6 M, 2,2 M menos que el segundo equipo (150,8 M). Del público al privado el error sube 2,9 M (2,0 %) y el puesto se mantiene. La validación espacial prueba el modelo en zonas de Bogotá que no vio al entrenar; da 124,8 M, por debajo de los dos tableros. Los precios son de oferta: lo que pide el vendedor. La pregunta de la presentación es si este modelo sirve para comprar.

## 2. Chapinero frente al resto de Bogotá

Entrenamos con el resto de Bogotá y predecimos Chapinero, que el modelo casi no ve. Chapinero es distinto: las casas son el 3 % de los anuncios frente al 24 % en el resto de la ciudad, el estrato medio es más alto y el valor del terreno por metro cuadrado es 1,9 veces el del resto de Bogotá. El test no trae precios, así que el valor del terreno es nuestra mejor referencia de su nivel. Al 36 % de los anuncios del test les falta el área (3.710 anuncios); al leer el texto y tomar el área de otros anuncios del mismo punto, la falta baja al 20 % (2.029).

## 3. Predictores

Los predictores externos que entran al mejor modelo son el estrato de la manzana (Planeación), el valor del terreno por metro cuadrado (Catastro) y la distancia a cuatro hitos de la ciudad. El lote, los pisos y el área construida del Catastro se probaron: cambiaron la validación espacial en −1,1 M, que es ruido, y quedaron fuera. Los 16 predictores de OpenStreetMap empeoraron la espacial (+7,9 a +8,6 M) y también quedaron fuera. Del texto entran 18 señales armadas con reglas (parqueadero, terraza, ascensor, vista, estrato declarado, piso...) y dos lectores del texto completo. El ridge sobre TF-IDF es una regresión que le pone un peso a cada palabra del anuncio; BETO es un modelo de lenguaje en español, afinado aquí para estimar el precio desde el texto. Los dos se ajustan fuera de muestra: la estimación de un anuncio nunca usa su propio precio.

## 4. Qué es el modelo

El modelo tiene dos partes que ven lo mismo: el inmueble, la ubicación y las dos lecturas del texto. El boosting son muchos árboles de decisión pequeños, cada uno corrige el error del anterior; la red neuronal es un modelo que puede predecir por encima de los precios que vio. Los dos apuntan al cuantil 0,70: un precio que el 70 % de los inmuebles parecidos no supera. Cada uno se entrena aparte para apartamentos y para casas; el boosting promedia 10 semillas y la red 5. La predicción final es una mezcla, un promedio ponderado: 70 % del boosting y 30 % de la red.

## 5. El camino en seis pasos

La tabla resume el camino en el tablero público. Del ingenuo (377,1 M) al modelo final (145,7 M) el error baja 61 %; los pasos 2 y 3 explican 216,1 de los 231,4 M. El primer boosting usa el inmueble (tipo, alcobas, baños, área), las 18 señales del texto, las distancias a los hitos y la latitud y longitud. Antes del paso 3 multiplicamos a mano ese boosting por 1,15 y bajó a 171,7 M: es una calibración con el tablero, sin modelo, que mostró que el nivel estaba bajo. El cuantil 0,70 lo escogimos con la retroalimentación del tablero: el 0,65 dio 161,7 M, una diferencia de ruido. El paso 4 agrega el texto completo, el estrato y el valor del terreno; el 5, la red; el 6, BETO. Las diferencias de menos de 2 M en el tablero son ruido; el paso 6 gana 2,6 M.

## 6. Qué aportó BETO

Comparamos los dos lectores del texto solos, fuera de muestra, con las mismas particiones espaciales. BETO se equivoca en promedio 148,5 M y el ridge 169,1 M. Por decil de precio, el ridge acierta más en los precios del medio (decil 6: 87 contra 133 M) y BETO en los extremos (decil 10: 341 contra 537 M): el ridge encoge más sus predicciones hacia la media. Con los dos lectores, en la validación espacial el boosting pasa de 127,0 a 124,4 M y la red de 137,6 a 133,7 M.

## 7. Predictores más importantes

La gráfica es la importancia por ganancia del boosting sin BETO, ajustado una vez con todos los datos de entrenamiento. No tenemos esa medición para el boosting con BETO, porque obtenerla exige reajustarlo. El texto resumido por el ridge concentra el 22 % de la ganancia en apartamentos: describe calidad, acabados y amenidades que las columnas no recogen. Le sigue el área (10 %), que se asocia directamente con el precio. El tercero es el largo de la descripción (6 %): los anuncios más largos se asocian con inmuebles más caros. El boosting usa 63 hojas por árbol, profundidad máxima 16, tasa de aprendizaje 0,03 y 1.500 árboles; la búsqueda de hiperparámetros no mejoró el tablero, así que no se sintonizó.

## 8. Lo que no funcionó

OpenStreetMap empeoró la validación espacial (+7,9 a +8,6 M). Pensábamos que en Chapinero todo queda cerca, pero los datos no lo respaldan: en el test las distancias varían entre 0,6 y 1,5 veces lo que varían en entrenamiento (rango intercuartílico). Nuestra hipótesis es que la relación entre amenidades y precio cambia de una zona a otra; no la medimos. El modelo local entrena con muchos menos anuncios y empeora la espacial (+8,9 a +60,3 M). El holdout ponderado son anuncios de entrenamiento vecinos a Chapinero, ponderados para parecerse al test. Darles más peso a los caros cambia el nivel y deja igual el orden: el holdout ponderado empeora (+0,7 a +4,1 M). El apilamiento aprende el tope de precios del holdout, que no existe en el test (+15,1 a +23,2 M). Mezclar con el bosque deja el holdout ponderado igual: sus errores son casi los del boosting (correlación 0,98).

## 9. Sobrepredicción

Con las predicciones de la validación espacial, la mezcla sobrepredice el 67 % de los inmuebles, con un sesgo de +39,4 M. Es lo esperable al apuntar al cuantil 0,70. El panel por precio pedido muestra que sobrepredice el 92 % de los de menos de 400 M y el 18 % de los de más de 1.200 M; ese patrón es en parte mecánico, porque cualquier modelo acerca sus predicciones a la media. Para la calibración sirve el panel por precio predicho, que es lo que ve la empresa: en cada rango el sesgo es positivo, de +9 a +61 M, y la sobrepredicción media crece con la predicción. El error es mayor en las casas (MAE 175,7 M frente a 108,3 M en apartamentos).

## 10. Pérdida asimétrica

Si sobrepredecir cuesta k veces lo que subpredecir, el MAE deja de ser la medida correcta. Con k = 1, el MAE, la mezcla queda en el puesto 4 de 13: el boosting en cuantil 0,50 tiene menos error espacial (117,9 M) porque la validación espacial premia bajar el nivel. Kaggle se comportó distinto: el cuantil 0,70 dio 160,9 M y el 0,65 161,7 M; Chapinero es más caro que las zonas con que se valida. Con k = 3 gana el boosting en cuantil 0,20 (174 M) y la mezcla cae al puesto 6 (289 M), como predice la teoría: con costo k veces mayor, el cuantil óptimo es 1/(1+k). Los cuantiles bajos son el boosting sin BETO.

## 11. Regla de compra

Simulamos una regla: la empresa ofrece la predicción con un descuento y compra si la oferta alcanza el precio pedido; la diferencia es sobrepago. La oferta solo alcanza el precio cuando el modelo sobreestima, así que toda compra es un sobrepago frente al precio pedido. Con la predicción de la mezcla, sin descuento, se cierra el 67 % de las ofertas y la empresa sobrepaga el 18 % del capital. A igual número de compras, bajar el cuantil sobrepaga menos que descontar. Ninguna regla baja del 10 % del capital.

## 12. Recomendación

Sí recomendamos usar el modelo para comprar, con una oferta conservadora y un avalúo. La oferta es el cuantil 0,20: un precio que el 80 % de los inmuebles parecidos supera. En apartamentos se cierra el 32 % de las ofertas y la empresa sobrepaga el 10,9 % del capital; al ofertar la predicción de Kaggle (cuantil 0,70) en apartamentos sobrepaga el 16,6 %. En casas sobrepaga el 14,9 %, y en apartamentos con oferta desde 900 M, el 12,0 %, con 124 M por compra frente a 53 M por debajo de ese valor; por eso los dejamos fuera. La regla se simuló en zonas de Bogotá que el modelo no vio, con el boosting sin BETO, y se escogió con esas mismas predicciones, así que es algo optimista. Antes de cerrar, un avalúo fija el precio. Riesgos: los precios son de oferta, los datos son de 2019 a 2021 y en el holdout vecino a Chapinero el sesgo cambia de signo (−53,3 M).

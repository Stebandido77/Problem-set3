# Notas del orador: compare_equipo_06

Comparación de modelos y de validaciones. Neislen García y Esteban Labastidas, Equipo 06. Una sección por diapositiva de contenido, en orden. Generado por `scripts/36_build_presentations.py`; las cifras salen de `views/`.

## 1. Vista general

Predecimos el precio de oferta, lo que pide el vendedor, de vivienda en Chapinero, entrenando con el resto de Bogotá. El mejor modelo, la mezcla 203, quedó primera en los dos tableros de Kaggle: 145,7 M en el público y 148,6 M en el privado, 2,2 M menos que el segundo. El MAE es el error promedio en pesos. El boosting son muchos árboles pequeños, cada uno corrige el error del anterior; la red neuronal puede predecir por encima de los precios que vio; la mezcla pondera 70 % el boosting y 30 % la red. La validación aleatoria prueba en anuncios al azar de las mismas zonas que el modelo ya vio; la espacial, en zonas que no vio, y da 124,8 M.

## 2. Tabla por familia

Cada fila es la mejor especificación de su familia con la receta común. La columna de repetidos es la validación aleatoria con las copias de un mismo inmueble en la misma partición. El holdout ponderado son anuncios de entrenamiento vecinos a Chapinero, ponderados para parecerse al test. El boosting gana la espacial y el tablero; la red gana el holdout ponderado y en el tablero queda en 157,1 M. El elastic net puro no tiene envío; los envíos 49 y 54 son híbridos: un elastic net con un boosting sobre sus residuos. Para la 203 no reportamos el ponderado, porque BETO se afinó con precios de ese holdout.

## 3. Predicción máxima

Los precios de entrenamiento van de 300 a 1.650 M. La etiqueta de cada barra cuenta cuántas de las 10.286 predicciones del test pasan de 1.650 M. El CART (1.580 M) y el bosque (1.590 M) promedian precios vistos y se quedan bajo el tope. El boosting suma las salidas de muchos árboles, y esa suma puede pasar el tope: la 127 llega a 1.730 M con 7 predicciones encima. La red llega a 2.002 M, con 167 encima. La mezcla 203 tiene 20 predicciones sobre el tope, con máximo de 1.745 M. Los sondeos del tablero indican que el test tampoco pasa de 1.650 M, así que esas predicciones deben recortarse al tope.

## 4. Por qué aporta la red

La figura mezcla, sin reentrenar, el boosting 127 con cada familia. Con la red, la validación espacial cambia +0,7 M, que es ruido, y el holdout ponderado −3,8 M. En el tablero, del mejor boosting solo (152,1 M) a la primera mezcla (148,2 M). Al separar por decil de precio, la mezcla 203 le gana al boosting solo en los deciles 9 y 10 (desde 891 M), por 2,4 y 9,0 M, y pierde hasta 3,9 M en los demás. En el decil más caro, el boosting subpredice −215 M y la red −163 M. La correlación de errores entre los dos es 0,90 en la espacial y 0,96 en el holdout; la del bosque con el boosting es 0,98, más alta que la de la red (0,96).

## 5. Receta común

Cada flecha va de la versión inicial de una familia (número a la derecha) a su mejor versión (número a la izquierda); los números son filas de la tabla de resultados. La receta: un modelo por tipo, el precio en pesos con el cuantil 0,70 como objetivo (un precio que el 70 % de los parecidos no supera), el estrato y el valor del terreno, y el ridge del texto. En el holdout ponderado todas ganan entre 12,3 y 44,8 M; en la espacial ninguna pasa al mejor boosting.

## 6. BETO en cada modelo

El ridge sobre TF-IDF es una regresión que le pone un peso a cada palabra del anuncio. BETO es un modelo de lenguaje en español, afinado aquí para estimar el precio desde el texto. Lo sumamos al ridge en los dos miembros de la mezcla. En la validación espacial el boosting pasa de 127,0 a 124,4 M, la red de 137,6 a 133,7 M y la mezcla de 127,6 a 124,8 M. En el tablero público la mezcla baja 2,6 M. Las predicciones de los dos miembros con BETO tienen una correlación de 0,96 en el test.

## 7. Validación contra Kaggle

Comparamos los cuatro esquemas sobre los mismos 8 envíos. El Spearman mide si un esquema ordena los envíos como el tablero público (1 es el mismo orden): aleatoria 0,76, repetidos juntos 0,74, espacial 0,93 y holdout ponderado 0,88. Con el mismo conjunto, la espacial es la que mejor ordena y las dos aleatorias quedan atrás, muy cerca entre sí. Todos los esquemas son optimistas; la espacial, en 17,6 M en promedio. ¿Escogen la aleatoria y la espacial el mismo mejor modelo? Entre estos envíos, sí: las dos escogen la mezcla 152, que también es la mejor del tablero entre ellos. Entre las 58 especificaciones con las dos validaciones, escogen la misma receta de boosting con distinta regla de anuncios repetidos (126 y 127). El puntaje privado solo está registrado para el modelo final, así que no hay comparación contra el privado.

## 8. Anuncios repetidos

Muchos anuncios del entrenamiento son el mismo inmueble publicado varias veces. Si una copia queda en el ajuste y otra en la validación, el modelo acierta por memoria. Al poner las copias en la misma partición, el MAE aleatorio sube entre 5,3 y 21,2 M; en la mezcla sin BETO, de 95,5 a 112,2 M. Aun así, todos los modelos quedan a más de 32 M de su puntaje en el tablero.

## 9. Nivel en la espacial

La validación espacial prefiere modelos de nivel más bajo. En la escalera de cuantiles, menos sesgo da menos MAE espacial, mientras el holdout ponderado casi no cambia. El cuantil 0,65 gana 6,5 M en la espacial y en el tablero queda 0,7 M detrás del 0,70, una diferencia de ruido. Por eso usamos la espacial para ordenar modelos de nivel parecido, y el holdout y el tablero para el nivel.

## 10. Lo que no funcionó

Ninguna de las 16 variantes mejoró a la vez la espacial y el holdout ponderado: la esquina de abajo a la izquierda está vacía. OpenStreetMap empeora la espacial (+7,9 a +8,6 M). En el test sus distancias varían entre 0,6 y 1,5 veces lo que varían en entrenamiento, así que la idea de que en Chapinero todo queda cerca no se sostiene; por qué empeora no está medido.

## 11. Intercambios

El boosting tiene el menor error en zonas que no vio y subpredice los inmuebles más caros. La red aporta en los dos deciles más caros y sola queda en 157,1 M en el tablero. La regresión lineal es la más fácil de leer y queda en 180,5 M. BETO mejora a los dos miembros de la mezcla, requiere GPU y deja optimista el holdout ponderado. Para comprar, el orden cambia: si sobrepagar cuesta el triple, la mezcla cae al puesto 6 y gana un boosting en cuantil 0,20.

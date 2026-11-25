# Notas del orador: compare_equipo_06

Comparación de modelos y de validaciones. Una sección por diapositiva de contenido, en orden. Generado por `scripts/36_build_presentations.py`; las cifras salen de `views/`.

## 1. Vista general

Predecimos el precio de oferta, lo que pide el vendedor, de vivienda en Chapinero, entrenando con el resto de Bogotá. El mejor modelo, la mezcla 203, combina un boosting (70 %) y una red neuronal (30 %), los dos con dos lectores del texto: un ridge sobre TF-IDF y BETO. Quedó primera en los dos tableros de Kaggle: 145,7 M en el público y 148,6 M en el privado, 2,2 M menos que el segundo. El MAE es el error promedio en pesos. En nuestra validación espacial, que prueba en zonas que el modelo no vio, da 124,8 M. Esta presentación explica por qué ganan el boosting y la red, y cómo escogimos entre esquemas de validación.

## 2. Tabla por familia

Cada fila es la mejor especificación de su familia con la receta común: un modelo por tipo, precio en pesos, cuantil 0,70, estrato y valor del terreno, y el ridge del texto. Las columnas son tres formas de validar y el puntaje público. La aleatoria con gemelos prueba en anuncios al azar, de zonas que el modelo ya vio, pero mantiene juntas las copias del mismo inmueble. La espacial deja zonas completas fuera. El ponderado es un holdout al lado de Chapinero, ponderado como el test. El boosting gana en la espacial y la red en el ponderado, pero en el tablero público la red sola (157,1 M) quedó detrás del boosting (152,1 M). La elastic net no se subió a Kaggle. Para la 203 no reportamos el ponderado: BETO se afinó con precios de ese holdout y la cifra sería optimista.

## 3. Rango de predicción

Los precios de entrenamiento están entre 300 y 1.650 millones. Un árbol predice promedios o cuantiles de lo que vio, así que el CART y el bosque nunca pasan ese tope y el boosting apenas lo roza. La red sí puede pasarlo. La regresión lineal no extrapola en precio porque su receta la recorta al rango del train, pero sus splines de latitud y longitud se estiman casi sin datos de Chapinero: sin coordenadas, su error espacial baja de 164,1 a 156,3 M, y en el tablero público fue de las peores (180,5 M).

## 4. Por qué la red y no el bosque

Calculamos, sin reentrenar, una mezcla 70/30 del boosting 127 con cada familia, sobre las predicciones fuera de muestra. La correlación de errores con el boosting en el holdout nororiental es 0,98 para el bosque y 0,96 para la red. Con la red el ponderado cambia −3,8 M; con el bosque, 0,0 M (negativo es mejora). La red es la única que es buena sola y a la vez se equivoca en otros inmuebles: llega a la cola alta, donde el boosting subpredice. Las lineales y el kNN son más distintos, pero tan malos solos que empeoran la mezcla.

## 5. Receta común

Antes, cada familia usaba el logaritmo del precio y los mismos predictores. Trasladamos la receta del boosting a todas: un modelo por tipo, el precio en pesos con el cuantil 0,70 como objetivo, el estrato y el valor del terreno de la manzana, y el texto completo resumido por el ridge. Todas mejoran en el holdout ponderado, entre 12,3 y 44,8 M. En la espacial la mejora es menor y el boosting sigue adelante: la ventaja es del algoritmo, no solo de la receta.

## 6. BETO en cada modelo

El último paso fue BETO, un modelo de lenguaje en español que afinamos para estimar el precio desde el texto, como segundo lector junto al ridge. Solo lo probamos en los dos miembros de la mezcla. En la validación espacial, el boosting pasa de 127,0 a 124,4 M, la red de 137,6 a 133,7 M y la mezcla de 127,6 a 124,8 M. En el tablero público, la mezcla bajó 2,6 M. BETO mejora a los dos miembros, pero no los vuelve más distintos: su mezcla gana porque cada uno es mejor, no por diversidad.

## 7. Validación contra Kaggle

Cada punto es un envío con puntaje público; la línea punteada es la igualdad. Todos los esquemas son optimistas; Chapinero es más caro y distinto que el resto de Bogotá; la espacial, en promedio 26,5 M. El Spearman mide si un esquema ordena los envíos como el tablero (1 es el mismo orden): aleatoria simple 0,44, con gemelos 0,78, espacial 0,74 y nororiental ponderado 0,83. Cada esquema tiene un número distinto de envíos, así que la comparación es aproximada. La aleatoria simple es la que peor ordena.

## 8. Gemelos

Muchos anuncios del train son el mismo inmueble publicado varias veces. Si una copia queda en el ajuste y otra en la validación, el modelo acierta por memoria. Al mantener los gemelos juntos, el error de la mezcla 152 sube de 95,5 a 112,2 M. En Chapinero casi no hay copias de inmuebles del train, así que la versión con gemelos es la relevante. La 203 no tiene validación aleatoria calculada.

## 9. Nivel en la espacial

La espacial ordena razonablemente, pero prefiere modelos con un nivel más bajo que el que premia el tablero: el cuantil 0,60 le gana al 0,70 en la espacial y no en el ponderado, y en el tablero el 0,65 (envío 53) quedó detrás del 0,70 (65), aunque por menos de 1 M. Por eso usamos la espacial para ordenar modelos de nivel parecido, y el ponderado y el tablero para el nivel.

## 10. Lo que no funcionó

Ninguna de estas 16 ideas mejoró a la vez la espacial y el ponderado; la esquina de abajo a la izquierda está vacía. OpenStreetMap empeora la espacial (+7,9 a +8,6 M): en Chapinero casi todo queda cerca. El modelo local pierde demasiados datos. El apilamiento aprende el tope del holdout, que no existe en el test. Balancear hacia los caros cambia la espacial por nivel, no por orden. La monotonía y el kNN no ganan en los dos criterios.

## 11. Intercambios

Cada familia gana algo y paga algo. El boosting ordena mejor en zonas que no vio, pero no pasa el tope del train. La red sí lo pasa, pero sola es peor en el tablero (157,1 M). La regresión lineal es fácil de leer y viaja mal a Chapinero por sus coordenadas. BETO lee mejor el texto, pero hay que afinarlo en GPU y se entrenó con particiones propias, así que su ponderado no es limpio. Y el modelo para el tablero no es el de comprar: con costo de sobrepago triple, la 203 cae al puesto 6. Escogimos con varios esquemas y usamos el tablero como juez final.

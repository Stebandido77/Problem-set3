# Notas del orador: compare_equipo_06

Comparación de modelos y de validaciones. Una sección por diapositiva de contenido, en orden. Generado por `scripts/36_build_presentations.py`; las cifras salen de `views/`.

## 1. Vista general

Predecimos el precio de oferta de vivienda en Chapinero entrenando con el resto de Bogotá. El mejor modelo combina un boosting cuantílico (70 %) con una red neuronal cuantílica (30 %), un modelo para apartamentos y otro para casas. Las tres cifras: el error público de Kaggle (148,2 M), la mejora frente a predecir la mediana (61 %) y el error en validación espacial (127,6 M). El mejor boosting solo obtiene 152,1 M; la mezcla gana 3,8 M. Las seis familias se entrenan con la misma receta: por tipo, precio en nivel, cuantil 0,70, estrato y valor del terreno, y el texto completo del anuncio.

## 2. Tabla por familia

Cada fila es la mejor especificación de su familia con la receta común. Las columnas son tres formas de validar y el puntaje público. La aleatoria con gemelos mantiene anuncios repetidos del mismo inmueble en la misma partición; la espacial deja zonas completas fuera; el ponderado es un holdout al lado de Chapinero, ponderado como el test. La elastic net no se subió a Kaggle. La red queda primera en el ponderado, pero en Kaggle fue peor que el boosting: ese holdout la favorece.

## 3. Receta común

Antes, cada familia usaba logaritmo del precio y los mismos predictores. La receta del boosting consiste en: un modelo por tipo de inmueble, el precio en nivel con el cuantil 0,70 como objetivo, el estrato y el valor del terreno de la manzana, y el texto completo resumido por un ridge fuera de muestra. Todas las familias mejoran en el holdout ponderado, entre 12,3 y 44,8 M. En la espacial la mejora es menor y el boosting sigue adelante: la ventaja es del algoritmo, no solo de la receta.

## 4. Rango de predicción

Los precios de entrenamiento están entre 300 y 1.650 millones: un árbol predice promedios o cuantiles de lo que vio, así que el CART y el bosque nunca pasan ese tope, y el boosting apenas lo roza. La red sí extrapola. La regresión lineal no extrapola en precio, porque su receta la recorta al rango del train, pero sus splines de latitud y longitud se estiman casi sin datos de Chapinero: sin coordenadas su error espacial baja de 164,1 a 156,3 M (fila 130), y en Kaggle fue de las peores (180,5 M).

## 5. Mezcla

Calculamos, sin reentrenar, una mezcla 70/30 del boosting 127 con cada familia sobre las predicciones fuera de muestra. La correlación de errores con el boosting en el holdout nororiental es 0,98 para el bosque y 0,96 para la red. Con la red el ponderado cambia −3,8 M; con el bosque, 0,0 M (negativo = mejora). La diferencia decisiva es que la red sola es buena y sí pasa el tope de 1.650 M, donde el boosting subpredice los inmuebles caros. Las lineales y el kNN son más distintos, pero tan malos solos que empeoran la mezcla.

## 6. Resultados negativos

Ninguna de estas 16 ideas mejoró a la vez la espacial y el ponderado. OpenStreetMap empeora la validación espacial: las distancias a amenidades no viajan bien a Chapinero, donde casi todo está cerca. Un modelo local solo con la zona parecida pierde demasiados datos. El apilamiento aprende a corregir el tope artificial del holdout, algo que no se puede llevar al test. Balancear hacia los caros baja la espacial por nivel, no por orden. La monotonía y el kNN no ganan en los dos criterios.

## 7. Validación contra Kaggle

Cada punto es un envío con puntaje público; la línea es la igualdad. Todos los esquemas son optimistas, porque Chapinero es más caro y distinto del resto de Bogotá. Spearman con Kaggle sin el ingenuo: aleatoria simple 0,44, con gemelos 0,78, espacial 0,69, nororiental ponderado 0,83 (sin ponderar llega a 0,89). El holdout nororiental ordena mejor y, ponderado como el test, acierta el nivel en boosting y bosque; pero es optimista con la red, la lineal, el CART y los híbridos. La aleatoria simple es la que peor ordena.

## 8. Gemelos

Muchos anuncios del train son el mismo inmueble publicado varias veces. Si una copia queda en el ajuste y otra en la validación, el modelo acierta por memoria. Al mantener los gemelos juntos el error de la mezcla 152 sube de 95,5 a 112,2 M. En Chapinero casi no hay copias de inmuebles del train, así que la versión con gemelos es la relevante.

## 9. Nivel en la espacial

La espacial ordena razonablemente, pero prefiere modelos con un nivel más bajo que el que premia Kaggle: el cuantil 0,60 le gana al 0,70 en la espacial y no en el ponderado, y en el tablero el 0,65 (envío 53) quedó detrás del 0,70 (65). Por eso no escogimos la 186, que mejora la espacial casi solo por bajar el nivel. Usamos la espacial para ordenar modelos de nivel parecido y el ponderado y Kaggle para el nivel.

## 10. Pérdida asimétrica

Con el error absoluto, la mezcla 152 queda en el puesto 5 de la validación espacial. Si sobrepredecir cuesta dos o tres veces más que subpredecir, el ranking premia a los modelos que apuntan más bajo: con costo triple gana el 190 y la 152 cae al puesto 6. El modelo para Kaggle y el modelo para comprar no tienen por qué ser el mismo: esa es la pregunta de la otra presentación.

## 11. Conclusión

El boosting cuantílico por tipo es la base porque ordena bien los precios en zonas que no vio. La receta común ayuda a todas las familias, pero la brecha con el boosting se mantiene, así que la ventaja es del algoritmo. La red entra a la mezcla porque llega a la cola alta. Para escoger modelos no confiamos en un solo esquema: la aleatoria simple se engaña con los anuncios repetidos, la espacial premia bajar el nivel y el holdout ponderado favorece a las redes. Por eso escogimos con los tres y usamos Kaggle como juez final.

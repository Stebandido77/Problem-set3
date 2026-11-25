# Notas del orador: best_equipo_06

El mejor modelo y la decisión de compra. Una sección por diapositiva de contenido, en orden. Generado por `scripts/36_build_presentations.py`; las cifras salen de `views/`.

## 1. Resultado

Empezamos por el resultado. La mezcla 203 quedó en el primer puesto de los dos tableros de Kaggle. El MAE es el error promedio, en pesos, entre el precio predicho y el observado. En el tablero público, que se calcula con el 20 % del test, fue de 145,7 millones. En el privado, con el 80 % restante y el que define la competencia, fue de 148,6 M, 2,2 M menos que el segundo equipo (150,8 M). Del público al privado el error subió 2,9 M (2,0 %): el modelo se sostuvo y conservó el puesto. Nuestra validación espacial, que prueba el modelo en zonas de Bogotá que no vio al entrenar, da 124,8 M: es optimista frente a los dos tableros; Chapinero es más caro y distinto que cualquier zona del train. Los precios son de oferta, lo que pide el vendedor. La pregunta de la presentación es si este modelo sirve para comprar.

## 2. Qué es el modelo

El modelo tiene dos partes que leen lo mismo. Ve el inmueble (tipo, alcobas, baños y área, también la recuperada de anuncios del mismo punto), la ubicación (estrato y valor del terreno de la manzana, distancias a cuatro hitos) y el texto del anuncio. El texto entra por dos lectores. El ridge sobre TF-IDF es una regresión que le pone un peso a cada palabra; BETO es un modelo de lenguaje en español que afinamos para estimar el precio desde el texto. Los dos lectores se ajustan fuera de muestra, así que su estimación de un anuncio nunca usa el precio de ese anuncio. Con todo eso se entrenan un boosting, muchos árboles pequeños en el que cada uno corrige el error del anterior, y una red neuronal, que puede predecir por encima de los precios que vio. Los dos apuntan al cuantil 0,70, un precio que el 70 % de los inmuebles parecidos no supera, y se entrenan por separado para apartamentos y casas. La predicción final es una mezcla: 70 % del boosting y 30 % de la red.

## 3. Progresión

Este es el camino en el tablero público, en seis pasos. Del ingenuo, que predice la mediana de Bogotá para todo (377,1 M), al modelo final (145,7 M) el error baja 61 %. Casi toda la ganancia está en los tres primeros pasos: el primer boosting y corregir el nivel. Los tres últimos suman menos, pero cada uno supera los 2 M que tomamos como ruido del tablero.

## 4. Pasos 1 a 3

El primer boosting con las variables del anuncio y las coordenadas baja el error de 377,1 a 180,1 M. Luego vimos que el modelo subpredecía: Chapinero es más caro que el resto de Bogotá. Escalar a mano el envío 10 por 1,15 lo bajó a 171,7 M; esa versión es una calibración con el tablero, no un modelo, y solo nos sirvió para confirmar el problema de nivel. La solución dentro del modelo fue entrenar uno por tipo de inmueble, con el precio en pesos y no en logaritmo, y apuntar al cuantil 0,70. Ese cuantil lo escogimos con la retroalimentación del tablero: el 0,65 (envío 53) dio 161,7 M y el 0,70 (envío 65) 160,9 M, una diferencia que está dentro del ruido.

## 5. Pasos 4 a 6

El cuarto paso lee el texto completo del anuncio con un ridge sobre TF-IDF y agrega el estrato (Planeación) y el valor del terreno (Catastro), con 10 semillas: el error baja 8,9 M, y de eso 7,9 M aparecen solo con el texto (envío 75, 153,1 M). El quinto mezcla el boosting con una red neuronal, que se equivoca en inmuebles distintos y puede predecir por encima del tope de 1.650 M del train: baja 3,8 M. El sexto agrega BETO como segundo lector del texto, en el boosting y en la red: baja 2,6 M, hasta 145,7 M. Las dos últimas ganancias son pequeñas, pero las dos superan los 2 M de ruido.

## 6. Qué aportó BETO

Comparamos los dos lectores del texto solos, fuera de muestra, con las mismas particiones espaciales. BETO se equivoca en promedio 148,5 M y el ridge 169,1 M. Por decil de precio, el ridge acierta más en los precios del medio (decil 6: 87 contra 133 M) y BETO en los extremos (decil 10: 341 contra 537 M). El ridge encoge mucho las predicciones hacia la media; BETO, menos. Por eso conviene tener los dos: en la validación espacial el boosting pasa de 127,0 a 124,4 M y la red de 137,6 a 133,7 M. En el tablero público, la mezcla pasa de 148,2 a 145,7 M.

## 7. Lo que no funcionó

También probamos ideas que no sirvieron, y explican el camino. OpenStreetMap: las distancias y conteos de amenidades empeoran la validación espacial (+7,9 a +8,6 M); en Chapinero casi todo queda cerca y la variable distingue poco. El modelo local, entrenado solo con anuncios cercanos o parecidos a Chapinero, se queda con muchos menos datos y empeora la espacial (+8,9 a +60,3 M). Darles más peso a los inmuebles caros, con cuantiles de 0,55 a 0,65, cambia la espacial sobre todo por el nivel, y en el holdout ponderado empeora (+0,7 a +4,1 M). El apilamiento aprende a corregir el tope del holdout, que no existe en el test (+15,1 a +23,2 M en el ponderado). Mezclar con el bosque no ayuda: se equivoca en los mismos inmuebles que el boosting (correlación de errores 0,98).

## 8. Dónde se equivoca

Usamos las predicciones de la validación espacial: zonas que el modelo no vio al entrenar. La mezcla 203 sobrepredice el 67 % de los inmuebles, con un sesgo de +39,4 M. Sobrepredice el 92 % de los de menos de 400 M y solo el 18 % de los de más de 1.200 M, donde el sesgo es de −205 M. El error es mayor en las casas (MAE 175,7 M frente a 108,3 M en apartamentos) y cuando falta el área (141,7 frente a 117,0 M). Para comprar importa el panel del precio predicho, que es lo único que ve la empresa: cuanto más alta la predicción, mayor la sobrepredicción media.

## 9. Sobre contra subpredicción

Si sobrepredecir cuesta más que subpredecir, el MAE no es la medida correcta. Recalculamos el costo de cada modelo cuando la sobrepredicción cuesta k veces la subpredicción. Con k = 1 es el MAE y la 203 queda en el puesto 4 de 13. Con k = 3 gana el boosting 190, que apunta al cuantil 0,20 (174 M), y la 203 cae al puesto 6 (289 M). Es lo que dice la teoría: con costo k veces mayor, el cuantil óptimo es 1/(1+k), es decir 0,25. Los cuantiles bajos son el boosting sin BETO; no reentrenamos la mezcla 203 en cuantiles bajos.

## 10. Regla de compra

Simulamos una regla: la empresa ofrece la predicción con un descuento y compra si la oferta alcanza el precio pedido; la diferencia es sobrepago. Hay selección adversa: la oferta solo alcanza el precio cuando el modelo sobreestima. Si ofrece lo que predice la 203, compra el 67 % de los anuncios y sobrepaga el 18 % del capital. Con el cuantil 0,20 compra el 33 % y sobrepaga el 12 %. A igual número de compras, bajar el cuantil sobrepaga menos que descontar: la 203 con 15 % de descuento compra el 34 % con 14 %. Ninguna regla baja del 10 % del capital.

## 11. Regla por segmento

Con el cuantil 0,20, en apartamentos la empresa compra el 32 % y sobrepaga el 10,9 % del capital; en casas, el 14,9 %. Que falte el área cambia poco el sobrepago. La regla y el segmento se escogieron con las mismas predicciones espaciales con que se reportan, así que estas cifras son algo optimistas.

## 12. Recomendación

¿Se puede confiar en el modelo para comprar? Para fijar el precio, no: ofrecer su predicción compra justo los inmuebles que sobreestima. Para escoger candidatos, sí, con una regla conservadora: el cuantil 0,20, solo en apartamentos y con una tasación física antes de cerrar. No lo usaríamos en casas ni con predicciones de más de 900 M, donde la sobrepredicción media pasa de 204 M. Riesgos: los precios son de oferta, no de cierre, así que el sobrepago real puede ser mayor; falta el área en el 26 % de los anuncios; los datos son de 2019 a 2021; y el nivel en Chapinero es incierto: en el holdout junto a Chapinero el sesgo cambia de signo (−53,3 M).

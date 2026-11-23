# Notas del orador: best_equipo_06

El mejor modelo y la decisión de compra. Una sección por diapositiva de contenido, en orden. Generado por `scripts/36_build_presentations.py`; las cifras salen de `views/`.

## 1. Resultado

Empezamos por el resultado. La pregunta guía es si podemos confiar en el modelo lo suficiente para comprar vivienda en Chapinero. El mejor modelo (la mezcla 152) tiene un error medio de 148,2 millones en el tablero público, 61 % menos que predecir la mediana (377,1 M). En validación espacial, que deja zonas completas de Bogotá por fuera, el error es 127,6 M. Pero si la empresa ofrece lo que predice el modelo, solo compra cuando el modelo sobreestima, y paga de más: cierra el 67 % de los anuncios con un sobrepago del 18 % del capital. Con el cuantil 0,20 cierra el 33 % con 12 %. La respuesta corta: sirve para filtrar candidatos con una regla conservadora, no para fijar el precio.

## 2. El problema

Los puntos grises son los anuncios de entrenamiento; los rojos, los de Chapinero que hay que predecir. El modelo casi no ve Chapinero al entrenar. Además, Chapinero es distinto: casi todo son apartamentos (casas: 24 % en el train y 3 % en Chapinero), el estrato es más alto y el terreno cuesta 1,9 veces más. Por eso validamos dejando zonas enteras por fuera.

## 3. Qué es el modelo

El boosting aprende bien el orden de los precios, pero un árbol no predice por encima de lo que vio, y el train está recortado en 1.650 M. La red puede extrapolar y corrige en parte la subpredicción de los inmuebles caros. Los dos apuntan al cuantil 0,70 del precio y no a la mediana: Chapinero es más caro que el resto de Bogotá, y apuntar alto compensa ese salto de nivel. Cada uno se entrena por separado para apartamentos y para casas; el boosting promedia 10 semillas y aplica la regla de anuncios repetidos. Qué ve el modelo, en detalle: tipo, alcobas, baños y área; área y baños recuperados de anuncios del mismo punto; estrato y valor del terreno de la manzana (Catastro); distancias a cuatro hitos; 18 señales del texto y un ridge TF-IDF fuera de muestra sobre el texto completo. Quitamos latitud y longitud crudas porque no viajan a una zona que no se vio: la ubicación entra por el estrato, el valor del terreno y las distancias.

## 4. Progresión

De dónde salió cada mejora, en puntaje público. Del ingenuo al primer boosting. Subir el nivel (primero escalando, luego con la pérdida cuantílica) y separar apartamentos y casas. El texto completo del anuncio fue el último salto grande de un modelo solo. La mezcla con la red dio los últimos 3,8 millones frente al mejor boosting (152,1 M). El envío 32 es una calibración con el tablero, no un modelo, y lo mostramos solo para explicar el efecto del nivel.

## 5. Importancia

Importancia por ganancia del boosting, ajustado una vez con todo el train. El ridge sobre el texto completo resume en un número lo que dicen el título y la descripción: 22 % de la ganancia en apartamentos. Le siguen el área (también la recuperada de anuncios del mismo punto), el largo de la descripción y la ubicación a través de distancias y valor del terreno. Es una importancia de un ajuste con una semilla: sirve para ordenar, no para medir efectos causales.

## 6. Dónde se equivoca

Usamos las predicciones fuera de muestra de la validación espacial. En total sobrepredice el 67 % de los inmuebles, con un sesgo de 41,7 millones. Sobrepredice el 92 % de los de menos de 400 M y solo el 18 % de los de más de 1.200 M. El error crece con el precio y es mayor en las casas y cuando falta el área. Para comprar importa el precio predicho, que es lo único que ve la empresa: cuanto más alta la predicción, más grande la sobrepredicción media.

## 7. Pérdida asimétrica

Recalculamos el costo medio de cada modelo cuando sobrepredecir cuesta dos o tres veces más que subpredecir. Entre familias el orden casi no cambia; lo que cambia es el nivel: ganan los cuantiles bajos, como predice la teoría (con costo k veces mayor, el cuantil óptimo es 1/(1+k): 0,33 y 0,25). Con costo triple gana el 190 (cuantil 0,20); la 152 de Kaggle queda en el puesto 6. El modelo que gana en Kaggle no es el que conviene para comprar.

## 8. Regla de compra

Simulamos: la empresa ofrece la predicción con un descuento; si la oferta alcanza el precio pedido, compra y la diferencia es sobrepago. Hay selección adversa: solo se compra cuando el modelo sobreestima. Cada curva es un cuantil y cada punto un descuento de 0 a 20 %. Ofrecer lo que predice el modelo de Kaggle cierra el 67 % con 18 % de sobrepago. A igual número de compras, bajar el cuantil sobrepaga menos que descontar: el cuantil 0,20 cierra el 33 % con 12 %, y la 152 con 16 % de descuento cierra el 33 % con 14 %. Ninguna regla baja del 10,0 % del capital.

## 9. Regla por segmento

Con el cuantil 0,20 la empresa compra uno de cada tres anuncios y el sobrepago cae frente al modelo de Kaggle. En apartamentos el sobrepago es 10,9 % del capital; en casas, 14,9 %: por eso no recomendamos usar el modelo en casas. Que falte el área cambia poco la tasa de sobrepago, pero sí el error.

## 10. Recomendación

El modelo ordena bien los inmuebles y su error está muy por debajo del ingenuo, pero ofrecer su predicción compra justo los inmuebles que sobreestima. Lo usaríamos como filtro: el cuantil 0,20 marca candidatos en apartamentos (32 % de ellos, con 10,9 % de sobrepago frente a 18,3 % ofreciendo la predicción de Kaggle), y una tasación física decide el precio. No lo usaríamos en casas ni con predicciones de más de 900 millones, donde la sobrepredicción media pasa de 204 millones. Antes de comprometer capital, revisaríamos la regla con precios de cierre.

## 11. Riesgos

Primero, medimos el sobrepago contra el precio pedido, no contra el de cierre: si los inmuebles se venden por menos, el sobrepago real es mayor. Segundo, falta el área en el 26 % de los anuncios, y ahí el error sube de 119,0 a 145,6 M. Tercero, los anuncios son de 2019 a 2021. Cuarto, la validación espacial es optimista en nivel frente a Kaggle (127,6 frente a 148,2 M), y en el holdout junto a Chapinero el sesgo cambia de signo: el nivel correcto en Chapinero es incierto.

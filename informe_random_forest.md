# Resultados para el informe: Random Forest

## Motivación y momento de predicción

Una aseguradora puede usar la probabilidad estimada de fraude para priorizar
qué siniestros revisar. Se asume una decisión posterior a la declaración del
siniestro, con los importes reclamados y datos del incidente ya registrados.
El modelo no establece culpabilidad ni sustituye una investigación. La fuente
indicada es [Auto Insurance Claims Data (Kaggle)](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data).
La disponibilidad real de cada variable al momento de decidir debe confirmarse
en el informe grupal.

## Datos y metodología

El archivo `insurance_claims_limpio.csv` contiene 1.000 siniestros, 35
predictores y la variable binaria `fraud_reported`; 247 casos (24,7 %) tienen
etiqueta positiva. Se reservaron 200 casos (49 positivos) para prueba con una
partición aleatoria estratificada de 80/20 y semilla 42. El CSV limpio no
incluye fecha completa, aunque el original sí contiene `incident_date`, que
permitiría revisar una partición temporal.

Las variables numéricas se imputaron con la mediana y las categóricas con la
moda. En el origen, 91 casos de `authorities_contacted` tienen la categoría
`None`; en el CSV limpio se convirtieron en vacíos y el modelo actual los
imputa con la moda. Esto debe corregirse en una nueva versión común de datos
antes de reentrenar todos los modelos. Las categorías se codificaron con one-hot y las desconocidas en prueba
se toleran sin error. Toda transformación se estimó solo con entrenamiento,
también dentro de cada pliegue de validación. Se compararon ocho combinaciones
de profundidad (`None` u 8), mínimo de observaciones por hoja (1 o 5) y peso
de clases (`None` o `balanced_subsample`). Se usaron 150 árboles, semilla 42,
cinco pliegues estratificados y AUC media para elegir la configuración. La
mejor fue profundidad 8, una observación mínima por hoja y sin pesos de clase;
su AUC media de validación fue 0,8574 (desviación estándar entre pliegues:
0,0286). La búsqueda tomó 65,84 segundos en el equipo de ejecución; el tiempo
depende del hardware.

El umbral de clasificación 0,309 se eligió maximizando F1 en predicciones
fuera de pliegue de los 800 casos de entrenamiento. Los 200 casos de prueba
no intervinieron en la selección de hiperparámetros ni del umbral.

## Desempeño

| Umbral | Muestra | AUC | Precisión | Recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0,309 | Entrenamiento | 0,992 | 0,708 | 0,970 | 0,819 |
| 0,309 | Prueba | 0,857 | 0,636 | 0,857 | 0,730 |
| 0,500 | Entrenamiento | 0,992 | 1,000 | 0,864 | 0,927 |
| 0,500 | Prueba | 0,857 | 0,364 | 0,082 | 0,133 |

Con el umbral ajustado, la matriz de confusión en prueba tiene 127 verdaderos
negativos, 24 falsos positivos, 7 falsos negativos y 42 verdaderos positivos.
Así, para detectar 42 de los 49 fraudes observados, se enviarían 66 de los 200
siniestros a revisión. Con umbral 0,5 solo se detectarían 4 fraudes. La AUC
es la misma con ambos umbrales porque mide la capacidad de ordenación de las
probabilidades, no una decisión binaria particular.

La brecha entrenamiento-prueba del umbral ajustado es 0,135 en AUC y 0,088
en F1. Hay sobreajuste visible en entrenamiento, aunque la AUC de validación
(0,857) y la de prueba (0,857) son cercanas. La diferencia entre la AUC y el
recall con umbral 0,5 muestra que un clasificador que ordena razonablemente
bien los casos puede producir malas decisiones si el umbral no se adapta al
objetivo. El umbral que maximiza F1 no incorpora el costo monetario de revisar
un caso ni el de no detectar un fraude; una decisión operativa requeriría esos
costos y comprobar la calibración de probabilidades.

Entre las variables codificadas con mayor importancia por reducción de
impureza aparecen daño grave, afición `chess`, afición `cross-fit` e importe
reclamado por vehículo. La importancia de las aficiones resulta difícil de
justificar causalmente y merece cautela antes de usar el modelo fuera de esta
muestra. Estas importancias pueden favorecer variables con muchos valores y
no prueban relaciones causales. El bosque capta interacciones y relaciones no
lineales sin escalar variables, pero resulta menos interpretable que una
regresión logística y requiere validar umbral, sobreajuste y estabilidad.

## Para la comparación grupal

Usar la partición de `resultados_random_forest/particion.csv` en los otros dos
modelos y la regresión logística. La tabla final debe presentar las mismas
métricas en entrenamiento y prueba para todos, además de sus brechas, tiempo
de entrenamiento, sensibilidad a hiperparámetros e interpretabilidad. Aún no
se puede recomendar un modelo frente a los otros sin sus resultados. Incluir
en el informe la cita de Kaggle, la aprobación del tema por el
profesor y la declaración de uso de herramientas de IA exigida en el enunciado.

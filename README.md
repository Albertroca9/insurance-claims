# Detección de fraude en siniestros de seguros

Trabajo para **Tarea 2: Comparación de modelos de Machine Learning en finanzas**.
La variable a predecir es `fraud_reported` (`1` = fraude reportado). La aplicación
propuesta es priorizar siniestros para revisión después de recibir la declaración
y los datos del incidente. Las columnas de importes, daños, autoridades y reporte
policial se usan bajo la suposición de que ya están disponibles en ese momento.
Si la decisión debe tomarse antes, hay que retirar las columnas que todavía no
se conocen y volver a entrenar todos los modelos.

El [informe definitivo](Tarea_2_Finanzas_ML.pdf) incluye las cuatro figuras
del anexo. Los PNG están en `figuras_informe/`.

La auditoría reproducible del CSV está en
`auditoria_csv.py` y se ejecuta con `python auditoria_csv.py`.
La fuente indicada para los datos es [Auto Insurance Claims Data (Kaggle)](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data).
La comparación con una copia pública del CSV original se resume en
`comparacion_origen.md`.

## Reproducir Random Forest

Desde esta carpeta:

```powershell
python -m pip install -r requirements.txt
python random_forest.py
python sensibilidad_random_forest.py
python -m unittest discover -s tests -v
```

El script lee `insurance_claims_limpio.csv` y genera en
`resultados_random_forest/`:

- `particion.csv`: `row_id` es la posición de la fila en el CSV original y
  `split` indica entrenamiento o prueba.
- `metricas.csv`: AUC, precisión, recall y F1 en entrenamiento y prueba.
  Las dos primeras filas usan el umbral ajustado en entrenamiento; las dos
  siguientes usan 0,5 como referencia.
- `validacion_cruzada.csv`: ocho configuraciones evaluadas con cinco pliegues.
- `predicciones_prueba.csv`: probabilidades y predicciones finales por `row_id`.
- `importancias.csv`: importancias por reducción de impureza de las variables
  codificadas. Son descriptivas y pueden favorecer variables con muchos valores.
- `resumen.json`: parámetros, tamaños, tiempo y matrices de confusión.
- `roc_prueba.png`: curva ROC en la muestra de prueba.

`sensibilidad_random_forest.py` reproduce la comparación de la sección 2.2
del informe. Calcula AUC media en validación sobre entrenamiento al excluir
aficiones, severidad o ambas, manteniendo los hiperparámetros fijos.

## Protocolo común para comparar modelos

Todos los modelos del grupo deben leer el mismo CSV **sin reordenar filas** y
usar `particion.csv` para formar las mismas muestras. El conjunto de prueba
tiene 200 filas y se reserva exclusivamente para la evaluación final. Toda
imputación, codificación y estandarización se ajusta dentro de entrenamiento
y de cada pliegue de validación. KNN y SVM requieren estandarizar las columnas
numéricas. El Random Forest selecciona hiperparámetros por AUC media en cinco
pliegues estratificados y ajusta el umbral maximizando F1 con predicciones
fuera de pliegue de entrenamiento.

Para la tabla comparativa del informe, usar como positivo `fraud_reported = 1`
y reportar AUC, precisión, recall y F1 en entrenamiento y prueba, junto con la
brecha entre ambos. También se requiere una regresión logística de referencia,
además de los tres modelos de Machine Learning.

La partición aleatoria estratificada se eligió porque el CSV limpio solo
conserva mes y día de semana del incidente. El archivo original sí incluye
`incident_date`; si el equipo adopta una partición temporal, deberá recuperar
esa fecha y aplicar la misma separación a todos los modelos.

La comparación del origen mostró que 91 valores `None` de
`authorities_contacted` se convirtieron en vacíos. El modelo actual imputa
esos vacíos con la moda, mientras que el notebook de KNN y logística los
codifica como categoría separada. Para recuperar el valor original, hay que crear una nueva versión
común del CSV y volver a entrenar y comparar todos los modelos.

## Reproducir KNN y regresión logística

Abre `modelos_knn_logistica.ipynb` en Jupyter o VS Code y ejecuta todas las
celdas desde la raíz del repositorio, después de generar la partición de
Random Forest. El notebook comprueba que usa las mismas filas, ajusta
codificación y escalado dentro de la validación de KNN y genera
`figuras_informe/knn_diagnostico.png`. Las cifras del informe proceden de
este notebook revisado.

El notebook y el PDF recibidos se conservan en `referencias/`. El notebook
original ajustaba el escalado antes de la validación cruzada y el anexo
del PDF mostraba la AUC de prueba para varios valores de `k`. El nuevo
gráfico evalúa la sensibilidad a `k` en los pliegues de entrenamiento.

## Reproducir XGBoost

```powershell
python -m pip install -r requirements_xgboost.txt
python xgboost_fraude.py
```

El script verifica la partición común cuando encuentra el CSV de Random Forest
y guarda métricas, predicciones y gráficos en `resultados_xgboost/`. El
[detalle del modelo](README_xgboost.md) explica también la logística exploratoria
con dos variables. El [guion de presentación en PDF](output/pdf/guion_presentacion.pdf)
incluye el texto para decir en cada diapositiva y los tiempos orientativos.
Las notas de apoyo y la tabla comparativa están en el [guion editable](guion_presentacion.md).

## Requisitos del trabajo grupal

El enunciado requiere aprobación previa del tema por el profesor, citar la
fuente original del CSV y declarar el uso de herramientas de IA en el informe.

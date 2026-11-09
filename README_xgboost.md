# XGBoost - detección de reclamos fraudulentos

Modelo XGBoost de la Tarea 2 (Fernán Gazmuri).

## Cómo correrlo

```
python -m pip install -r requirements_xgboost.txt
python xgboost_fraude.py
```

Necesita `insurance_claims_limpio.csv` en la misma carpeta. Tarda cerca de un minuto y guarda todo en `resultados_xgboost/`.

## Qué hace

1. Separa la muestra igual que el resto del grupo con `train_test_split(test_size=0.20, stratify=y, random_state=42)`. Si existe `resultados_random_forest/particion.csv`, comprueba la asignación y la etiqueta de cada fila.
2. Elige los hiperparámetros con búsqueda aleatoria (200 combinaciones) y validación cruzada de 5 pliegues, solo con entrenamiento. Con las versiones fijadas, la búsqueda eligió 300 árboles, profundidad 6 y tasa de aprendizaje 0.05.
3. Elige el umbral de clasificación (0.26) maximizando el F1 con validación cruzada.
4. Mide la importancia de las variables y estima una logística con las 2 más importantes, para comparar.
5. Calcula las métricas en entrenamiento y prueba y genera los gráficos. `predicciones_prueba.csv` permite verificar las probabilidades y decisiones de XGBoost y de la logística de dos variables para cada fila de prueba.

## Resultados (`resultados_xgboost/resultados.csv`)

| Set | AUC | Precisión | Recall | F1 |
|---|---|---|---|---|
| Entrenamiento | 0.9801 | 0.6679 | 0.8939 | 0.7646 |
| Prueba | 0.8227 | 0.6364 | 0.8571 | 0.7304 |
| Brecha | 0.1574 | 0.0316 | 0.0368 | 0.0341 |

Estas cifras corresponden a la ejecución local con XGBoost 3.2.0. En el
borrador del equipo figuraba una AUC de prueba de 0.8304.

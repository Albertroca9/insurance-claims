# Comparación del CSV del proyecto con el dataset de origen

Fuente indicada por el grupo: [Auto Insurance Claims Data (Kaggle)](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data). Para el cotejo fila a fila se utilizó una [copia pública del CSV original](https://raw.githubusercontent.com/IrfanFathima/Insurance_claims/main/insurance_claims.csv), porque la descarga de Kaggle no estaba accesible desde este entorno. La copia coincide en las 1.000 filas, orden, variables conservadas y valores esperados; conviene confirmar su identidad exacta con la descarga directa de Kaggle si el grupo la tiene.

## Cambios comprobados

- La copia pública del origen tiene 1.000 filas y 39 columnas; el archivo del proyecto tiene 1.000 filas y 36 columnas.
- Se eliminaron `policy_number`, `insured_zip`, `incident_location`, `policy_bind_date`, `incident_date` y `total_claim_amount`.
- Se añadieron `dias_desde_emision`, `incident_month` e `incident_day_of_week`. Las tres coinciden en las 1.000 filas con los cálculos a partir de las fechas originales interpretadas con día primero. El total reclamado eliminado es exactamente la suma de `injury_claim`, `property_claim` y `vehicle_claim` en las 1.000 filas.
- `fraud_reported` pasó de `Y/N` a `1/0` sin cambios de etiqueta.
- `?` pasó a `MISSING` en `collision_type` (178), `property_damage` (360) y `police_report_available` (343).
- **Cambio problemático:** la categoría literal `None` de `authorities_contacted` en 91 filas se convirtió en una celda vacía. En el Random Forest actual esos vacíos se imputan con la moda, por lo que se pierde el significado de «no se contactó a ninguna autoridad». Conviene recuperar `None` como categoría explícita en una nueva versión común del CSV y reentrenar todos los modelos.
- Las demás columnas conservadas son iguales fila a fila.

Con los hiperparámetros actuales fijos, una comprobación de cinco pliegues
solo en entrenamiento dio AUC media 0,8574 con la imputación actual y 0,8482
al recuperar `None` como categoría. La diferencia es pequeña frente a la
variación entre pliegues (aproximadamente 0,029) y no justifica conservar una
transformación semánticamente incorrecta. Esta comprobación no usó prueba ni
volvió a optimizar hiperparámetros.

## Valores anómalos y fechas

- `umbrella_limit = -1000000` en `row_id=290` **ya figura en el origen**; no lo introdujo la limpieza. Su significado sigue sin estar documentado.
- `dias_desde_emision = -20` en `row_id=578` también reproduce fielmente el origen: `policy_bind_date = 22-02-2015` e `incident_date = 2/2/2015`. Es un caso con incidente anterior a la fecha de vinculación de la póliza, no un error aritmético de la variable derivada.
- Las fechas de incidente del original van de 2015-01-01 a 2015-03-01. El CSV limpio solo conserva el mes y día de la semana; para una evaluación temporal habría que recuperar `incident_date` desde el origen.

## Decisión recomendada

Conservar el CSV actual y los resultados obtenidos como referencia reproducible. Crear una versión nueva que preserve `authorities_contacted = None` como categoría explícita y documente la decisión sobre los dos valores negativos. Acordar esa versión y la partición con todo el grupo antes de reentrenar los tres modelos y la regresión. También confirmar si `incident_severity`, `insured_hobbies` y los importes reclamados estarían disponibles en el momento de predicción.

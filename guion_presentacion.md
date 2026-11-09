# Guion de presentación: detección de fraude en reclamos de seguros

Ocho diapositivas, unos **8:45 minutos**. El enunciado da un máximo de 10 minutos de exposición y 5 de preguntas.

| Diapositiva | Tiempo | Qué pondría en pantalla | Qué diría |
|---|---:|---|---|
| **1. Qué reclamos conviene revisar primero** | 0:35 | Título, integrantes y una pregunta grande: «Si la aseguradora tiene capacidad limitada, ¿qué casos debería revisar primero?». | «Predecimos si un reclamo queda marcado como fraude reportado. El puntaje sirve para ordenar expedientes y asignar tiempo de revisión. La decisión final la toma quien investiga el caso». |
| **2. Datos y diseño de la prueba** | 1:00 | Tres cifras grandes: 1.000 reclamos, 247 positivos y 200 casos de prueba. Esquema 800 entrenamiento → validación interna → 200 prueba. Fuente Kaggle al pie. | «Los cuatro modelos se evaluaron en los mismos 200 reclamos, con 49 positivos. Los ajustes se hicieron con entrenamiento. Quitamos identificadores y el importe total redundante. Suponemos que la severidad y los importes ya están registrados al priorizar. La disponibilidad de aficiones debe verificarse». |
| **3. Qué aportan KNN y la logística** | 1:05 | Panel ROC de la figura de KNN y dos AUC de prueba: **0,500** y **0,781**. Anotar «KNN: 8/49» y «Logística: 31/49». | «KNN estandariza y busca casos parecidos. Con muchas variables codificadas, la distancia parece poco informativa en esta muestra y la ROC queda cerca de la diagonal. La logística con pesos de clase es una referencia rápida y más interpretable. Detecta más fraudes en el umbral usado». |
| **4. Random Forest: capacidad de ordenar** | 1:00 | Curva ROC de Random Forest. Tres cifras: AUC en entrenamiento **0,992**, validación **0,857** y prueba **0,857**. | «El bosque combina 150 árboles. La validación eligió una profundidad máxima de 8. La diferencia entre entrenamiento y prueba muestra sobreajuste, aunque la validación anticipó bien el resultado de prueba. La AUC indica qué tan bien ordena casos positivos por encima de negativos». |
| **5. Random Forest: qué cambia con el umbral** | 1:05 | Dos columnas: umbral 0,50 → **4/49** fraudes detectados, umbral 0,309 → **42/49**. Debajo: **66 revisiones = 42 positivos + 24 falsos positivos**. | «Con un umbral de 0,50, el modelo detecta 4 de los 49 fraudes reportados. Con el umbral de 0,309, elegido al maximizar F1 en validación, detecta 42 de 49. Para conseguirlo, marcaría 66 de los 200 reclamos, incluidos 24 falsos positivos. La AUC es la misma porque el orden de los puntajes no cambia. El umbral operativo dependería de cuántos reclamos se puedan revisar y del costo de cada tipo de error». |
| **6. Qué aprendió XGBoost** | 1:25 | Figura de importancia de XGBoost. Destacar severidad y aficiones, que suman cerca del 60 % de la ganancia. Mostrar **AUC 0,823** y **F1 0,730** en prueba. | «XGBoost construye árboles de forma secuencial. Cada árbol busca reducir los errores que quedan en el conjunto. Probamos 200 configuraciones en validación cruzada y elegimos la de mayor AUC. Después fijamos el umbral de 0,26 maximizando F1 con predicciones de validación. En prueba, el modelo obtuvo una AUC de 0,823 y un F1 de 0,730. La severidad y las aficiones concentraron cerca del 60 % de la ganancia de los árboles. Esa concentración nos llevó a probar una regresión logística con solo esas dos variables». |
| **7. Comparación final** | 1:15 | Tabla de abajo. Resaltar la AUC de Random Forest y el empate de F1 con XGBoost. Indicar que cada modelo usa su umbral de decisión. | «Random Forest obtuvo la mayor AUC de prueba, 0,857. A los umbrales elegidos, el bosque y XGBoost marcaron los mismos 66 reclamos y detectaron 42 de los 49 positivos. La logística exploratoria con severidad y aficiones también marcó esos mismos casos y logró una AUC de 0,844. Eso muestra que los puntajes pueden ordenar distinto y llevar a la misma decisión de revisión. En la tabla, la logística de referencia usa todos los predictores y obtiene 0,781. KNN queda cerca del azar». |
| **8. Recomendación y límites** | 1:20 | Tres ideas: **priorizar con Random Forest**, **validar predictores disponibles**, **probar en datos posteriores y fijar capacidad de revisión**. | «Para ordenar reclamos en esta muestra, recomendamos Random Forest. La recomendación depende de que la señal se mantenga en otros datos. Si retiramos severidad y aficiones, la AUC de validación del bosque cae de 0,857 a 0,600. Antes de usarlo, verificaríamos de dónde salen esas variables y haríamos una evaluación temporal». |

## Tabla para la diapositiva 7

| Modelo | AUC prueba | Precisión | Recall | F1 | Brecha AUC |
|---|---:|---:|---:|---:|---:|
| KNN | 0,500 | 0,267 | 0,163 | 0,203 | 0,363 |
| Regresión logística | 0,781 | 0,544 | 0,633 | 0,585 | 0,184 |
| Random Forest | **0,857** | **0,636** | **0,857** | **0,730** | 0,135 |
| XGBoost | 0,823 | **0,636** | **0,857** | **0,730** | 0,157 |

## Para los cinco minutos de preguntas

- **¿Cómo se evitó ajustar con la prueba?** La codificación y el ajuste se hicieron en entrenamiento. KNN usó validación para elegir vecinos y los modelos de árboles usaron validación para elegir hiperparámetros y el umbral. La permutación en prueba se usó como diagnóstico posterior.
- **¿La logística de dos variables es la de la tabla?** Es un análisis exploratorio. La de la tabla es el modelo de referencia ajustado con todos los predictores preparados.
- **¿Por qué las aficiones son tan importantes?** Es una asociación de esta muestra. La dependencia se ve en XGBoost y en la sensibilidad de Random Forest, y requiere revisar cómo se generaron y registraron los datos.
- **¿Por qué usar AUC y F1?** AUC evalúa el orden de los puntajes. F1 combina precisión y recall para la decisión binaria con un umbral específico. El umbral operativo también debe considerar el costo de investigar y dejar pasar fraude.

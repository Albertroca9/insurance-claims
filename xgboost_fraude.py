# Tarea 2 - Aplicaciones en Finanzas con Machine Learning
# Modelo XGBoost para detectar reclamos fraudulentos (Fernán Gazmuri)
#
# Uso: python xgboost_fraude.py
# Necesita insurance_claims_limpio.csv en la misma carpeta.
# Guarda tablas y gráficos en la carpeta resultados_xgboost/.

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

SEMILLA = 42
CARPETA = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(CARPETA, "resultados_xgboost")
os.makedirs(SALIDA, exist_ok=True)

# ---------------------------------------------------------------
# 1. Datos y separación entrenamiento / prueba
# ---------------------------------------------------------------
# Conserva como categoría las celdas vacías de authorities_contacted.
# En el archivo original esas filas tenían el valor literal "None".
df = pd.read_csv(os.path.join(CARPETA, "insurance_claims_limpio.csv"),
                 keep_default_na=False)
y = df["fraud_reported"]
X = df.drop(columns="fraud_reported")
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()

# Misma separación que el resto de los modelos del grupo
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEMILLA)

archivo_particion = os.path.join(CARPETA, "resultados_random_forest", "particion.csv")
if os.path.isfile(archivo_particion):
    particion = pd.read_csv(archivo_particion).set_index("row_id").sort_index()
    asignacion = pd.Series("train", index=y.index)
    asignacion.loc[X_te.index] = "test"
    if (not particion["split"].equals(asignacion)
            or not particion["fraud_reported"].equals(y)):
        raise ValueError("La partición de XGBoost no coincide con la partición común.")

# One-hot de las categóricas (los árboles no necesitan estandarizar)
prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", "passthrough", num_cols),
])

# ---------------------------------------------------------------
# 2. Hiperparámetros: búsqueda aleatoria con validación cruzada
#    (solo con la muestra de entrenamiento)
# ---------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
pipe = Pipeline([
    ("prep", prep),
    ("xgb", XGBClassifier(eval_metric="logloss", tree_method="hist",
                          random_state=SEMILLA, n_jobs=1)),
])
espacio = {
    "xgb__n_estimators": [50, 100, 200, 300, 500],
    "xgb__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    "xgb__max_depth": [1, 2, 3, 4, 5, 6, 8],
    "xgb__min_child_weight": [1, 3, 5, 10],
    "xgb__subsample": [0.6, 0.8, 1.0],
    "xgb__colsample_bytree": [0.3, 0.5, 0.8, 1.0],
    "xgb__gamma": [0, 0.5, 1, 2],
    "xgb__reg_lambda": [1, 5, 10],
    "xgb__scale_pos_weight": [1, 3],  # 3 = da más peso a los fraudes (~1 de cada 4)
}
busqueda = RandomizedSearchCV(pipe, espacio, n_iter=200, scoring="roc_auc",
                              cv=cv, random_state=SEMILLA, n_jobs=-1)
t0 = time.perf_counter()
busqueda.fit(X_tr, y_tr)
t_busqueda = time.perf_counter() - t0
mejores = {k.replace("xgb__", ""): v for k, v in busqueda.best_params_.items()}
print("Mejores hiperparámetros:", mejores)
print(f"AUC de validación cruzada: {busqueda.best_score_:.3f}")


def elegir_umbral(modelo, X_, y_):
    """Umbral que maximiza el F1 con predicciones fuera de pliegue (entrenamiento)."""
    p = cross_val_predict(modelo, X_, y_, cv=cv, method="predict_proba")[:, 1]
    umbrales = np.arange(0.10, 0.91, 0.01)
    return float(umbrales[np.argmax([f1_score(y_, p >= u) for u in umbrales])])


modelo = busqueda.best_estimator_
umbral = elegir_umbral(modelo, X_tr, y_tr)
t0 = time.perf_counter()
modelo.fit(X_tr, y_tr)
t_final = time.perf_counter() - t0
print(f"Umbral elegido: {umbral:.2f} | ajuste final: {t_final:.2f} s | "
      f"búsqueda: {t_busqueda:.0f} s")

# ---------------------------------------------------------------
# 3. Importancia de variables
# ---------------------------------------------------------------
# (a) Ganancia de los árboles, sumando las dummies de cada variable original
nombres = modelo.named_steps["prep"].get_feature_names_out()
gan = modelo.named_steps["xgb"].get_booster().get_score(importance_type="total_gain")
gan = pd.Series({nombres[int(k[1:])].split("__")[1]: v for k, v in gan.items()})
original = [next((c for c in cat_cols if n.startswith(c + "_")), n) for n in gan.index]
imp_gan = gan.groupby(original).sum().sort_values(ascending=False)
imp_gan = imp_gan / imp_gan.sum()

# (b) Permutación: cuánto cae el AUC de prueba al desordenar cada variable
perm = permutation_importance(modelo, X_te, y_te, scoring="roc_auc",
                              n_repeats=30, random_state=SEMILLA, n_jobs=-1)
imp_perm = pd.DataFrame({"variable": X_te.columns, "caida_AUC": perm.importances_mean,
                         "sd": perm.importances_std}).sort_values("caida_AUC", ascending=False)
imp_gan.rename("importancia_ganancia").to_csv(
    os.path.join(SALIDA, "importancia_ganancia.csv"), float_format="%.4f")
imp_perm.to_csv(os.path.join(SALIDA, "importancia_permutacion.csv"),
                index=False, float_format="%.4f")

# ---------------------------------------------------------------
# 4. Logística con las 2 variables más importantes del XGBoost
#    (para ver cuánto aporta la flexibilidad del XGBoost)
# ---------------------------------------------------------------
top2 = imp_gan.index[:2].tolist()
print("Dos variables más importantes:", top2)
logit2 = Pipeline([("prep", OneHotEncoder(handle_unknown="ignore")),
                   ("lr", LogisticRegression(max_iter=5000))])
umbral2 = elegir_umbral(logit2, X_tr[top2], y_tr)
logit2.fit(X_tr[top2], y_tr)

# ---------------------------------------------------------------
# 5. Métricas en entrenamiento y prueba
# ---------------------------------------------------------------
def metricas(p, y_, u):
    yhat = (p >= u).astype(int)
    return {"AUC": roc_auc_score(y_, p), "Precision": precision_score(y_, yhat),
            "Recall": recall_score(y_, yhat), "F1": f1_score(y_, yhat),
            "Accuracy": accuracy_score(y_, yhat)}


filas = []
for nombre, prob, u in [
        ("XGBoost", lambda d: modelo.predict_proba(d)[:, 1], umbral),
        ("Logistica 2 variables", lambda d: logit2.predict_proba(d[top2])[:, 1], umbral2)]:
    tr = metricas(prob(X_tr), y_tr, u)
    te = metricas(prob(X_te), y_te, u)
    filas.append({"Modelo": nombre, "Set": "Entrenamiento", **tr})
    filas.append({"Modelo": nombre, "Set": "Prueba", **te})
    filas.append({"Modelo": nombre, "Set": "Brecha (entr. - prueba)",
                  **{k: tr[k] - te[k] for k in tr}})
resultados = pd.DataFrame(filas)
resultados.to_csv(os.path.join(SALIDA, "resultados.csv"), index=False, float_format="%.4f")
print(resultados.round(4).to_string(index=False))

prob_xgb = modelo.predict_proba(X_te)[:, 1]
prob_logit2 = logit2.predict_proba(X_te[top2])[:, 1]
pd.DataFrame({
    "row_id": X_te.index,
    "fraud_reported": y_te.to_numpy(),
    "probabilidad_xgboost": prob_xgb,
    "prediccion_xgboost": (prob_xgb >= umbral).astype(int),
    "probabilidad_logistica_2_variables": prob_logit2,
    "prediccion_logistica_2_variables": (prob_logit2 >= umbral2).astype(int),
}).sort_values("row_id").to_csv(os.path.join(SALIDA, "predicciones_prueba.csv"), index=False)

# ---------------------------------------------------------------
# 6. Sensibilidad a hiperparámetros: se cambia uno a la vez,
#    dejando el resto en su mejor valor (validación cruzada)
# ---------------------------------------------------------------
def curva(param, valores):
    out = []
    for v in valores:
        m = Pipeline([("prep", prep), ("xgb", XGBClassifier(
            eval_metric="logloss", tree_method="hist", random_state=SEMILLA,
            n_jobs=1, **{**mejores, param: v}))])
        auc_tr, auc_va = [], []
        for i_tr, i_va in cv.split(X_tr, y_tr):
            m.fit(X_tr.iloc[i_tr], y_tr.iloc[i_tr])
            auc_tr.append(roc_auc_score(y_tr.iloc[i_tr], m.predict_proba(X_tr.iloc[i_tr])[:, 1]))
            auc_va.append(roc_auc_score(y_tr.iloc[i_va], m.predict_proba(X_tr.iloc[i_va])[:, 1]))
        out.append({"param": param, "valor": v, "AUC_entr": np.mean(auc_tr),
                    "AUC_valid": np.mean(auc_va), "AUC_valid_sd": np.std(auc_va)})
    return pd.DataFrame(out)


sens = pd.concat([curva("max_depth", [1, 2, 3, 4, 6, 8, 10]),
                  curva("n_estimators", [10, 25, 50, 100, 200, 400, 800]),
                  curva("learning_rate", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5])])
sens.to_csv(os.path.join(SALIDA, "sensibilidad_hiperparametros.csv"),
            index=False, float_format="%.4f")

# ---------------------------------------------------------------
# 7. Gráficos
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
titulos = {"max_depth": "Profundidad máxima", "n_estimators": "N.º de árboles",
           "learning_rate": "Tasa de aprendizaje"}
for ax, (p, g) in zip(axes, sens.groupby("param", sort=False)):
    ax.plot(g["valor"], g["AUC_entr"], "o-", color="#1f4e79", label="Entrenamiento")
    ax.plot(g["valor"], g["AUC_valid"], "o-", color="#c55a11", label="Validación (CV)")
    ax.fill_between(g["valor"], g["AUC_valid"] - g["AUC_valid_sd"],
                    g["AUC_valid"] + g["AUC_valid_sd"], color="#c55a11", alpha=0.15)
    ax.axvline(mejores[p], color="grey", ls=":", lw=1)
    if p != "max_depth":
        ax.set_xscale("log")
    ax.set_title(titulos[p]); ax.set_xlabel(p); ax.grid(alpha=0.3)
axes[0].set_ylabel("AUC"); axes[0].legend(loc="lower right", fontsize=8)
fig.suptitle("XGBoost: sensibilidad a hiperparámetros (línea punteada = valor elegido)")
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "fig_sensibilidad.png"), dpi=200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
top = imp_gan.head(10)[::-1]
axes[0].barh(top.index, top.values, color="#1f4e79")
axes[0].set_title("Participación en la ganancia total (top 10)")
tp = imp_perm.head(10)[::-1]
axes[1].barh(tp["variable"], tp["caida_AUC"], xerr=tp["sd"], color="#c55a11")
axes[1].set_title("Caída de AUC en prueba al permutar (top 10)")
for ax in axes:
    ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "fig_importancia.png"), dpi=200)

fig, ax = plt.subplots(figsize=(5, 4.5))
for nombre, p, c, ls in [("XGBoost", prob_xgb, "#1f4e79", "-"),
                         ("Logística, 2 variables", prob_logit2,
                          "#c55a11", "--")]:
    fpr, tpr, _ = roc_curve(y_te, p)
    ax.plot(fpr, tpr, color=c, ls=ls, label=f"{nombre} (AUC {roc_auc_score(y_te, p):.3f})")
ax.plot([0, 1], [0, 1], ls="--", color="lightgrey")
ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
ax.set_title("Curva ROC en la muestra de prueba"); ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "fig_roc.png"), dpi=200)

pd.DataFrame({"Etapa": ["Búsqueda de hiperparámetros (200 x 5 pliegues)", "Ajuste final"],
              "Segundos": [t_busqueda, t_final]}).to_csv(
    os.path.join(SALIDA, "tiempos.csv"), index=False, float_format="%.2f")
print("Resultados guardados en", SALIDA)

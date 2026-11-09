"""Reproducible Random Forest analysis for insurance claim fraud."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "insurance_claims_limpio.csv"
OUTPUT_DIR = ROOT / "resultados_random_forest"
TARGET = "fraud_reported"
SEED = 42
THRESHOLD = 0.5


def make_split(df: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return original row positions for a reproducible stratified 80/20 split."""
    if TARGET not in df:
        raise ValueError(f"Falta la columna objetivo: {TARGET}")
    return train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        stratify=df[TARGET],
        random_state=seed,
    )


def make_pipeline(X: pd.DataFrame, n_estimators: int = 150) -> Pipeline:
    """Fit all imputers and encoders inside the training folds."""
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.columns.difference(numeric, sort=False).tolist()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", forest)])


def select_threshold(y: pd.Series | np.ndarray, probability: np.ndarray) -> float:
    """Choose the F1-maximizing threshold using out-of-fold training scores only."""
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, label: str, threshold: float) -> dict:
    probability = model.predict_proba(X)[:, 1]
    predicted = (probability >= threshold).astype(int)
    return {
        "modelo": "Random Forest",
        "conjunto": label,
        "umbral": threshold,
        "n": len(y),
        "auc": roc_auc_score(y, probability),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("fraud_reported debe ser binaria (0/1) y no tener vacíos")
    train_idx, test_idx = make_split(df)
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    OUTPUT_DIR.mkdir(exist_ok=True)
    split = pd.DataFrame({"row_id": np.arange(len(df)), "fraud_reported": y, "split": "train"})
    split.loc[test_idx, "split"] = "test"
    split.to_csv(OUTPUT_DIR / "particion.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        estimator=make_pipeline(X_train),
        param_grid={
            "model__max_depth": [None, 8],
            "model__min_samples_leaf": [1, 5],
            "model__class_weight": [None, "balanced_subsample"],
        },
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=1,
        return_train_score=True,
    )
    start = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    model = search.best_estimator_

    out_of_fold_probability = cross_val_predict(
        model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    tuned_threshold = select_threshold(y_train, out_of_fold_probability)

    metrics = pd.DataFrame(
        [
            evaluate(model, X_train, y_train, "entrenamiento", tuned_threshold),
            evaluate(model, X_test, y_test, "prueba", tuned_threshold),
            evaluate(model, X_train, y_train, "entrenamiento", THRESHOLD),
            evaluate(model, X_test, y_test, "prueba", THRESHOLD),
        ]
    )
    metrics.to_csv(OUTPUT_DIR / "metricas.csv", index=False, float_format="%.6f")
    cv_results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    cv_results.to_csv(OUTPUT_DIR / "validacion_cruzada.csv", index=False, float_format="%.6f")

    probability = model.predict_proba(X_test)[:, 1]
    predicted = (probability >= tuned_threshold).astype(int)
    predictions = pd.DataFrame(
        {
            "row_id": test_idx,
            "real": y_test.to_numpy(),
            "probabilidad_fraude": probability,
            "prediccion_umbral_ajustado": predicted,
            "prediccion_umbral_0_5": (probability >= THRESHOLD).astype(int),
        }
    ).sort_values("row_id")
    predictions.to_csv(OUTPUT_DIR / "predicciones_prueba.csv", index=False, float_format="%.6f")

    names = model.named_steps["preprocess"].get_feature_names_out()
    importances = pd.DataFrame(
        {"variable_codificada": names, "importancia_impureza": model.named_steps["model"].feature_importances_}
    ).sort_values("importancia_impureza", ascending=False)
    importances.to_csv(OUTPUT_DIR / "importancias.csv", index=False, float_format="%.6f")

    fpr, tpr, _ = roc_curve(y_test, probability)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Random Forest (AUC = {metrics.iloc[1]['auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Azar")
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Recall", title="Curva ROC en prueba")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_prueba.png", dpi=160)
    plt.close(fig)

    summary = {
        "archivo": DATA_PATH.name,
        "filas": len(df),
        "semilla": SEED,
        "particion": "80% entrenamiento / 20% prueba, estratificada aleatoria",
        "clase_positiva": "fraud_reported = 1",
        "umbral_ajustado": tuned_threshold,
        "criterio_umbral": "F1 máximo en predicciones fuera de pliegue del entrenamiento",
        "umbral_referencia": THRESHOLD,
        "n_entrenamiento": len(train_idx),
        "n_prueba": len(test_idx),
        "fraudes_entrenamiento": int(y_train.sum()),
        "fraudes_prueba": int(y_test.sum()),
        "validacion": "StratifiedKFold 5 pliegues, selección por AUC media",
        "mejores_hiperparametros": search.best_params_,
        "auc_validacion_media": float(search.best_score_),
        "segundos_busqueda_hiperparametros": round(elapsed, 2),
        "segundos_entrenamiento_total": round(time.perf_counter() - start, 2),
        "matriz_confusion_prueba_umbral_ajustado": confusion_matrix(y_test, predicted).tolist(),
        "matriz_confusion_prueba_umbral_0_5": confusion_matrix(y_test, probability >= THRESHOLD).tolist(),
        "version_sklearn": sklearn.__version__,
    }
    (OUTPUT_DIR / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"AUC media en validación: {search.best_score_:.4f}")
    print(f"Mejores hiperparámetros: {search.best_params_}")
    print(f"Resultados: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

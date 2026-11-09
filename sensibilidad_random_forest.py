"""Reproduce la comparación de variables de Random Forest del informe."""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from random_forest import DATA_PATH, SEED, TARGET, make_pipeline, make_split


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    train_idx, _ = make_split(data)
    train = data.iloc[train_idx]
    X = train.drop(columns=TARGET)
    y = train[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    escenarios = [
        ("Todas las variables", []),
        ("Sin aficiones", ["insured_hobbies"]),
        ("Sin severidad", ["incident_severity"]),
        ("Sin ambas", ["insured_hobbies", "incident_severity"]),
    ]

    for nombre, excluidas in escenarios:
        predictores = X.drop(columns=excluidas)
        modelo = make_pipeline(predictores).set_params(
            model__max_depth=8, model__min_samples_leaf=1, model__class_weight=None
        )
        auc = cross_val_score(modelo, predictores, y, scoring="roc_auc", cv=cv)
        print(f"{nombre}: AUC media {auc.mean():.3f}")


if __name__ == "__main__":
    main()

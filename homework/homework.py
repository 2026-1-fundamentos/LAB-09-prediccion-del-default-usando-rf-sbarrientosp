"""Pronostico de default de clientes de tarjeta de credito."""

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def pregunta_01():

    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)

    # -------------------------------------------------------------------------
    # Paso 1: Limpieza
    # -------------------------------------------------------------------------
    def clean(df):
        df = df.copy()
        df = df.rename(columns={"default payment next month": "default"})
        df = df.drop(columns=["ID"])
        df = df[df["EDUCATION"] != 0]
        df = df[df["MARRIAGE"] != 0]
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
        return df

    train = clean(pd.read_csv("files/input/train_default_of_credit_card_clients.csv"))
    test = clean(pd.read_csv("files/input/test_default_of_credit_card_clients.csv"))

    # -------------------------------------------------------------------------
    # Paso 2: Split x/y
    # -------------------------------------------------------------------------
    x_train = train.drop(columns=["default"])
    y_train = train["default"]
    x_test = test.drop(columns=["default"])
    y_test = test["default"]

    # -------------------------------------------------------------------------
    # Paso 3: Pipeline con OneHotEncoding + RandomForest
    # -------------------------------------------------------------------------
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # -------------------------------------------------------------------------
    # Paso 4: Optimización de hiperparámetros con validación cruzada
    # -------------------------------------------------------------------------
    param_grid = {
        "classifier__n_estimators": [100],
        "classifier__max_depth": [None, 10],
        "classifier__min_samples_split": [2],
    }

    cv = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    cv.fit(x_train, y_train)
    best_model = cv.best_estimator_

    # -------------------------------------------------------------------------
    # Paso 5: Guardar modelo comprimido con gzip
    # -------------------------------------------------------------------------
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(best_model, f)

    # -------------------------------------------------------------------------
    # Paso 6 y 7: Métricas y matrices de confusión
    # -------------------------------------------------------------------------
    def compute_metrics(model, x, y, dataset):
        y_pred = model.predict(x)

        metrics = {
            "dataset": dataset,
            "precision": float(round(precision_score(y, y_pred, zero_division=0), 4)),
            "balanced_accuracy": float(round(balanced_accuracy_score(y, y_pred), 4)),
            "recall": float(round(recall_score(y, y_pred, zero_division=0), 4)),
            "f1_score": float(round(f1_score(y, y_pred, zero_division=0), 4)),
        }

        cm = confusion_matrix(y, y_pred)
        cm_entry = {
            "type": "cm_matrix",
            "dataset": dataset,
            "true_0": {
                "predicted_0": int(cm[0, 0]),
                "predicte_1": int(cm[0, 1]),
            },
            "true_1": {
                "predicted_0": int(cm[1, 0]),
                "predicted_1": int(cm[1, 1]),
            },
        }

        return metrics, cm_entry

    train_metrics, train_cm = compute_metrics(best_model, x_train, y_train, "train")
    test_metrics, test_cm = compute_metrics(best_model, x_test, y_test, "test")

    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(train_metrics) + "\n")
        f.write(json.dumps(test_metrics) + "\n")
        f.write(json.dumps(train_cm) + "\n")
        f.write(json.dumps(test_cm) + "\n")
# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import json
import gzip
import pickle
import zipfile
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer, 
    balanced_accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)
from sklearn.impute import SimpleImputer

#Paso 1. Carga y limpieza de datos
def clean_dataset(path):
    """
    Carga y limpia el dataset optimizando memoria y velocidad.
    """
    # Verificación básica de existencia
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo no existe: {path}")

    with zipfile.ZipFile(path, "r") as z:
        csv_file = z.namelist()[0]
        with z.open(csv_file) as f:
            # Optimización: No cargar la columna ID para ahorrar memoria
            df = pd.read_csv(f, usecols=lambda col: col != "ID")

    # Encadenamiento de métodos para limpieza
    df = (df
          .rename(columns={"default payment next month": "default"})
          .dropna()
    )

    df["EDUCATION"] = df["EDUCATION"].clip(upper=4)

    return df

#Paso 2. Construcción del Pipeline
def build_pipeline():
    """
    Construye un pipeline robusto con manejo de nulos y Random Forest.
    """
    categorical_features = ["EDUCATION", "MARRIAGE", "SEX"]

    # Pipeline para categóricas: Imputación (seguridad) + OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[("cat", categorical_transformer, categorical_features)],
        remainder="passthrough",
    )

    # RandomForest base
    random_forest_model = RandomForestClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", random_forest_model)]
    )

    return pipeline

#Paso 3. Optimización de Hiperparámetros
def optimize_pipeline(pipeline, x_train, y_train):
    """
    Optimiza el pipeline usando GridSearchCV.
    """
    scoring = make_scorer(balanced_accuracy_score)

    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    # n_jobs=-1 usa todos los núcleos para acelerar la búsqueda
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=2,
    )

    print("Iniciando optimización de hiperparámetros...")
    grid_search.fit(x_train, y_train)
    print(f"Mejores parámetros encontrados: {grid_search.best_params_}")

    return grid_search

#Paso 4. Guardar Modelo y Evaluar Métricas
def save_model(model, file_path="files/models/model.pkl.gz"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with gzip.open(file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modelo guardado exitosamente en: {file_path}")

def evaluate_model(model, x_train, y_train, x_test, y_test, file_path="files/output/metrics.json"):
    """
    Calcula métricas y las guarda en JSON.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    def get_metrics(y_true, y_pred, dataset_name):
        return {
            "type": "metrics",
            "dataset": dataset_name,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

    def get_confusion_matrix_dict(y_true, y_pred, dataset_name):
        cm = confusion_matrix(y_true, y_pred)
        return {
            "type": "cm_matrix",
            "dataset": dataset_name,
            "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
            "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
        }

    # Recopilar resultados
    results = [
        get_metrics(y_train, y_train_pred, "train"),
        get_metrics(y_test, y_test_pred, "test"),
        get_confusion_matrix_dict(y_train, y_train_pred, "train"),
        get_confusion_matrix_dict(y_test, y_test_pred, "test")
    ]

    # Guardar en archivo (formato JSON lines)
    with open(file_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Métricas guardadas en: {file_path}")

#Paso 5. Ejecución Principal
if __name__ == "__main__":
    # Configuración de rutas
    TRAIN_PATH = "./files/input/train_data.csv.zip"
    TEST_PATH = "./files/input/test_data.csv.zip"
    MODEL_PATH = "./files/models/model.pkl.gz"
    METRICS_PATH = "./files/output/metrics.json"

    df_train = clean_dataset(TRAIN_PATH)
    df_test = clean_dataset(TEST_PATH)

    x_train = df_train.drop(columns=["default"])
    y_train = df_train["default"]
    x_test = df_test.drop(columns=["default"])
    y_test = df_test["default"]

    pipeline = build_pipeline()
    best_model = optimize_pipeline(pipeline, x_train, y_train)
    evaluate_model(best_model, x_train, y_train, x_test, y_test, METRICS_PATH)

    save_model(best_model, MODEL_PATH)
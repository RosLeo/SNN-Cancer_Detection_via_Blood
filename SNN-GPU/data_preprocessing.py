# data_preprocessing.py

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle

def load_and_preprocess_data(excel_path: str, sheet_name: int = 5):
    """
    Legge il file Excel, pulisce i dati e restituisce:
    - X_train, X_val, X_test
    - y_train, y_val, y_test
    - X_train_resh, X_val_resh, X_test_resh (reshape per Conv1D)
    - y_train_ohe, y_val_ohe, y_test_ohe
    - label encoder (le)
    - numero di classi (n_classes)
    - data frame unito (X_scaled + colonna 'CANCER_TYPE') per usi successivi
    """
    # -------------------------------------------------------------------
    # 1) LETTURA E PULIZIA DATI
    # -------------------------------------------------------------------
    data = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Trasforma la colonna 'AJCC Stage' con il label encoding
    if 'AJCC Stage' in data.columns:
        data['AJCC Stage'] = data['AJCC Stage'].map({
            'I': 1,
            'II': 2,
            'III': 3,
            'NA': 0
        }).fillna(0)  # Riempie eventuali NaN con 0

    # Rimuove i simboli '*' e '**' dalle colonne testuali
    for columna in data.columns:
        if data[columna].dtype == 'object':
            for secuencia in ['*', '**']:
                data[columna] = data[columna].apply(
                    lambda x: x.replace(secuencia, '')
                    if isinstance(x, str) and secuencia in x else x
                )

    # Rimuove colonne non utili
    cols_to_remove = [
        'Patient ID #',
        'Sample ID #',
        'CancerSEEK Logistic Regression Score',
        'CancerSEEK Test Result'
    ]
    data = data.drop(columns=cols_to_remove, errors='ignore')

    # Converte eventuali colonne "object" in numerico se non contengono lettere
    def convert_to_numeric(column):
        if column.dtype in ['object', 'category']:
            contains_letters = any(
                isinstance(val, str) and any(c.isalpha() for c in val)
                for val in column
            )
            if not contains_letters:
                return pd.to_numeric(column, errors='coerce')
        return column

    data = data.apply(convert_to_numeric)

    # Riempie i NaN con la mediana (solo per le colonne numeriche)
    numeric_columns = data.select_dtypes(include='number')
    median_values = numeric_columns.median()
    for col in median_values.index:
        data[col].fillna(median_values[col], inplace=True)

    # Visualizza il conteggio delle categorie di y prima di SMOTE
    print("Conteggio prima di SMOTE:")
    print(data['Tumor type'].value_counts())

    # Estrae la colonna target e le features
    y = data['Tumor type']
    X = data.drop(columns=['Tumor type'])

    # Definizione dell'oversampler
    smote = SMOTE(
        sampling_strategy={
            'Liver': 500,
            'Esophagus': 500,
            'Ovary': 500,
            'Stomach': 500,
            'Pancreas': 500,
            'Lung': 500,
            'Breast': 500,
            'Colorectum': 500
        },
        random_state=42
    )

    X, y = smote.fit_resample(X, y)

    # Conteggio delle categorie di y dopo SMOTE
    print("\nConteggio dopo SMOTE:")
    print(y.value_counts())

    # Label Encoding sul target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("\nClassi trovate dal LabelEncoder:", le.classes_)

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # -------------------------------------------------------------------
    # 2) COSTRUZIONE DATASET & SPLIT TRAIN/VAL/TEST
    # -------------------------------------------------------------------
    # Split per test (10%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.10,
        random_state=42
    )
    # Split per validation (10% del totale => 1/9 di X_train_val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=1/9,
        random_state=42
    )

    print(f"\nTrain set: {X_train.shape[0]} campioni")
    print(f"Validation set: {X_val.shape[0]} campioni")
    print(f"Test set: {X_test.shape[0]} campioni")

    n_classes = len(le.classes_)

    # -------------------------------------------------------------------
    # 3) PREPARAZIONE DEI DATI PER LA RETE (RESHAPE) E OHE
    # -------------------------------------------------------------------
    # Reshape per Conv1D: (n_samples, n_features, 1)
    X_train_resh = np.asarray(X_train).reshape(-1, X_train.shape[1], 1)
    X_val_resh = np.asarray(X_val).reshape(-1, X_val.shape[1], 1)
    X_test_resh = np.asarray(X_test).reshape(-1, X_test.shape[1], 1)

    # One-hot encoding del target
    y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_val_ohe = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)
    y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

    # Uniamo le feature scalate con la colonna target per usi successivi (Siamese)
    dataset_genes = X_scaled.copy()
    dataset_genes['CANCER_TYPE'] = y_encoded

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_resh, X_val_resh, X_test_resh,
            y_train_ohe, y_val_ohe, y_test_ohe,
            le, n_classes, dataset_genes)

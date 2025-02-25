# main.py

# Se vuoi installare le dipendenze da script (non consigliato in produzione), puoi farlo con:
# import os
# os.system('pip install tensorflow imbalanced-learn')

import os
import numpy as np
from sklearn.metrics import classification_report
from data_preprocessing import load_and_preprocess_data
from cnn_model import train_cnn_model, plot_history
from siamese_model import (
    indices_save, get_siamese_model, train_siamese_model,
    test_oneshot, confusion_matrix_siamese
)
from tensorflow.keras.optimizers import Adam

# -----------------------------------------------------------------------------
# Caricamento e pulizia dati
# -----------------------------------------------------------------------------
EXCEL_PATH = "/home/musimathicslab/Desktop/Leone_Genito/Dataset/Tables_S1_to_S11.xlsx"
SHEET_NAME = 5

(X_train, X_val, X_test,
 y_train, y_val, y_test,
 X_train_resh, X_val_resh, X_test_resh,
 y_train_ohe, y_val_ohe, y_test_ohe,
 le, n_classes, dataset_genes) = load_and_preprocess_data(EXCEL_PATH, SHEET_NAME)

# -----------------------------------------------------------------------------
# TRAIN CNN
# -----------------------------------------------------------------------------
model_cnn_path = "/home/musimathicslab/Desktop/Leone_Genito/Leone/Models/model_pretrained_Ros_60.keras"
class_weights_path = "/home/musimathicslab/Desktop/Leone_Genito/Leone/Models/class_weights.txt"

cnn_model, history = train_cnn_model(
    X_train_resh, X_val_resh,
    y_train_ohe, y_val_ohe,
    n_classes=n_classes,
    class_weights_path=class_weights_path,
    model_save_path=model_cnn_path,
    patience=5,
    epochs=20,
    batch_size=32
)

# Plot storia training
plot_history(history)

# -----------------------------------------------------------------------------
# EVALUATE CNN sul Test Set
# -----------------------------------------------------------------------------
test_loss, test_acc, test_precision, test_recall, test_auc = cnn_model.evaluate(X_test_resh, y_test_ohe, verbose=0)
print("\n=== Risultati CNN sul Test set ===")
print(f"Loss: {test_loss:.4f}")
print(f"Acc: {test_acc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"AUC: {test_auc:.4f}")

preds_test = cnn_model.predict(X_test_resh)
y_pred_cnn = np.argmax(preds_test, axis=1)
print("\nClassification report (CNN):\n")
print(classification_report(y_test, y_pred_cnn, target_names=le.classes_))

# -----------------------------------------------------------------------------
# TRAIN SIAMESE
# -----------------------------------------------------------------------------
# Creazione DataFrame di train/val/test per la Siamese
x_train_df = dataset_genes.iloc[X_train.index].reset_index(drop=True)
x_val_df   = dataset_genes.iloc[X_val.index].reset_index(drop=True)
x_test_df  = dataset_genes.iloc[X_test.index].reset_index(drop=True)

train_ind = indices_save(x_train_df)
val_ind   = indices_save(x_val_df)
test_ind  = indices_save(x_test_df)

genes_len = x_train_df.drop(columns=['CANCER_TYPE']).shape[1]

# Creiamo il modello Siamese (con o senza caricamento pesi dal modello CNN)
siamese_model_path = "/home/musimathicslab/Desktop/Leone_Genito/Leone/Models/siamese_best_Ros_60.keras"

# Esempio: costruiamo la Siamese SENZA caricare i pesi del pretrained model
siamese = get_siamese_model(input_shape=(genes_len, 1),
                            pretrained_model=None,  # o cnn_model se vuoi caricare i pesi
                            layer_name='L2')

siamese.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Training siamese
siamese = train_siamese_model(
    siamese_model=siamese,
    x_train_df=x_train_df,
    train_ind=train_ind,
    x_val_df=x_val_df,
    val_ind=val_ind,
    genes_len=genes_len,
    model_save_path_siamese=siamese_model_path,
    n_iter=20000,
    batch_size=64,
    evaluate_every=100,
    patience=15
)

# -----------------------------------------------------------------------------
# TEST SIAMESE (One-Shot e Confusion Matrix)
# -----------------------------------------------------------------------------
# Test One-Shot su 9 classi, con 500 tasks
final_acc, preds_detail = test_oneshot(
    siamese,
    genes_len,
    x_test_df,
    test_ind,
    N=9,
    k=500,
    verbose=1
)

print(f"\nOne-Shot Accuracy (9-way) su 500 tasks: {final_acc:.2f}%")

# Mostra alcune predizioni
print("\nEsempio di predizioni One-Shot [pred_class, real_class]:\n")
for i, (pred_class, real_class) in enumerate(preds_detail[:10]):
    print(f"Task {i}: Predetto={le.classes_[pred_class]}, Reale={le.classes_[real_class]}")

# Calcola Confusion Matrix con la strategia del "prototipo"
y_true_siam, y_pred_siam, cm_siam = confusion_matrix_siamese(
    siamese_model=siamese,
    x_train_df=x_train_df,
    train_ind=train_ind,
    x_test_df=x_test_df,
    test_ind=test_ind,
    genes_len=genes_len,
    le=le
)

accuracy_siam = np.trace(cm_siam) / np.sum(cm_siam)
print(f"\nAccuracy (dalla Confusion Matrix Siamese): {accuracy_siam:.4f}")

print("\n=== Classification Report Siamese ===\n")
print(classification_report(y_true_siam, y_pred_siam, target_names=le.classes_))

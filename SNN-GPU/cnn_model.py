# cnn_model.py

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils import class_weight
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, Flatten, MaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def weighted_categorical_crossentropy(weights):
    """
    Restituisce una loss function che pesa le classi in base a 'weights'.
    'weights' deve avere dimensione [n_classes].
    """
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Normalizza y_pred
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # Clip per evitare log(0)
        y_pred = tf.clip_by_value(y_pred,
                                  tf.keras.backend.epsilon(),
                                  1 - tf.keras.backend.epsilon())
        # Calcolo della loss
        loss_val = y_true * tf.math.log(y_pred) * weights
        loss_val = -tf.reduce_sum(loss_val, axis=-1)
        return loss_val

    return loss

def create_cnn_model(input_shape, n_classes):
    """
    Crea e restituisce il modello CNN basato su Conv1D.
    """
    input_layer = Input(shape=input_shape)

    conv1 = Conv1D(filters=64, kernel_size=11, strides=1,
                   activation='relu', padding='same')(input_layer)
    mp1 = MaxPooling1D(pool_size=2)(conv1)
    drop1 = Dropout(0.3)(mp1)

    conv2 = Conv1D(filters=128, kernel_size=9, strides=1,
                   activation='relu', padding='same')(drop1)
    mp2 = MaxPooling1D(pool_size=2)(conv2)
    drop2 = Dropout(0.2)(mp2)

    conv3 = Conv1D(filters=256, kernel_size=7, strides=1,
                   activation='relu', padding='same')(drop2)
    mp3 = MaxPooling1D(pool_size=2)(conv3)
    drop3 = Dropout(0.2)(mp3)

    conv4 = Conv1D(filters=512, kernel_size=5, strides=1,
                   activation='relu', padding='same')(drop3)
    mp4 = MaxPooling1D(pool_size=2)(conv4)
    drop4 = Dropout(0.2)(mp4)

    flatten = Flatten()(drop4)
    dense = Dense(512, activation='relu')(flatten)
    output = Dense(n_classes, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output, name="classification")
    return model

def train_cnn_model(X_train_resh, X_val_resh,
                    y_train_ohe, y_val_ohe,
                    n_classes, class_weights_path,
                    model_save_path, patience=5, epochs=20, batch_size=32):
    """
    Allena il modello CNN con una loss pesata, restituisce il modello e lo history.
    """
    # Calcolo dei pesi di classe (bilanciati in base ai dati di training)
    # y_train_ohe ha forma (n_samples, n_classes), ma per compute_class_weight
    # serve la forma numerica (0..n_classes-1). Quindi troviamo la label index.
    y_train_class_indices = y_train_ohe.argmax(axis=1)

    # calcolo i pesi
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_class_indices),
        y=y_train_class_indices
    )

    # normalizzazione opzionale
    # weights = weights / np.sum(weights)
    print("Class Weights (non normalizzati):", weights)

    # Salvataggio dei pesi di classe su file (opzionale)
    with open(class_weights_path, 'w') as f:
        f.write("Class Weights:\n")
        f.write(str(weights.tolist()))

    # Crea il modello CNN
    input_shape = (X_train_resh.shape[1], 1)
    model = create_cnn_model(input_shape, n_classes)

    # Compilazione
    model.compile(
        loss=weighted_categorical_crossentropy(weights),
        optimizer=Adam(),
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )

    model.summary()

    # Callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    # Training
    history = model.fit(
        X_train_resh,
        y_train_ohe,
        validation_data=(X_val_resh, y_val_ohe),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint]
    )

    return model, history

def plot_history(history):
    """
    Traccia i grafici di training (loss e accuracy).
    """
    plt.figure(figsize=(12, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

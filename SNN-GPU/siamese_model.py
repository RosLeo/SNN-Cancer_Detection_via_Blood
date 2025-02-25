# siamese_model.py

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, Flatten, MaxPooling1D, Lambda
)
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

def initialize_bias(shape, name=None, dtype=None):
    """
    Inizializza i pesi di bias con una leggera perturbazione attorno a 0.5
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)

def last_layer(encoded_l, encoded_r, lyr_name='L2'):
    """
    Ultimo layer di confronto tra encoded_l e encoded_r.
    - 'L1': differenza assoluta
    - 'L2': L2 distance "normalizzata"
    - 'COS': similarità coseno
    """
    if lyr_name == 'L1':
        # L1-layer
        L1_layer = Lambda(lambda tensors: tf.keras.backend.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        dense_1 = Dense(512, activation='relu', bias_initializer=initialize_bias)(L1_distance)
        prediction = Dense(1, activation='sigmoid')(dense_1)
        return prediction

    elif lyr_name == 'L2':
        # L2-layer personalizzata
        L2_layer = Lambda(
            lambda tensors: (tensors[0] - tensors[1])**2
                            / (tensors[0] + tensors[1] + K.epsilon())
        )
        L2_distance = L2_layer([encoded_l, encoded_r])
        dense_1 = Dense(256, activation='relu', bias_initializer=initialize_bias)(L2_distance)
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(dense_1)
        return prediction

    else:
        # Cosine similarity
        cos_layer = Lambda(
            lambda tensors:
            K.sum(tensors[0] * tensors[1], axis=-1, keepdims=True) /
            (tf.keras.backend.l2_normalize(tensors[0], axis=-1) *
             tf.keras.backend.l2_normalize(tensors[1], axis=-1) + K.epsilon())
        )
        cos_distance = cos_layer([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(cos_distance)
        return prediction

def indices_save(dataset):
    """
    Crea un dict: { classe: [lista_indici] } prendendo la colonna 'CANCER_TYPE'.
    """
    cancer_map = {}
    for index, cancer_type in dataset['CANCER_TYPE'].items():
        if cancer_type not in cancer_map:
            cancer_map[cancer_type] = []
        cancer_map[cancer_type].append(index)
    return cancer_map

def get_siamese_branch(input_shape):
    """
    Ritorna una 'branch' CNN (convoluzionale) per la siamese.
    """
    branch = Sequential()
    branch.add(Input(shape=input_shape))

    branch.add(Conv1D(filters=64, kernel_size=11, strides=1,
                      activation='relu', padding='same'))
    branch.add(MaxPooling1D(pool_size=2))
    branch.add(Dropout(0.2))

    branch.add(Conv1D(filters=128, kernel_size=9, strides=1,
                      activation='relu', padding='same'))
    branch.add(MaxPooling1D(pool_size=2))
    branch.add(Dropout(0.2))

    branch.add(Conv1D(filters=256, kernel_size=7, strides=1,
                      activation='relu', padding='same'))
    branch.add(MaxPooling1D(pool_size=2))
    branch.add(Dropout(0.2))

    branch.add(Conv1D(filters=512, kernel_size=5, strides=1,
                      activation='relu', padding='same'))
    branch.add(MaxPooling1D(pool_size=2))

    branch.add(Flatten())
    branch.add(Dense(128, activation='relu'))

    return branch

def get_siamese_model(input_shape, pretrained_model=None, layer_name='L2'):
    """
    Costruisce il modello Siamese.
    Se `pretrained_model` non è None, carica i pesi nei layer corrispondenti.
    """
    # Crea la branch (una volta sola) e sostituisce i pesi (facoltativo)
    branch = get_siamese_branch(input_shape)

    if pretrained_model is not None:
        # Esempio di caricamento pesi dai layer del pretrained_model.
        # Qui si suppone che il modello pre-addestrato abbia:
        #   layer[1] => Conv1D(64,...)
        #   layer[3] => Conv1D(128,...)
        #   layer[5] => Conv1D(256,...)
        #   layer[7] => Conv1D(512,...)
        # Adatta l'indice se necessario.
        branch.layers[0].set_weights(pretrained_model.layers[1].get_weights())
        branch.layers[2].set_weights(pretrained_model.layers[3].get_weights())
        branch.layers[4].set_weights(pretrained_model.layers[5].get_weights())
        branch.layers[6].set_weights(pretrained_model.layers[7].get_weights())

    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)

    encoded_l = branch(left_input)
    encoded_r = branch(right_input)

    prediction = last_layer(encoded_l, encoded_r, lyr_name=layer_name)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_net

def get_batch(batch_size, x_df, class_indices, genes_len):
    """
    Crea batch di 'batch_size' coppie (pairs).
    Metà sono same-class, metà different-class.
    """
    pairs = [np.zeros((batch_size, genes_len, 1)) for _ in range(2)]
    targets = np.zeros((batch_size,))
    # la seconda metà del batch con '1' => same-class
    targets[batch_size // 2:] = 1

    categories = list(class_indices.keys())
    random.shuffle(categories)
    n_classes = len(categories)
    j = 0

    for i in range(batch_size):
        if j >= n_classes:
            random.shuffle(categories)
            j = 0

        category = categories[j]
        # prendi un campione
        idx_1 = random.choice(class_indices[category])
        pairs[0][i, :, 0] = x_df.drop(columns=['CANCER_TYPE']).values[idx_1]

        if i >= batch_size // 2:
            # same class
            idx_2 = random.choice(class_indices[category])
        else:
            # different class
            diff_cats = [c for c in categories if c != category]
            category_2 = random.choice(diff_cats)
            idx_2 = random.choice(class_indices[category_2])

        pairs[1][i, :, 0] = x_df.drop(columns=['CANCER_TYPE']).values[idx_2]
        j += 1
    return pairs, targets

def make_oneshot_task(genes_len, x_test, class_test_ind, N):
    """
    Crea un task one-shot N-way.
    - test_image = un campione
    - support_set = N campioni, uno per ciascuna delle N classi
    - targets = array con 1 in posizione corrispondente alla stessa classe, 0 altrove
    """
    X = x_test.drop(columns=['CANCER_TYPE']).values
    class_test_dic = class_test_ind

    list_N_samples = random.sample(list(class_test_dic.keys()), N)
    true_category = list_N_samples[0]
    out_ind = np.array([random.sample(class_test_dic[c], 2) for c in list_N_samples])
    indices = out_ind[:, 1]
    ex1 = out_ind[0, 0]

    test_image = np.asarray([X[ex1]] * N).reshape(N, genes_len, 1)
    support_set = X[indices].reshape(N, genes_len, 1)

    targets = np.zeros((N,))
    targets[0] = 1

    targets, test_image, support_set, list_N_samples = shuffle(
        targets, test_image, support_set, list_N_samples, random_state=42
    )
    pairs = [test_image, support_set]
    return pairs, targets, true_category, list_N_samples

def test_oneshot(model, genes_len, x_test, class_test_ind, N, k, verbose=0):
    """
    Test average N-way one-shot learning accuracy del modello siamese
    su k one-shot tasks.
    Ritorna (percent_correct, predictions_list).
    predictions_list contiene tuple (pred_class, real_class).
    """
    n_correct = 0
    predictions = []

    for _ in range(k):
        inputs, targets, true_category, list_N_samples = make_oneshot_task(
            genes_len, x_test, class_test_ind, N
        )
        probs = model.predict(inputs, verbose=0)  # shape (N, 1)

        pred_idx = np.argmax(probs)
        real_idx = np.argmax(targets)

        predicted_class = list_N_samples[pred_idx]
        real_class = list_N_samples[real_idx]
        predictions.append((predicted_class, real_class))

        if pred_idx == real_idx:
            n_correct += 1

    percent_correct = 100.0 * n_correct / k
    if verbose:
        print(f"One-shot accuracy: {percent_correct:.2f}%")

    return percent_correct, predictions

def get_class_references(x_train_df, train_ind, genes_len):
    """
    Ritorna un dict {class_label: feature_vector} con un solo riferimento
    per classe dal train set.
    """
    references = {}
    for c in train_ind.keys():
        ref_index = train_ind[c][0]  # ad esempio il primo indice
        ref_vector = x_train_df.drop(columns=['CANCER_TYPE']).iloc[ref_index].values
        references[c] = ref_vector
    return references

def confusion_matrix_siamese(siamese_model,
                            x_train_df, train_ind,
                            x_test_df, test_ind,
                            genes_len, le):
    """
    Calcola la confusion matrix (9 classi) usando una strategia a "prototipi":
    - Per ogni sample di test si confronta con 1 riferimento di ciascuna classe.
    - Si predice la classe con la probabilità "same class" più alta.
    """
    references = get_class_references(x_train_df, train_ind, genes_len)
    class_labels = sorted(train_ind.keys())

    y_true = []
    y_pred = []

    for idx, row in x_test_df.iterrows():
        true_label = row['CANCER_TYPE']
        x_features = row.drop(labels=['CANCER_TYPE']).values.reshape(1, genes_len, 1)

        best_class = None
        best_prob = -1.0

        # Confronto con tutte le classi
        for c in class_labels:
            ref_vec = references[c].reshape(1, genes_len, 1)
            pair = [x_features, ref_vec]
            prob = siamese_model.predict(pair, verbose=0)[0][0]  # scalare
            if prob > best_prob:
                best_prob = prob
                best_class = c

        y_pred.append(best_class)
        y_true.append(true_label)

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix - Siamese (N-way classification)")
    plt.show()

    return y_true, y_pred, cm

def train_siamese_model(siamese_model,
                        x_train_df, train_ind,
                        x_val_df, val_ind,
                        genes_len,
                        model_save_path_siamese,
                        n_iter=20000, batch_size=64,
                        evaluate_every=100, patience=15):
    """
    Esegue il training iterativo della rete Siamese con un semplice training on-batch.
    """
    train_losses = []
    val_acc_history = []
    iter_history = []

    best_val_acc = -1
    no_improve_count = 0

    for i in range(1, n_iter + 1):
        pairs, targets = get_batch(batch_size, x_train_df, train_ind, genes_len)
        loss, acc = siamese_model.train_on_batch(pairs, targets)
        train_losses.append(loss)

        if i % 100 == 0:
            print(f"Iter {i} - loss: {loss:.4f}, acc: {acc:.4f}")

        # valutazione su validation set ogni evaluate_every
        if i % evaluate_every == 0:
            val_pairs, val_targets = get_batch(len(x_val_df), x_val_df, val_ind, genes_len)
            val_loss, val_acc_ = siamese_model.evaluate(val_pairs, val_targets, verbose=0)
            print(f"[VAL] Iter {i} => val_loss={val_loss:.4f}, val_acc={val_acc_:.4f}")
            val_acc_history.append(val_acc_)
            iter_history.append(i)

            # Early Stopping
            if val_acc_ > best_val_acc:
                best_val_acc = val_acc_
                no_improve_count = 0
                siamese_model.save(model_save_path_siamese)
                print("  Nuovo best model salvato!")
            else:
                no_improve_count += 1
                print(f"  no_improve_count={no_improve_count}/{patience}")
                if no_improve_count >= patience:
                    print("Early stopping! Il modello ha smesso di migliorare.")
                    break

    # Plot dei risultati
    plt.figure(figsize=(12, 5))

    # Training loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel("Iterazioni")
    plt.ylabel("Loss")
    plt.title("Training Loss (Siamese)")
    plt.legend()

    # Validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(iter_history, val_acc_history, marker='o', color='red', label='Val Accuracy')
    plt.xlabel("Iterazioni")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy (Siamese)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return siamese_model

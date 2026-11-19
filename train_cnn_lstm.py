
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    LSTM, Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 2. LOAD CURRENT DATASET
# ==========================================

train_data = np.load("results/split_data/train_dataset.npz")
test_data = np.load("results/split_data/test_dataset.npz")

X = train_data["X"]
y = train_data["y"]
record_ids = train_data["record_ids"]

X_test = test_data["X"]
y_test = test_data["y"]

print("========== CNN-LSTM DATA ==========")
print("Training data:", X.shape)
print("Testing data:", X_test.shape)

# ==========================================
# 3. RECORD-BASED VALIDATION
# ==========================================

unique_records = np.unique(record_ids)
validation_records = unique_records[-4:]

train_mask = ~np.isin(record_ids, validation_records)
val_mask = np.isin(record_ids, validation_records)

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[val_mask]
y_val = y[val_mask]

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Validation records:", validation_records)

# ==========================================
# 4. CLASS WEIGHTS
# ==========================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(c): float(w)
    for c, w in zip(classes, weights)
}

print("Class weights:", class_weights)

# ==========================================
# 5. BUILD CNN-LSTM MODEL
# ==========================================

model = Sequential([
    Input(shape=(200, 2)),

    Conv1D(64, kernel_size=5, activation="relu"),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),

    LSTM(32, return_sequences=True),
    Dropout(0.3),

    LSTM(16),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# ==========================================
# 6. EARLY STOPPING
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\n========== CNN-LSTM TRAINING ==========")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=128,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 8. TEST MODEL
# ==========================================

print("\n========== CNN-LSTM TEST RESULTS ==========")

probabilities = model.predict(X_test, batch_size=128).ravel()
y_pred = (probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))

# ==========================================
# 9. SAVE MODEL AND RESULTS
# ==========================================

os.makedirs("results/models", exist_ok=True)

model.save("results/models/cnn_lstm_model.keras")

np.savez(
    "results/models/cnn_lstm_history.npz",
    **history.history
)

np.savez_compressed(
    "results/models/cnn_lstm_predictions.npz",
    y_true=y_test,
    y_pred=y_pred,
    probabilities=probabilities
)

np.savez(
    "results/models/cnn_lstm_metrics.npz",
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1_score=f1
)

print("\nCNN-LSTM MODEL SAVED SUCCESSFULLY!")
print("Location: results/models")


import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. PATHS
# ==========================================

TRAIN_PATH = "results/split_data/train_dataset.npz"
TEST_PATH = "results/split_data/test_dataset.npz"
MODEL_DIR = "results/models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================

with np.load(TRAIN_PATH) as data:
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int32)
    record_ids = data["record_ids"].astype(str)

with np.load(TEST_PATH) as data:
    X_test = data["X"].astype(np.float32)
    y_test = data["y"].astype(np.int32)

# ==========================================
# 3. RECORD-BASED VALIDATION SPLIT
# ==========================================

unique_records = np.unique(record_ids)

validation_records = unique_records[-4:]

train_mask = ~np.isin(record_ids, validation_records)
val_mask = np.isin(record_ids, validation_records)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# ==========================================
# 4. CLASS WEIGHTS
# ==========================================

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes.tolist(), weights.tolist()))

print("Class weights:", class_weights)

# ==========================================
# 5. BUILD LSTM
# ==========================================

model = Sequential([
    Input(shape=(200, 2)),

    LSTM(32, return_sequences=True),
    Dropout(0.3),

    LSTM(16),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# ==========================================
# 6. COMPILE
# ==========================================

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
# 7. TRAIN
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

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
# 8. EVALUATE
# ==========================================

results = model.evaluate(
    X_test,
    y_test,
    verbose=1,
    return_dict=True
)

print("\n========== LSTM TEST RESULTS ==========")

for name, value in results.items():
    print(f"{name}: {value:.4f}")

# ==========================================
# 9. SAVE MODEL AND HISTORY
# ==========================================

model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))

np.savez(
    os.path.join(MODEL_DIR, "lstm_history.npz"),
    **history.history
)

# ==========================================
# 10. SAVE PREDICTIONS
# ==========================================

probabilities = model.predict(X_test, batch_size=128).ravel()
predictions = (probabilities >= 0.5).astype(np.uint8)

np.savez_compressed(
    os.path.join(MODEL_DIR, "lstm_predictions.npz"),
    y_true=y_test,
    y_pred=predictions,
    probabilities=probabilities
)

print("\nLSTM MODEL SAVED SUCCESSFULLY!")
print("Location:", MODEL_DIR)

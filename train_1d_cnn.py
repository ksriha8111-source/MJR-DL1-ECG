
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    Dropout, GlobalAveragePooling1D, Dense
)
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

with np.load(TRAIN_PATH) as train_data:
    X_train = train_data["X"].astype(np.float32)
    y_train = train_data["y"].astype(np.int32)

with np.load(TEST_PATH) as test_data:
    X_test = test_data["X"].astype(np.float32)
    y_test = test_data["y"].astype(np.int32)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ==========================================
# 3. CLASS WEIGHTS
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
# 4. BUILD CNN
# ==========================================

model = Sequential([
    Input(shape=(200, 2)),

    Conv1D(32, 5, activation="relu"),
    MaxPooling1D(2),

    Conv1D(64, 5, activation="relu"),
    MaxPooling1D(2),

    Conv1D(128, 3, activation="relu"),

    GlobalAveragePooling1D(),

    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

# ==========================================
# 5. COMPILE
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
# 6. TRAIN
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 7. EVALUATE
# ==========================================

results = model.evaluate(X_test, y_test, verbose=1)

print("\n========== CNN TEST RESULTS ==========")

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# ==========================================
# 8. SAVE MODEL AND HISTORY
# ==========================================

model.save(os.path.join(MODEL_DIR, "1d_cnn_model.keras"))

np.savez(
    os.path.join(MODEL_DIR, "1d_cnn_history.npz"),
    **history.history
)

# ==========================================
# 9. SAVE PREDICTIONS
# ==========================================

probabilities = model.predict(X_test, batch_size=128).ravel()
predictions = (probabilities >= 0.5).astype(np.uint8)

np.savez_compressed(
    os.path.join(MODEL_DIR, "1d_cnn_predictions.npz"),
    y_true=y_test,
    y_pred=predictions,
    probabilities=probabilities
)

print("\nCNN MODEL SAVED SUCCESSFULLY!")
print("Location:", MODEL_DIR)

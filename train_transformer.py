
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D,
    Add, Embedding
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

print("========== TRANSFORMER DATA ==========")
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
# 5. TRANSFORMER BLOCK
# ==========================================

def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.2):

    attention = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout
    )(inputs, inputs)

    attention = Dropout(dropout)(attention)

    x = Add()([inputs, attention])
    x = LayerNormalization(epsilon=1e-6)(x)

    feed_forward = Dense(ff_dim, activation="relu")(x)
    feed_forward = Dropout(dropout)(feed_forward)
    feed_forward = Dense(int(inputs.shape[-1]))(feed_forward)

    x = Add()([x, feed_forward])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x

# ==========================================
# 6. BUILD TRANSFORMER MODEL
# ==========================================

inputs = Input(shape=(200, 2))

x = Dense(64)(inputs)

positions = tf.range(start=0, limit=200, delta=1)

position_embedding = Embedding(
    input_dim=200,
    output_dim=64
)(positions)

x = x + position_embedding

x = transformer_block(
    x,
    head_size=16,
    num_heads=4,
    ff_dim=128,
    dropout=0.2
)

x = transformer_block(
    x,
    head_size=16,
    num_heads=4,
    ff_dim=128,
    dropout=0.2
)

x = GlobalAveragePooling1D()(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)

outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inputs, outputs=outputs)

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
# 7. EARLY STOPPING
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ==========================================
# 8. TRAIN TRANSFORMER
# ==========================================

print("\n========== TRANSFORMER TRAINING ==========")

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
# 9. TEST TRANSFORMER
# ==========================================

print("\n========== TRANSFORMER TEST RESULTS ==========")

probabilities = model.predict(
    X_test,
    batch_size=128
).ravel()

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
# 10. SAVE MODEL AND RESULTS
# ==========================================

os.makedirs("results/models", exist_ok=True)

model.save("results/models/transformer_model.keras")

np.savez(
    "results/models/transformer_history.npz",
    **history.history
)

np.savez_compressed(
    "results/models/transformer_predictions.npz",
    y_true=y_test,
    y_pred=y_pred,
    probabilities=probabilities
)

np.savez(
    "results/models/transformer_metrics.npz",
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1_score=f1
)

print("\nTRANSFORMER MODEL SAVED SUCCESSFULLY!")
print("Location: results/models")

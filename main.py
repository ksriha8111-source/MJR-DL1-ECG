
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import wfdb
import wfdb.processing
import tensorflow as tf

from pathlib import Path
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="MJR-DL1 | ECG Event Detection",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# 2. WHITE AND BLUE THEME
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F8FC;
    color: #183153;
}

h1, h2, h3 {
    color: #145DA0 !important;
    font-weight: 700;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #D5E5F5;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(20, 93, 160, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #527A9F;
}

[data-testid="stMetricValue"] {
    color: #145DA0;
}

.stButton > button {
    background-color: #1976D2;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton > button:hover {
    background-color: #0D47A1;
    color: white;
}

</style>
""", unsafe_allow_html=True)


st.title("MJR-DL1 - Medical Signal Event Detection")

st.write(
    "ECG classification using four deep-learning models."
)

# =====================================================
# 3. PROJECT PATHS AND MODEL FILES
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
SPLIT_DIR = RESULTS_DIR / "split_data"

MODEL_FILES = {
    "1D CNN": "1d_cnn_model.keras",
    "LSTM": "lstm_model.keras",
    "CNN-LSTM": "cnn_lstm_model.keras",
    "Transformer": "transformer_model.keras"
}

PREDICTION_FILES = {
    "1D CNN": "1d_cnn_predictions.npz",
    "LSTM": "lstm_predictions.npz",
    "CNN-LSTM": "cnn_lstm_predictions.npz",
    "Transformer": "transformer_predictions.npz"
}

LABEL_NAMES = {
    0: "Normal",
    1: "Abnormal"
}

# =====================================================
# 4. LOAD SAVED DEEP LEARNING MODEL
# =====================================================

@st.cache_resource
def load_deep_learning_model(model_path):
    return tf.keras.models.load_model(model_path)


# =====================================================
# 5. LOAD TEST DATA
# =====================================================

@st.cache_data
def load_test_data():

    path = SPLIT_DIR / "test_dataset.npz"

    with np.load(path) as data:

        X_test = data["X"]
        y_test = data["y"]

    return X_test, y_test


# =====================================================
# 6. LOAD SAVED MODEL PREDICTIONS
# =====================================================

def find_array(data, possible_names):

    for name in possible_names:

        if name in data.files:
            return data[name]

    return None


@st.cache_data
def load_predictions(filename, test_labels):

    path = MODEL_DIR / filename

    with np.load(path) as data:

        probabilities = find_array(
            data,
            [
                "probabilities",
                "y_prob",
                "y_probability",
                "probability"
            ]
        )

        predictions = find_array(
            data,
            [
                "y_pred",
                "predictions",
                "predicted_labels",
                "y_prediction"
            ]
        )

        saved_labels = find_array(
            data,
            [
                "y_true",
                "true_labels",
                "labels"
            ]
        )

    y_true = (
        np.asarray(saved_labels).reshape(-1)
        if saved_labels is not None
        else np.asarray(test_labels).reshape(-1)
    )

    if probabilities is not None:

        probabilities = np.asarray(probabilities)

        if probabilities.ndim == 2 and probabilities.shape[1] == 2:

            probabilities = probabilities[:, 1]

        else:

            probabilities = probabilities.reshape(-1)

    if predictions is not None:

        predictions = np.asarray(predictions)

        if predictions.ndim == 2 and predictions.shape[1] == 2:

            predictions = np.argmax(predictions, axis=1)

        else:

            predictions = predictions.reshape(-1)

        if not np.all(
            np.isin(np.unique(predictions), [0, 1])
        ):

            predictions = (
                predictions >= 0.5
            ).astype(int)

    elif probabilities is not None:

        predictions = (
            probabilities >= 0.5
        ).astype(int)

    else:

        raise ValueError(
            f"No predictions found in {filename}"
        )

    if len(y_true) != len(predictions):

        raise ValueError(
            f"Prediction count mismatch: {filename}"
        )

    if len(y_true) != len(test_labels):

        raise ValueError(
            f"Test label count mismatch: {filename}"
        )

    if probabilities is not None and len(probabilities) != len(y_true):

        raise ValueError(
            f"Probability count mismatch: {filename}"
        )

    return y_true, predictions, probabilities


# =====================================================
# 7. CALCULATE MODEL METRICS
# =====================================================

def calculate_metrics(y_true, y_pred):

    return {

        "Accuracy": accuracy_score(
            y_true, y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1-Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }


# =====================================================
# 8. LIVE ECG UPLOAD AND PREDICTION
# =====================================================

st.divider()

st.header("1. Upload ECG and Predict")

st.write(
    "Upload a compatible MIT-BIH ECG recording. "
    "Both .hea and .dat files are required."
)

uploaded_files = st.file_uploader(
    "Choose ECG .hea and .dat files",
    type=["hea", "dat"],
    accept_multiple_files=True
)

selected_model = st.selectbox(
    "Select Deep Learning Model",
    list(MODEL_FILES.keys()),
    index=2
)

st.caption(
    "CNN-LSTM currently has the highest test F1-score "
    "in this project. You can select any of the four models."
)

if st.button("Run ECG Prediction", type="primary"):

    header_files = [
        file for file in uploaded_files or []
        if file.name.lower().endswith(".hea")
    ]

    data_files = [
        file for file in uploaded_files or []
        if file.name.lower().endswith(".dat")
    ]

    if len(header_files) != 1 or len(data_files) != 1:

        st.error(
            "Please upload exactly one .hea file "
            "and one .dat file."
        )

    else:

        try:

            header_file = header_files[0]
            data_file = data_files[0]

            with tempfile.TemporaryDirectory() as temp_folder:

                temp_path = Path(temp_folder)

                # Save header
                header_path = temp_path / header_file.name

                header_path.write_bytes(
                    header_file.getvalue()
                )

                # Read header contents
                header_text = header_file.getvalue().decode(
                    "utf-8",
                    errors="ignore"
                )

                header_lines = [
                    line.strip()
                    for line in header_text.splitlines()
                    if line.strip()
                    and not line.strip().startswith("#")
                ]

                signal_filenames = sorted({

                    Path(line.split()[0]).name

                    for line in header_lines[1:]

                    if line.split()[0].lower().endswith(".dat")

                })

                if len(signal_filenames) != 1:

                    raise ValueError(
                        "This application currently supports "
                        "one .dat signal file."
                    )

                if Path(signal_filenames[0]).name != data_file.name:

                    raise ValueError(
                        "The uploaded .dat filename does not match "
                        "the filename referenced in the .hea file."
                    )

                # Save signal file
                signal_path = temp_path / signal_filenames[0]

                signal_path.write_bytes(
                    data_file.getvalue()
                )

                # Load ECG recording
                record_base = (
                    temp_path / Path(header_file.name).stem
                )

                record = wfdb.rdrecord(
                    str(record_base)
                )

                signal = record.p_signal
                sampling_rate = record.fs

                if signal is None:

                    raise ValueError(
                        "Unable to read ECG signal."
                    )

                if sampling_rate != 360:

                    raise ValueError(
                        f"Expected 360 Hz sampling frequency, "
                        f"but received {sampling_rate} Hz."
                    )

                if signal.shape[1] != 2:

                    raise ValueError(
                        "This project requires exactly "
                        "two ECG channels."
                    )

                # Detect R-peaks
                st.write("Detecting heartbeats...")

                r_peaks = wfdb.processing.xqrs_detect(
                    sig=signal[:, 0],
                    fs=sampling_rate,
                    verbose=False
                )

                # Extract heartbeat segments
                segments = []
                valid_r_peaks = []

                for peak in r_peaks:

                    start = int(peak) - 90
                    end = int(peak) + 110

                    if start >= 0 and end <= len(signal):

                        segment = signal[start:end, :]

                        if segment.shape == (200, 2):

                            segments.append(segment)
                            valid_r_peaks.append(peak)

                if not segments:

                    raise ValueError(
                        "No valid heartbeat segments detected."
                    )

                X_upload = np.asarray(
                    segments,
                    dtype=np.float32
                )

                valid_r_peaks = np.asarray(
                    valid_r_peaks
                )

                # Load selected model
                model_path = (
                    MODEL_DIR / MODEL_FILES[selected_model]
                )

                model = load_deep_learning_model(
                    str(model_path)
                )

                # Predict
                st.write(
                    f"Running {selected_model} prediction..."
                )

                probabilities = model.predict(
                    X_upload,
                    batch_size=128,
                    verbose=0
                ).reshape(-1)

                predictions = (
                    probabilities >= 0.5
                ).astype(int)

                # Store results
                st.session_state["live_result"] = {

                    "signal": signal,

                    "sampling_rate": sampling_rate,

                    "r_peaks": valid_r_peaks,

                    "segments": X_upload,

                    "probabilities": probabilities,

                    "predictions": predictions,

                    "model_name": selected_model

                }

            st.success(
                "ECG prediction completed successfully."
            )

        except Exception as error:

            st.error(
                f"ECG prediction failed: {error}"
            )


# =====================================================
# 9. DISPLAY LIVE ECG RESULTS
# =====================================================

if "live_result" in st.session_state:

    live = st.session_state["live_result"]

    signal = live["signal"]

    fs = live["sampling_rate"]

    predictions = live["predictions"]

    probabilities = live["probabilities"]

    segments = live["segments"]

    r_peaks = live["r_peaks"]

    st.divider()

    st.header("2. Live ECG Analysis")

    # -------------------------------------------------
    # UPLOADED ECG WAVEFORM
    # -------------------------------------------------

    st.subheader("Uploaded ECG Waveform")

    display_samples = min(
        len(signal),
        fs * 10
    )

    time_axis = (
        np.arange(display_samples) / fs
    )

    fig, ax = plt.subplots(
        figsize=(15, 5)
    )

    ax.plot(
        time_axis,
        signal[:display_samples, 0],
        color="#1976D2",
        linewidth=1.1,
        label="MLII"
    )

    ax.plot(
        time_axis,
        signal[:display_samples, 1],
        color="#00A6A6",
        linewidth=1.0,
        alpha=0.85,
        label="V5"
    )

    visible_peaks = r_peaks[
        r_peaks < display_samples
    ]

    ax.scatter(
        visible_peaks / fs,
        signal[visible_peaks, 0],
        color="red",
        s=22,
        marker="o",
        label="Detected R-peaks",
        zorder=5
    )

    ax.xaxis.set_major_locator(
        MultipleLocator(0.2)
    )

    ax.xaxis.set_minor_locator(
        MultipleLocator(0.04)
    )

    ax.grid(
        which="major",
        color="#A8C8E8",
        linewidth=0.7
    )

    ax.grid(
        which="minor",
        color="#E2EDF7",
        linewidth=0.4
    )

    ax.set_title(
        "ECG Signal with Detected R-Peaks",
        fontsize=15,
        fontweight="bold",
        color="#145DA0"
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Signal Amplitude")

    ax.set_xlim(
        0,
        display_samples / fs
    )

    ax.legend()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # -------------------------------------------------
    # PREDICTION SUMMARY
    # -------------------------------------------------

    total_beats = len(predictions)

    normal_count = int(
        np.sum(predictions == 0)
    )

    abnormal_count = int(
        np.sum(predictions == 1)
    )

    st.subheader("Prediction Summary")

    normal_percentage = (
        normal_count / total_beats
    ) * 100

    abnormal_percentage = (
        abnormal_count / total_beats
    ) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Heartbeats",
        f"{total_beats:,}",
        "Detected beats"
    )

    col2.metric(
        "Normal Heartbeats",
        f"{normal_count:,}",
        f"{normal_percentage:.2f}% of total"
    )

    col3.metric(
        "Abnormal Heartbeats",
        f"{abnormal_count:,}",
        f"{abnormal_percentage:.2f}% of total"
    )

    st.write(
        f"**Selected Model:** {live['model_name']}"
    )

    # -------------------------------------------------
    # HEARTBEAT DISTRIBUTION CHART
    # -------------------------------------------------

    st.subheader("Heartbeat Distribution")

    distribution_df = pd.DataFrame({

        "Heartbeat Type": [
            "Normal",
            "Abnormal"
        ],

        "Count": [
            normal_count,
            abnormal_count
        ]

    })

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    distribution_df.set_index(
        "Heartbeat Type"
    ).plot(
        kind="bar",
        ax=ax,
        legend=False,
        width=0.5
    )

    ax.set_ylabel(
        "Number of Heartbeats"
    )

    ax.set_xlabel(
        "Classification"
    )

    ax.set_title(
        "Normal vs Abnormal Heartbeat Distribution"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # -------------------------------------------------
    # CREATE RESULT DATAFRAME
    # -------------------------------------------------

    result_df = pd.DataFrame({

        "Heartbeat": np.arange(
            1,
            total_beats + 1
        ),

        "Prediction": [
            LABEL_NAMES[int(value)]
            for value in predictions
        ],

        "Abnormal Probability (%)": (
            probabilities * 100
        ).round(2)

    })

    # -------------------------------------------------
    # NORMAL HEARTBEATS ONLY
    # -------------------------------------------------

    st.subheader("Normal Heartbeats")

    normal_df = result_df[
        result_df["Prediction"] == "Normal"
    ]

    if not normal_df.empty:

        st.dataframe(
            normal_df,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "No Normal heartbeats detected by this model."
        )

    # -------------------------------------------------
    # INDIVIDUAL HEARTBEAT PREDICTION
    # -------------------------------------------------

    st.subheader(
        "Individual Heartbeat Prediction"
    )

    beat_index = st.slider(
        "Select detected heartbeat",
        min_value=0,
        max_value=total_beats - 1,
        value=0,
        key="live_beat_slider"
    )

    selected_prediction = int(
        predictions[beat_index]
    )

    abnormal_score = float(
        probabilities[beat_index]
    )

    # Individual heartbeat waveform
    heartbeat = segments[beat_index]

    fig, ax = plt.subplots(
        figsize=(13, 5)
    )

    time_axis = (
        np.arange(200) - 90
    ) / fs

    ax.plot(
        time_axis,
        heartbeat[:, 0],
        color="#1976D2",
        linewidth=1.5,
        label="MLII"
    )

    ax.plot(
        time_axis,
        heartbeat[:, 1],
        color="#00A6A6",
        linewidth=1.3,
        alpha=0.85,
        label="V5"
    )

    ax.axvline(
        x=0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Detected R-peak"
    )

    ax.scatter(
        [0],
        [heartbeat[90, 0]],
        color="red",
        s=55,
        zorder=5
    )

    ax.xaxis.set_major_locator(
        MultipleLocator(0.1)
    )

    ax.xaxis.set_minor_locator(
        MultipleLocator(0.02)
    )

    ax.grid(
        which="major",
        color="#A8C8E8",
        linewidth=0.7
    )

    ax.grid(
        which="minor",
        color="#E2EDF7",
        linewidth=0.4
    )

    ax.set_title(
        f"Heartbeat {beat_index + 1} - ECG Analysis",
        fontsize=15,
        fontweight="bold",
        color="#145DA0"
    )

    ax.set_xlabel(
        "Time relative to R-peak (seconds)"
    )

    ax.set_ylabel(
        "Signal Amplitude"
    )

    ax.legend()

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # -------------------------------------------------
    # PREDICTION RESULT
    # -------------------------------------------------

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Class",
        LABEL_NAMES[selected_prediction]
    )

    col2.metric(
        "Predicted Abnormal Probability",
        f"{abnormal_score * 100:.2f}%"
    )

    st.caption(
        "The model uses a 0.5 probability threshold. "
        "This score is not a calibrated clinical confidence value."
    )

    # -------------------------------------------------
    # HEARTBEAT ANALYSIS EXPLANATION
    # -------------------------------------------------

    st.subheader(
        "Heartbeat Analysis Explanation"
    )

    if selected_prediction == 0:

        st.success(
            "Normal Classification"
        )

        st.write(
            "The selected heartbeat was classified as Normal "
            "based on patterns learned during model training."
        )

    else:

        st.warning(
            "Abnormal Classification"
        )

        st.write(
            "The model detected a pattern associated with "
            "abnormal training labels."
        )

        st.caption(
            "The model cannot identify the exact rhythm, "
            "disease, or medical cause."
        )

    st.write(
        f"Model abnormal probability: "
        f"{abnormal_score * 100:.2f}%"
    )

    # -------------------------------------------------
    # ALL HEARTBEAT PREDICTIONS
    # -------------------------------------------------

    st.subheader(
        "Heartbeat Prediction Details"
    )

    st.dataframe(
        result_df,
        width="stretch",
        hide_index=True
    )

    st.warning(
        "Research project only. This application does not "
        "provide a medical diagnosis and must not be used "
        "for medical decisions."
    )


# =====================================================
# 10. DEEP LEARNING MODEL COMPARISON
# =====================================================

st.divider()

st.header(
    "3. Deep Learning Model Comparison"
)

try:

    X_test, y_test = load_test_data()

    results = {}

    metric_rows = []

    for model_name, filename in PREDICTION_FILES.items():

        y_true, y_pred, probabilities = load_predictions(
            filename,
            y_test
        )

        metrics = calculate_metrics(
            y_true,
            y_pred
        )

        results[model_name] = {

            "y_true": y_true,

            "y_pred": y_pred,

            "probabilities": probabilities,

            "metrics": metrics

        }

        metric_rows.append({

            "Model": model_name,

            **metrics

        })

    metrics_df = pd.DataFrame(
        metric_rows
    )

except Exception as error:

    st.error(
        f"Unable to load model comparison: {error}"
    )

    st.stop()


# -----------------------------------------------------
# MODEL COMPARISON TABLE
# -----------------------------------------------------

st.subheader(
    "Model Performance Metrics"
)

display_df = metrics_df.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]:

    display_df[column] = (
        display_df[column] * 100
    ).round(2)

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)


# =====================================================
# 11. BEST PERFORMING MODEL
# =====================================================

st.subheader(
    "Best Performing Model"
)

best_row = metrics_df.loc[
    metrics_df["F1-Score"].idxmax()
]

best_model = best_row["Model"]

best_f1 = (
    best_row["F1-Score"] * 100
)

col1, col2 = st.columns(2)

col1.metric(
    "Best Model",
    best_model
)

col2.metric(
    "Best F1-Score",
    f"{best_f1:.2f}%"
)

st.success(
    f"{best_model} currently has the highest test F1-score."
)

st.caption(
    "This is a research comparison on the project's test "
    "dataset, not a clinical ranking."
)


# =====================================================
# 12. MODEL PERFORMANCE CHART
# =====================================================

st.subheader(
    "Model Performance Comparison Chart"
)

chart_df = metrics_df.set_index(
    "Model"
)[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]
] * 100

fig, ax = plt.subplots(
    figsize=(12, 5)
)

chart_df.plot(
    kind="bar",
    ax=ax,
    width=0.75,
    edgecolor="white"
)

ax.set_title(
    "Performance Comparison of Deep Learning Models",
    fontsize=15,
    fontweight="bold",
    color="#145DA0"
)

ax.set_xlabel(
    "Deep Learning Models"
)

ax.set_ylabel(
    "Performance (%)"
)

ax.set_ylim(
    0,
    100
)

ax.legend(
    title="Evaluation Metrics"
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.set_facecolor("white")
fig.patch.set_facecolor("white")

plt.xticks(rotation=0)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


# =====================================================
# 13. CONFUSION MATRIX - ALL FOUR MODELS
# =====================================================

st.divider()

st.header(
    "4. Confusion Matrix Analysis"
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9)
)

for ax, (model_name, result) in zip(
    axes.flatten(),
    results.items()
):

    cm = confusion_matrix(
        result["y_true"],
        result["y_pred"],
        labels=[0, 1]
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    ax.set_title(
        model_name,
        fontsize=13,
        fontweight="bold",
        color="#145DA0"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Normal", "Abnormal"]
    )

    ax.set_yticklabels(
        ["Normal", "Abnormal"]
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="black",
                fontsize=13,
                fontweight="bold"
            )

fig.colorbar(
    image,
    ax=axes.ravel().tolist(),
    shrink=0.8
)

fig.suptitle(
    "Confusion Matrix Comparison",
    fontsize=16,
    fontweight="bold",
    color="#145DA0"
)

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


# =====================================================
# 14. TEST DATASET ECG EXAMPLE
# =====================================================

st.divider()

st.header(
    "5. Test Dataset ECG Example"
)

sample_index = st.slider(
    "Select test ECG sample",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    key="test_ecg_slider"
)

ecg_sample = X_test[sample_index]

true_label = int(
    y_test[sample_index]
)

fig, ax = plt.subplots(
    figsize=(12, 4)
)

ax.plot(
    ecg_sample[:, 0],
    color="#1976D2",
    label="ECG Channel 1"
)

ax.plot(
    ecg_sample[:, 1],
    color="#00A6A6",
    label="ECG Channel 2",
    alpha=0.75
)

ax.set_title(
    "Test ECG Heartbeat Waveform",
    fontsize=14,
    fontweight="bold",
    color="#145DA0"
)

ax.set_xlabel(
    "Sample Number"
)

ax.set_ylabel(
    "Signal Amplitude"
)

ax.legend()

ax.grid(
    True,
    alpha=0.3
)

ax.set_facecolor("white")
fig.patch.set_facecolor("white")

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


# =====================================================
# 15. ACTUAL DATASET LABEL
# =====================================================

st.subheader(
    "Actual Dataset Label"
)

st.metric(
    "Actual Class",
    LABEL_NAMES.get(
        true_label,
        "Unknown"
    )
)


# =====================================================
# 16. PREDICTIONS FROM ALL FOUR MODELS
# =====================================================

st.subheader(
    "Predictions from All Four Models"
)

prediction_rows = []

for model_name, result in results.items():

    prediction = int(
        result["y_pred"][sample_index]
    )

    probability = result["probabilities"]

    probability_text = (

        f"{float(probability[sample_index]) * 100:.2f}%"

        if probability is not None

        else "Not available"

    )

    prediction_rows.append({

        "Model": model_name,

        "Prediction": LABEL_NAMES[prediction],

        "Abnormal Probability": probability_text,

        "Correct": (
            "Yes"
            if prediction == true_label
            else "No"
        )

    })

prediction_df = pd.DataFrame(
    prediction_rows
)

st.dataframe(
    prediction_df,
    width="stretch",
    hide_index=True
)


# =====================================================
# 17. FINAL DISCLAIMER
# =====================================================

st.divider()

st.info(
    """
    **MJR-DL1 - Medical Signal Event Detection**

    This is an academic deep-learning research project.

    The model classifies ECG heartbeat segments into
    Normal and Abnormal categories based on its training labels.

    It does not identify exact diseases or provide medical diagnosis.

    Model predictions should not be used for medical decisions.
    """
)

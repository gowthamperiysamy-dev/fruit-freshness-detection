
import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image

st.set_page_config(
    page_title="Fruit Freshness Detection",
    page_icon="🍎",
    layout="wide"
)

MODEL_PATH = "fruit_freshness_model.keras"
METRICS_PATH = "metrics.json"
MATRIX_PATH = "confusion_matrix.png"
GRAPH_PATH = "accuracy_graph.png"

st.title("Fruit Freshness Detection System")
st.write(
    "An AI-powered system that identifies whether a fruit is "
    "Fresh or Rotten using MobileNetV2."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

model = load_model()

st.sidebar.header("Fruit Image Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Fruit")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detection Result")

            resized_image = image.resize((160, 160))

            image_array = np.asarray(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(image_array, axis=0)

            probability = float(
                model.predict(image_array, verbose=0)[0][0]
            )

            fresh_probability = 1 - probability
            rotten_probability = probability

            if probability >= 0.5:
                st.error("ROTTEN FRUIT DETECTED")
                confidence = rotten_probability
                predicted_class = "Rotten"
            else:
                st.success("FRESH FRUIT DETECTED")
                confidence = fresh_probability
                predicted_class = "Fresh"

            st.metric(
                "Predicted Class",
                predicted_class
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.write("### Class Probabilities")

            st.write(
                f"Fresh: {fresh_probability * 100:.2f}%"
            )

            st.progress(fresh_probability)

            st.write(
                f"Rotten: {rotten_probability * 100:.2f}%"
            )

            st.progress(rotten_probability)

            st.info(
                "This prediction is based on visual appearance "
                "and does not guarantee that the fruit is safe to eat."
            )

    except Exception as error:
        st.error(f"Image processing error: {error}")

st.divider()

st.header("Model Performance Evaluation")

if os.path.exists(METRICS_PATH):

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%"
    )

    col4.metric(
        "F1-Score",
        f"{metrics['f1_score'] * 100:.2f}%"
    )

if os.path.exists(MATRIX_PATH):
    st.subheader("Confusion Matrix")
    st.image(
        MATRIX_PATH,
        use_container_width=True
    )

if os.path.exists(GRAPH_PATH):
    st.subheader("Training Accuracy Graph")
    st.image(
        GRAPH_PATH,
        use_container_width=True
    )

st.divider()

st.caption(
    "Fruit Freshness Detection | MobileNetV2 | "
    "TensorFlow | Streamlit"
)

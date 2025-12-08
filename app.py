# app.py — Professional ML Demo for Car Damage Severity
import os
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# -------------------- Page Settings --------------------
st.set_page_config(
    page_title="Car Damage Severity Classifier",
    page_icon="🚗",
    layout="wide",
)

CLASS_NAMES = ["01-minor", "02-moderate", "03-severe"]
IMG_SIZE = (256, 256)


# -------------------- Load Model --------------------
def load_model_any(path: str):
    p = Path(path)

    # If SavedModel directory
    if p.is_dir():
        return tf.saved_model.load(str(p))

    # If .keras or .h5 Keras file
    if p.suffix in [".keras", ".h5"]:
        return tf.keras.models.load_model(str(p), compile=False)

    raise ValueError(f"Unsupported model format: {path}")


# -------------------- Prediction for Both Model Types --------------------
def predict_any(model, x_batch_float32):
    """Handles prediction for both TF SavedModel + Keras .keras"""

    # Case A: Keras model
    if hasattr(model, "predict"):
        probs = model.predict(x_batch_float32, verbose=0)[0]
        return probs

    # Case B: TF SavedModel (signature-based output)
    fn = None

    if hasattr(model, "signatures"):
        fn = model.signatures.get("serving_default")

        # fallback: pick ANY signature if default missing
        if fn is None and len(model.signatures) > 0:
            fn = list(model.signatures.values())[0]

    if fn is None:
        raise RuntimeError("SavedModel has no callable signature.")

    out = fn(tf.constant(x_batch_float32))
    probs = next(iter(out.values())).numpy()[0]
    return probs


# ======================================================
#                 USER INTERFACE (UI)
# ======================================================

st.title("🚗 Car Damage Severity Classifier")
st.caption("Using VGG16 Transfer Learning (TensorFlow/Keras)")

# -------------------- Sidebar --------------------
st.sidebar.header("Model Selection")

default_model_path = "models/vgg16_finetuned_savedmodel"

model_option = st.sidebar.selectbox(
    "Pick model",
    ["SavedModel (directory)"],
)

# Show model path but locked (only 1 default)
model_path = default_model_path
st.sidebar.info(f"Using default model:\n`{default_model_path}`")

# Optional custom advanced section
with st.sidebar.expander("Advanced: custom model path"):
    custom_path = st.text_input("Custom path")
    if custom_path.strip():
        model_path = custom_path.strip()


# -------------------- Load Model With Cache --------------------
@st.cache_resource
def _load_cached(path: str):
    return load_model_any(path)


# Ensure path exists
if not (Path(model_path).is_file() or Path(model_path).is_dir()):
    st.error(f"❌ Model not found at: `{model_path}`")
    st.stop()

with st.spinner("Loading model..."):
    model = _load_cached(model_path)

st.success("✅ Model loaded successfully!")


# ======================================================
#                    IMAGE UPLOAD
# ======================================================

st.subheader("1) Upload an image")

file = st.file_uploader("JPEG/PNG only", type=["jpg", "jpeg", "png"])

if not file:
    st.stop()

image = Image.open(file).convert("RGB")

# Two-column layout: image on left, prediction on right
col_img, col_pred = st.columns([1, 1.2])

with col_img:
    st.image(image, caption="Uploaded image", use_container_width=True)

# Preprocess image
img_resized = image.resize(IMG_SIZE)
x = np.array(img_resized, dtype=np.float32) / 255.0
x = np.expand_dims(x, 0)

# ======================================================
#                   PREDICTION SECTION
# ======================================================

probs = predict_any(model, x)
probs = tf.nn.softmax(probs).numpy()
pred_idx = int(np.argmax(probs))
pred_label = CLASS_NAMES[pred_idx]
confidence = float(probs[pred_idx])

# -------------------- Display Prediction --------------------
with col_pred:
    st.subheader("2) Prediction")

    st.markdown(
        f"""
        <div style='padding:15px;border-radius:10px;background-color:#1f2937'>
            <h3 style='color:#10b981;margin:0'>Predicted class: {pred_label}</h3>
            <p style='color:#9ca3af;margin:0'>Confidence: {confidence*100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Bar Chart Visualization
    st.bar_chart(
        {
            "01-minor": probs[0],
            "02-moderate": probs[1],
            "03-severe": probs[2],
        }
    )

# End

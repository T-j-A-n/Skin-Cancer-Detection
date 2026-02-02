import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Skin Cancer AI", layout="wide")

MODEL_PATH = "my_model.keras"
IMG_SIZE = 224

CLASS_NAMES = [
    "Melanoma",
    "Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion",
    "Squamous Cell Carcinoma",
    "Unknown"
]

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

model = load_model()

# ----------------------------
# PREPROCESS
# ----------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image) / 255.0
    return np.expand_dims(img, axis=0)

# ----------------------------
# GRAD-CAM
# ----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()

# ----------------------------
# LIME
# ----------------------------
def lime_predict(images):
    images = images.astype("float32") / 255.0
    return model.predict(images, verbose=0)

def generate_lime(image_np):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        lime_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    return mark_boundaries(temp / 255.0, mask)

# ----------------------------
# UI
# ----------------------------
st.title("🧬 Skin Cancer Classification with Explainability")

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img_array = preprocess_image(image)

    preds = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(preds))
    confidence = float(preds[0][class_idx])

    st.markdown(
        f"### 🩺 Prediction: **{CLASS_NAMES[class_idx]}** ({confidence:.2%})"
    )

    # 🔧 CHANGE THIS IF NEEDED
    LAST_CONV_LAYER = "conv5_block32_concat"

    gradcam = make_gradcam_heatmap(
        img_array, model, LAST_CONV_LAYER
    )

    lime_img = generate_lime(
        np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Grad-CAM")
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(gradcam, cmap="jet", alpha=0.5)
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.subheader("LIME")
        fig, ax = plt.subplots()
        ax.imshow(lime_img)
        ax.axis("off")
        st.pyplot(fig)

    # ----------------------------
    # BOTTOM TEXT
    # ----------------------------
    st.markdown(
        "<h2 style='color:red; text-align:center;'>Disease</h2>",
        unsafe_allow_html=True
    )

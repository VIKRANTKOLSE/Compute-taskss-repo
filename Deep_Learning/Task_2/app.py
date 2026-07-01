import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
import os, gdown

MODEL_PATH = "resnet_model.keras"
GDRIVE_FILE_ID = "1Vuz75_D-GXkutfcEEOJEwHrVNx9SIwVI"

CLASS_NAMES = [
    'burger', 'butter_naan', 'chai', 'chapati', 'chole_bhature',
    'dal_makhani', 'dhokla', 'fried_rice', 'idli', 'jalebi',
    'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos',
    'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa'
]

st.set_page_config(
    page_title="Food Classifier",
    page_icon="🍽️",
    layout="centered",
)

# Only minimal CSS — no font overrides, everything inherits Streamlit defaults
st.markdown("""
<style>
/* Result box — font-family: inherit forces Streamlit's IBM Plex Sans */
.result-box {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    font-family: inherit;
}
.food-name {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 0.2rem;
    font-family: inherit;
}
.conf-text {
    font-size: 0.9rem;
    color: #666;
    font-family: inherit;
}
.conf-value {
    font-weight: 600;
    color: #c0541a;
}
.food-tag {
    display: inline-block;
    background: #f0f0f0;
    border-radius: 4px;
    font-size: 0.78rem;
    padding: 2px 7px;
    margin: 2px;
    color: #444;
    font-family: inherit;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights... (~600 MB, first run only)"):
            gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False, fuzzy=True)

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])
    base = tf.keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=(224, 224, 3)
    )
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(224, 224, 3)),
        aug,
        tf.keras.layers.Rescaling(1./255),
        base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(265, activation='relu'),
        tf.keras.layers.Dense(20, activation='softmax'),
    ])
    model.load_weights(MODEL_PATH)
    return model


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("About")
    st.write("ResNet50V2 trained on 20 Indian & popular food classes. Upload a photo — the model tells you what's on the plate.")

    st.divider()

    st.subheader("Supported dishes")
    tags_html = "".join(
        f'<span class="food-tag">{c.replace("_", " ")}</span>' for c in CLASS_NAMES
    )
    st.markdown(tags_html, unsafe_allow_html=True)

    st.divider()
    st.caption("ResNet50V2 · Input: 224×224 · 20 classes")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Food Classifier")
st.write("Drop a photo, get a prediction.")
st.divider()


# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Could not load model. Make sure `resnet_model.keras` is in this directory.\n\n{e}")
        st.stop()


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)
    st.caption(f"{uploaded.name}  ·  {image.size[0]}×{image.size[1]} px")

    # ── Predict ───────────────────────────────────────────────────────────────
    with st.spinner("Running inference..."):
        img_arr = tf.keras.preprocessing.image.img_to_array(image.resize((224, 224)))
        img_arr = np.expand_dims(img_arr, axis=0)
        preds = model.predict(img_arr, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    top_name = CLASS_NAMES[top_idx].replace("_", " ").title()
    top_conf = float(preds[top_idx]) * 100

    # ── Result ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-box">
        <div class="food-name">{top_name}</div>
        <div class="conf-text">Confidence: <span class="conf-value">{top_conf:.1f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Top 5 chart ───────────────────────────────────────────────────────────
    st.subheader("Top 5 predictions")

    top5 = np.argsort(preds)[::-1][:5]
    names5 = [CLASS_NAMES[i].replace("_", " ").title() for i in top5]
    confs5 = [float(preds[i]) * 100 for i in top5]
    bar_colors = ['#c0541a' if i == 0 else '#d4c4b0' for i in range(5)]

    fig = go.Figure(go.Bar(
        x=confs5[::-1],
        y=names5[::-1],
        orientation='h',
        marker_color=bar_colors[::-1],
        text=[f"{c:.1f}%" for c in confs5[::-1]],
        textposition='outside',
        cliponaxis=False,
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            ticksuffix='%',
            range=[0, max(confs5) * 1.3],
            zeroline=False,
        ),
        margin=dict(l=0, r=55, t=5, b=5),
        height=220,
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── All probabilities ─────────────────────────────────────────────────────
    with st.expander("All class probabilities"):
        all_idx = np.argsort(preds)[::-1]
        df = pd.DataFrame([
            {"#": i+1, "Food": CLASS_NAMES[idx].replace("_", " ").title(),
             "Confidence": f"{preds[idx]*100:.2f}%"}
            for i, idx in enumerate(all_idx)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("No image uploaded yet.")
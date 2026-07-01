import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd


# ── Model download (runs on cold start in the cloud) ──────────────────────────
import os, gdown

MODEL_PATH = "resnet_model.keras"
GDRIVE_FILE_ID = "1Vuz75_D-GXkutfcEEOJEwHrVNx9SIwVI"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights... (first run only, ~600 MB)"):
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:wght@600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #faf8f5;
    color: #1a1a1a;
}

/* top bar */
.top-bar {
    border-bottom: 1px solid #e8e2d9;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.app-title {
    font-family: 'Lora', serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0;
    line-height: 1.2;
}
.app-tagline {
    font-size: 0.88rem;
    color: #7a7066;
    margin-top: 0.25rem;
}

/* upload zone */
.upload-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #7a7066;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

/* result box */
.result-box {
    background: #fff;
    border: 1px solid #e8e2d9;
    border-radius: 10px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.5rem;
}
.food-name {
    font-family: 'Lora', serif;
    font-size: 1.9rem;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0 0 0.3rem;
}
.conf-text {
    font-size: 0.9rem;
    color: #7a7066;
}
.conf-value {
    font-weight: 600;
    color: #c0541a;
}

/* section heading */
.section-heading {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9e9189;
    margin: 2rem 0 0.8rem;
    border-top: 1px solid #e8e2d9;
    padding-top: 1.2rem;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background-color: #f2ede6;
    border-right: 1px solid #e8e2d9;
}
.sidebar-header {
    font-family: 'Lora', serif;
    font-size: 1rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 0.6rem;
}
.sidebar-body {
    font-size: 0.84rem;
    color: #5c554d;
    line-height: 1.8;
}
.food-tag {
    display: inline-block;
    background: #efe9df;
    border: 1px solid #ddd5c8;
    border-radius: 4px;
    font-size: 0.76rem;
    padding: 2px 8px;
    margin: 2px 2px;
    color: #4a4440;
}

#MainMenu, footer, header { visibility: hidden; }
.stFileUploader label { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])
    base = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(224, 224, 3))
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(224, 224, 3)),
        aug,
        tf.keras.layers.Rescaling(1./255),
        base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(265, activation='relu'),
        tf.keras.layers.Dense(20, activation='softmax'),
    ])
    model.load_weights('resnet_model.keras')
    return model


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-body">
    ResNet50V2 trained on 20 Indian & popular food classes.
    Upload a photo — the model tells you what's on the plate.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">Supported dishes</div>', unsafe_allow_html=True)
    tags_html = "".join(
        f'<span class="food-tag">{c.replace("_", " ")}</span>' for c in CLASS_NAMES
    )
    st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-body" style="color:#9e9189; font-size:0.78rem;">
    Model: ResNet50V2 &nbsp;·&nbsp; Input: 224×224 &nbsp;·&nbsp; Classes: 20
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="app-title">Food Classifier</div>
    <div class="app-tagline">Drop a photo, get a prediction.</div>
</div>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Could not load model weights. Make sure `resnet_model.keras` is in this directory.\n\n{e}")
        st.stop()


# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-label">Upload image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("upload", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

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

    # ── Top 5 chart ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Top 5 predictions</div>', unsafe_allow_html=True)

    top5 = np.argsort(preds)[::-1][:5]
    names5 = [CLASS_NAMES[i].replace("_", " ").title() for i in top5]
    confs5 = [preds[i] * 100 for i in top5]

    bar_colors = ['#c0541a' if i == 0 else '#d4c4b0' for i in range(5)]

    fig = go.Figure(go.Bar(
        x=confs5[::-1],
        y=names5[::-1],
        orientation='h',
        marker_color=bar_colors[::-1],
        text=[f"{c:.1f}%" for c in confs5[::-1]],
        textposition='outside',
        textfont=dict(size=12, color='#1a1a1a', family='Inter'),
        cliponaxis=False,
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#1a1a1a'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e8e2d9',
            ticksuffix='%',
            range=[0, max(confs5) * 1.3],
            tickfont=dict(size=11, color='#9e9189'),
            zeroline=False,
        ),
        yaxis=dict(tickfont=dict(size=12, color='#1a1a1a')),
        margin=dict(l=0, r=55, t=5, b=5),
        height=220,
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── All probabilities ─────────────────────────────────────────────────────
    with st.expander("All class probabilities"):
        all_idx = np.argsort(preds)[::-1]
        df = pd.DataFrame([
            {"#": i+1, "Food": CLASS_NAMES[idx].replace("_", " ").title(), "Confidence": f"{preds[idx]*100:.2f}%"}
            for i, idx in enumerate(all_idx)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div style="margin-top:2rem; padding:3rem 2rem; border:1px dashed #d4c4b0;
                border-radius:10px; text-align:center; color:#9e9189; font-size:0.9rem;">
        No image selected yet.
    </div>
    """, unsafe_allow_html=True)
from __future__ import annotations

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="EdgeVision QC", page_icon="▣", layout="wide")
st.markdown("<style>.stApp{background:#070b10;color:#eef3f8}.block-container{max-width:1300px;padding-top:2rem}.hero h1{font-size:3rem;letter-spacing:-.05em}.hero p{color:#91a0ae}</style>", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>EdgeVision QC</h1><p>Industrial visual inspection reference system.</p></div>', unsafe_allow_html=True)
api_url = st.sidebar.text_input("Inference API", "http://127.0.0.1:8000/predict/")
uploaded = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    left, right = st.columns([1.1, 0.9])
    with left:
        st.image(image, use_container_width=True)
    with right:
        if st.button("Run inspection", type="primary"):
            try:
                response = requests.post(api_url, files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}, timeout=30)
                response.raise_for_status()
                result = response.json()
                prediction = result["prediction"]
                if prediction.startswith("Normal"):
                    st.success(prediction)
                else:
                    st.error(prediction)
                st.metric("Confidence", f"{float(result['confidence']):.2f}%")
            except requests.RequestException as exc:
                st.error(f"API connection failed: {exc}")
            except (KeyError, ValueError) as exc:
                st.error(f"Unexpected API response: {exc}")
else:
    st.info("Upload an image to begin inspection.")
st.caption("Reference implementation. Validate against representative production imagery before operational use.")

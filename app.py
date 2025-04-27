import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import os
import tempfile
from io import BytesIO
import json

# Page config and custom style
st.set_page_config(page_title="Medical Image Analyzer", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background: #f4f4f9;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2e7d32 !important;
        }
        p, li, div {
            color: #222 !important;
        }
        .uploadedImage {
            border: 2px solid #4CAF50;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #45a049 !important;
        }
        /* Sidebar fix */
        section[data-testid="stSidebar"] {
            background-color: #f0f0f0 !important;
            color: #222 !important;
        }
        /* Fix for JSON text visibility */
        pre {
            background-color: #eaeaea !important;
            color: #111 !important;
        }
        /* Fix download button */
        .stDownloadButton button {
            background-color: #4CAF50 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.title("🩺 App Settings")
    st.markdown("Upload medical images and detect anomalies.")
    st.markdown("---")
    st.markdown("**Model**: YOLOv8")
    st.markdown("**Classes**: Custom Trained")
    st.markdown("**Version**: 1.0.0")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model_path = "best.pt"
    return YOLO(model_path)

model = load_model()

# Title and Intro
st.markdown("<h1>🧠 Medical Image Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a medical image to detect anomalies using YOLOv8.</p>", unsafe_allow_html=True)

# Layout into two columns
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing..."):
                gif_placeholder = st.empty()
                gif_placeholder.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif", width=120)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    image.save(temp_file.name)
                    results = model(temp_file.name)

                gif_placeholder.empty()  # 🧹 This clears the GIF once done


                result_img = results[0].plot()
                result_pil = Image.fromarray(result_img)

                # Store these in session to show in col2
                st.session_state['results'] = results
                st.session_state['result_img'] = result_img
                st.session_state['result_pil'] = result_pil
                st.session_state['temp_file_path'] = temp_file.name

with col2:
    if 'results' in st.session_state:
        results = st.session_state['results']
        result_img = st.session_state['result_img']
        result_pil = st.session_state['result_pil']
        temp_file_path = st.session_state['temp_file_path']

        tab1, tab2 = st.tabs(["🔬 Detection Result", "📊 Model Info"])

        with tab1:
            st.image(result_img, caption="✅ Analysis Complete", use_column_width=True)
            st.success("Detection Complete!")
            st.snow()

            # Detected classes
            st.subheader("📋 Detected Classes:")
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    st.write(f"- **{model.names[cls_id]}** with confidence **{conf:.2f}**")

            # Summary
            st.markdown("### 📊 Detection Summary")
            cols = st.columns(3)
            total_detections = len(results[0].boxes)
            top_class = model.names[int(results[0].boxes.cls[0])] if total_detections > 0 else "N/A"
            top_conf = float(results[0].boxes.conf[0]) if total_detections > 0 else 0

            cols[0].metric("Total Detections", total_detections)
            cols[1].metric("Top Class", top_class)
            cols[2].metric("Top Confidence", f"{top_conf*100:.2f}%")

            # Download result image
            buf = BytesIO()
            result_pil.save(buf, format="JPEG")
            st.download_button("📥 Download Result", data=buf.getvalue(), file_name="result.jpg", mime="image/jpeg")

        
        with tab2:
            st.markdown("### 🧬 Model Details")
            model_info = {
                "Model Path": "C:/Users/jayve/runs/detect/train30/weights/best.pt",
                "Framework": "YOLOv8",
                "Class Labels": model.names
            }
            st.code(json.dumps(model_info, indent=4), language="json")

        os.remove(temp_file_path)

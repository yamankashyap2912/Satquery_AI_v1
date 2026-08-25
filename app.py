import os
import tempfile
import torch
import numpy as np
import rasterio
import streamlit as st
from configilm.ConfigILM import ILMConfiguration, ConfigILM, ILMType
from configilm.util import get_default_tokenizer
from configilm.extra.BENv2_utils import band_combi_to_mean_std

# Page config
st.set_page_config(
    page_title="SatQuery AI - ISRO SIH 26167", 
    page_icon="🛰️", 
    layout="wide"
)

st.title("🛰️ SatQuery AI: Interactive Vision-Language Assistant")
st.markdown("### Multimodal Remote Sensing Image Analysis through Text Queries (ISRO / SIH)")

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Band Definitions
S2_BANDS_V020 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]

@st.cache_resource
def load_model():
    """Caches the model so it only loads once into memory."""
    vqa_config = ILMConfiguration(
        timm_model_name="resnet50",
        hf_model_name="prajjwal1/bert-tiny",
        classes=25,
        channels=12,
        image_size=120,
        network_type=ILMType.VQA_CLASSIFICATION,
        load_pretrained_timm_if_available=False,
        load_pretrained_hf_if_available=False
    )
    model = ConfigILM(vqa_config)
    # Load your locally saved fine-tuned weights
    model.load_state_dict(torch.load("satquery_vqa_finetuned.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

with st.spinner("Loading SatQuery AI model weights..."):
    model = load_model()
    tokenizer = get_default_tokenizer()

st.success("Model loaded successfully!")

# Sidebar File Uploader
st.sidebar.header("📁 Upload Satellite Patch")
st.sidebar.markdown("Upload your core GeoTIFF band files for analysis.")

uploaded_s1 = st.sidebar.file_uploader("Upload Sentinel-1 SAR Bands (VV, VH)", type=["tif"], accept_multiple_files=True)
uploaded_s2 = st.sidebar.file_uploader("Upload Sentinel-2 Optical Bands (B02-B12)", type=["tif"], accept_multiple_files=True)

# Main Query Interface
st.markdown("---")
user_query = st.text_input("💬 Ask a question about the satellite imagery:", "Is there arable land in this image?")

if st.button("Run Analysis", type="primary"):
    if not uploaded_s1 or not uploaded_s2:
        st.warning("⚠️ Please upload both Sentinel-1 and Sentinel-2 GeoTIFF files using the sidebar to proceed.")
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            try:
                # Save uploaded files locally to temporary directory
                s1_map = {}
                for file in uploaded_s1:
                    path = os.path.join(tmpdirname, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    # Map filename to band identifier
                    for b in S1_BANDS:
                        if b in file.name.upper():
                            s1_map[b] = path

                s2_map = {}
                for file in uploaded_s2:
                    path = os.path.join(tmpdirname, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    for b in S2_BANDS_V020:
                        if b in file.name.upper():
                            s2_map[b] = path

                # Ensure all bands are provided
                if len(s1_map) < 2 or len(s2_map) < 10:
                    st.error("❌ Missing specific band files! Make sure you upload all required VV, VH, and B02–B12 bands.")
                else:
                    with st.spinner("Processing optical & SAR multisensor fusion..."):
                        # Read bands via rasterio and normalize
                        fusion_mean, fusion_std = band_combi_to_mean_std(12)
                        fusion_mean, fusion_std = np.array(fusion_mean), np.array(fusion_std)

                        s1_arrays = [rasterio.open(s1_map[b]).read(1).astype(np.float32) for b in S1_BANDS]
                        s2_arrays = [rasterio.open(s2_map[b]).read(1).astype(np.float32) for b in S2_BANDS_V020]

                        s1 = np.stack(s1_arrays, axis=0)
                        s2 = np.stack(s2_arrays, axis=0)
                        stacked = np.concatenate([s1, s2], axis=0)
                        stacked = (stacked - fusion_mean[:, None, None]) / fusion_std[:, None, None]
                        
                        img_tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)
                        q_ids = torch.tensor([tokenizer.encode(user_query, max_length=32, padding="max_length", truncation=True)]).to(device)

                        # Inference
                        with torch.no_grad():
                            logits = model((img_tensor, q_ids))
                            pred_idx = torch.argmax(logits, dim=-1).item()

                    # Display Results layout
                    st.markdown("### 📊 Execution Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"**Query:** {user_query}")
                        st.success(f"**Predicted VQA Answer ID:** {pred_idx}")
                        
                    with col2:
                        st.json({
                            "selected_task": "visual_question_answering",
                            "model_backend": "ConfigILM-ResNet50-BERT",
                            "input_modality": "Optical + SAR Fusion (12 channels)",
                            "confidence_logits_max": float(torch.max(logits).item())
                        })

            except Exception as e:
                st.error(f"An error occurred during inference: {e}")
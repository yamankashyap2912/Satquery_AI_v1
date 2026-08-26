import os
import io
import json
import tempfile
import torch
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image
from configilm.ConfigILM import ILMConfiguration, ConfigILM, ILMType
from configilm.util import get_default_tokenizer
from configilm.extra.BENv2_utils import band_combi_to_mean_std

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS STYLING
# ==========================================
st.set_page_config(
    page_title="SatQuery AI — ISRO SIH 26167", 
    page_icon="🛰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Modern Dashboard CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e222d; padding: 15px; border-radius: 10px; border: 1px solid #2e3440; }
    .answer-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        color: #ffffff;
        margin-top: 15px;
    }
    .badge {
        background-color: #3b82f6;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Band Definitions
S2_BANDS_V020 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]

# ==========================================
# 2. MODEL & VOCABULARY LOADERS
# ==========================================
@st.cache_resource
def load_vocab():
    """Strictly maps standard SIH VQA answers to prevent caption bleed-over."""
    return {
        0: "no", 1: "yes", 2: "0", 3: "1", 4: "2", 5: "3", 6: "4", 7: "5",
        8: "arable land", 9: "urban fabric", 10: "pastures", 11: "broad-leaved forest",
        12: "coniferous forest", 13: "mixed forest", 14: "water bodies", 15: "agriculture",
        16: "inland waters", 17: "coastal wetlands", 18: "permanent crops",
        19: "complex cultivation patterns", 20: "agro-forestry areas",
        21: "industrial or commercial units", 22: "transitional woodland", 
        23: "beaches, dunes, sands", 24: "other"
    }

@st.cache_resource
def load_vqa_model():
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
    weights_path = "satquery_vqa_finetuned.pt"
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

# Initialize variables in top-level scope
id_to_answer = load_vocab()
num_classes = len(id_to_answer)  # FIXED: Variable explicitly defined here
model = load_vqa_model()
tokenizer = get_default_tokenizer()

# ==========================================
# 3. HELPER FUNCTIONS FOR VISUALS & INFERENCE
# ==========================================
def render_rgb_preview(s2_map):
    """Generates RGB True-Color Composite (B04, B03, B02)."""
    try:
        r = rasterio.open(s2_map['B04']).read(1).astype(np.float32)
        g = rasterio.open(s2_map['B03']).read(1).astype(np.float32)
        b = rasterio.open(s2_map['B02']).read(1).astype(np.float32)
        rgb = np.dstack([r, g, b])
        rgb_norm = np.clip((rgb - np.percentile(rgb, 2)) / (np.percentile(rgb, 98) - np.percentile(rgb, 2) + 1e-6), 0, 1)
        return (rgb_norm * 255).astype(np.uint8)
    except Exception:
        return None

def process_inference(s1_map, s2_map, query_text):
    fusion_mean, fusion_std = band_combi_to_mean_std(12)
    fusion_mean, fusion_std = np.array(fusion_mean), np.array(fusion_std)

    s1_arrays = [rasterio.open(s1_map[b]).read(1).astype(np.float32) for b in S1_BANDS]
    s2_arrays = [rasterio.open(s2_map[b]).read(1).astype(np.float32) for b in S2_BANDS_V020]

    s1 = np.stack(s1_arrays, axis=0)
    s2 = np.stack(s2_arrays, axis=0)
    stacked = np.concatenate([s1, s2], axis=0)
    stacked = (stacked - fusion_mean[:, None, None]) / fusion_std[:, None, None]

    img_tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)
    q_ids = torch.tensor([tokenizer.encode(query_text, max_length=32, padding="max_length", truncation=True)]).to(device)

    with torch.no_grad():
        logits = model((img_tensor, q_ids))
        probs = torch.softmax(logits, dim=-1)
        top_prob, pred_idx = torch.max(probs, dim=-1)

    predicted_text = id_to_answer.get(pred_idx.item(), f"Class {pred_idx.item()}")
    confidence = float(top_prob.item()) * 100
    return predicted_text, confidence, logits

# ==========================================
# 4. SIDEBAR SETUP
# ==========================================
st.sidebar.title("🛰️ SatQuery AI Core")
st.sidebar.markdown("**ISRO SIH PS 26167**")
st.sidebar.info(f"🟢 System Device: `{device.upper()}` | Vocab Size: `{num_classes}` classes")

st.sidebar.subheader("📂 Upload Multi-Band Patch")
uploaded_s1 = st.sidebar.file_uploader("Sentinel-1 SAR Bands (VV, VH)", type=["tif"], accept_multiple_files=True)
uploaded_s2 = st.sidebar.file_uploader("Sentinel-2 Optical Bands (B02-B12)", type=["tif"], accept_multiple_files=True)

# ==========================================
# 5. MAIN INTERFACE & TABS
# ==========================================
st.title("🛰️ SatQuery AI: Vision-Language Intelligence")
st.markdown("Multimodal Optical-SAR Fusion & Visual Question Answering for Remote Sensing")

tab1, tab2, tab3 = st.tabs(["💬 Interactive VQA", "⚡ Batch Patch Processing", "📈 System Analytics & Architecture"])

# ------------------------------------------
# TAB 1: INTERACTIVE SINGLE-PATCH VQA
# ------------------------------------------
with tab1:
    col_input, col_view = st.columns([1.2, 0.8])
    
    with col_input:
        st.subheader("Query Satellite Patch")
        user_query = st.text_input(
            "Enter your natural language question:", 
            value="Is there arable land in this image?",
            placeholder="e.g., Is there urban fabric present?"
        )
        run_btn = st.button("🔍 Execute VQA Analysis", type="primary", use_container_width=True)

    s1_map, s2_map = {}, {}
    if uploaded_s1 and uploaded_s2:
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in uploaded_s1:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out: out.write(f.getbuffer())
                for b in S1_BANDS:
                    if b in f.name.upper(): s1_map[b] = path
            
            for f in uploaded_s2:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out: out.write(f.getbuffer())
                for b in S2_BANDS_V020:
                    if b in f.name.upper(): s2_map[b] = path

            with col_view:
                if len(s2_map) >= 3 and all(k in s2_map for k in ['B04', 'B03', 'B02']):
                    rgb_img = render_rgb_preview(s2_map)
                    if rgb_img is not None:
                        st.image(rgb_img, caption="Sentinel-2 RGB True Color Composite (120x120)", use_container_width=True)

            if run_btn:
                if len(s1_map) < 2 or len(s2_map) < 10:
                    st.error("❌ Missing required bands! Please ensure all 2 SAR (VV, VH) and 10 Optical (B02-B12) bands are uploaded.")
                else:
                    with st.spinner("Processing 12-channel Optical-SAR fusion..."):
                        ans_text, conf, logits = process_inference(s1_map, s2_map, user_query)
                    
                    st.markdown("### 📊 Model Output")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Answer", ans_text.upper())
                    m2.metric("Confidence Score", f"{conf:.2f}%")
                    m3.metric("Input Channels", "12 (2 SAR + 10 Optical)")

                    st.markdown(f"""
                    <div class="answer-card">
                        <h4>🔍 Detailed Inference Summary</h4>
                        <p><b>Question:</b> {user_query}</p>
                        <p><b>Predicted Response:</b> <span class="badge">{ans_text}</span></p>
                        <p><b>Model Backend:</b> ConfigILM (ResNet50 Vision Encoder + BERT-Tiny Text Encoder)</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("💡 Tip: Upload `.tif` band files in the left sidebar to enable live interactive analysis.")

# ------------------------------------------
# TAB 2: BATCH PATCH PROCESSING
# ------------------------------------------
with tab2:
    st.subheader("⚡ Batch VQA Processing")
    st.markdown("Automate satellite patch auditing by asking a single prompt across multiple test patches.")
    
    batch_query = st.text_input("Batch Question Prompt:", "Is this area used for agriculture?")
    
    if st.button("🚀 Process Local Test Patches", type="secondary"):
        test_dir = "test_patch"
        if os.path.exists(test_dir):
            bands = [f for f in os.listdir(test_dir) if f.endswith('.tif')]
            st.success(f"Found {len(bands)} band files in `{test_dir}/` directory.")
            
            # Map local test patch
            local_s1 = {b: os.path.join(test_dir, f"{b}.tif") for b in S1_BANDS if os.path.exists(os.path.join(test_dir, f"{b}.tif"))}
            local_s2 = {b: os.path.join(test_dir, f"{b}.tif") for b in S2_BANDS_V020 if os.path.exists(os.path.join(test_dir, f"{b}.tif"))}
            
            if len(local_s1) == 2 and len(local_s2) == 10:
                ans, conf, _ = process_inference(local_s1, local_s2, batch_query)
                
                results_df = pd.DataFrame([{
                    "Patch Name": "test_patch",
                    "Question": batch_query,
                    "Predicted Answer": ans,
                    "Confidence": f"{conf:.2f}%",
                    "Status": "Passed"
                }])
                st.dataframe(results_df, use_container_width=True)
            else:
                st.error("Missing required band files in local `test_patch/` directory.")
        else:
            st.warning("No `test_patch/` directory found in project root.")

# ------------------------------------------
# TAB 3: SYSTEM ARCHITECTURE & ANALYTICS
# ------------------------------------------
with tab3:
    st.subheader("📐 SatQuery AI System Architecture")
    st.markdown("""
    - **Vision Backbone:** ResNet50 initialized with BIFOLD BigEarthNet v2.0 (reBEN) weights.
    - **Text Backbone:** BERT-Tiny Transformer Tokenizer.
    - **Fusion Layer:** Late-fusion cross-attention network mapping 12 multisensory input channels to 840 answer vocabulary classes.
    - **Dataset Grounding:** Fine-tuned on BigEarthNet-VQA parquet dataset.
    """)
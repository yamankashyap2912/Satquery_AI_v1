import os
import json
import tempfile
import torch
import numpy as np
import rasterio
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from configilm.ConfigILM import ILMConfiguration, ConfigILM, ILMType
from configilm.util import get_default_tokenizer
from configilm.extra.BENv2_utils import band_combi_to_mean_std

# ----------------------------------------------------------------------------
# Page Config & Styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SatQuery AI — ISRO SIH 26167", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
    .main > div { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px 8px 0 0; font-weight: 600; }
    .answer-banner { background: linear-gradient(135deg, #0f9d5815, #0f9d5805); border-left: 5px solid #0f9d58; border-radius: 8px; padding: 18px; font-size: 1.2rem; margin-bottom: 20px;}
    .title-banner { display: flex; align-items: center; gap: 15px; margin-bottom: 5px; }
    .isro-badge { background-color: #0b3d91; color: white; padding: 5px 12px; border-radius: 6px; font-size: 0.9rem; font-weight: bold; letter-spacing: 1px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-banner"><span class="isro-badge">ISRO SIH PS 26167</span><h1>🛰️ SatQuery AI</h1></div>', unsafe_allow_html=True)
st.caption("Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis")

device = "cuda" if torch.cuda.is_available() else "cpu"
S2_BANDS_V020 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]

# ----------------------------------------------------------------------------
# Image Processing & Normalization
# ----------------------------------------------------------------------------
def normalize_img(arr):
    arr_min, arr_max = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    if arr_max == arr_min: return np.zeros_like(arr)
    return np.clip((arr - arr_min) / (arr_max - arr_min), 0, 1)

def get_rgb(s2_arrays):
    return normalize_img(np.stack([s2_arrays["B04"], s2_arrays["B03"], s2_arrays["B02"]], axis=-1))

def get_false_color(s2_arrays):
    return normalize_img(np.stack([s2_arrays["B08"], s2_arrays["B04"], s2_arrays["B03"]], axis=-1))

def get_sar_composite(s1_arrays):
    vv, vh = s1_arrays["VV"], s1_arrays["VH"]
    ratio = np.clip(vv / (vh + 1e-8), -10, 10)
    return normalize_img(np.stack([vv, vh, ratio], axis=-1))

def get_ndvi(s2_arrays):
    b8, b4 = s2_arrays["B08"], s2_arrays["B04"]
    return np.clip((b8 - b4) / (b8 + b4 + 1e-8), -1, 1)

# ----------------------------------------------------------------------------
# Model & Vocabulary Loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_answer_vocab():
    vocab_path = "answer_vocab.json"
    if not os.path.exists(vocab_path):
        st.warning("answer_vocab.json not found — predictions will show raw class IDs.")
        return None, None
    answer_to_id = json.load(open(vocab_path))
    id_to_answer = {v: k for k, v in answer_to_id.items()}
    return answer_to_id, id_to_answer

@st.cache_resource
def load_vqa_model(num_classes: int):
    vqa_config = ILMConfiguration(
        timm_model_name="resnet50", hf_model_name="prajjwal1/bert-tiny",
        classes=num_classes, channels=12, image_size=120,
        network_type=ILMType.VQA_CLASSIFICATION,
        load_pretrained_timm_if_available=False, load_pretrained_hf_if_available=False,
    )
    model = ConfigILM(vqa_config)
    model.load_state_dict(torch.load("satquery_vqa_finetuned.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return get_default_tokenizer()

@st.cache_resource
def load_norm_stats():
    fusion_mean, fusion_std = band_combi_to_mean_std(12)
    return np.array(fusion_mean), np.array(fusion_std)

answer_to_id, id_to_answer = load_answer_vocab()
num_classes = len(answer_to_id) if answer_to_id else 25
tokenizer = load_tokenizer()
fusion_mean, fusion_std = load_norm_stats()

with st.spinner("Loading SatQuery AI multimodal weights..."):
    model = load_vqa_model(num_classes)

# ----------------------------------------------------------------------------
# File Reading & Inference Helpers
# ----------------------------------------------------------------------------
def read_bands_from_uploads(files, band_names):
    band_map = {}
    for f in files:
        name_upper = f.name.upper()
        name_no_ext = os.path.splitext(name_upper)[0].strip()
        if name_no_ext in ["B2", "B3", "B4", "B5", "B6", "B7", "B8"]:
            name_no_ext = name_no_ext.replace("B", "B0")
            
        for b in band_names:
            if b == name_no_ext or f"_{b}" in name_no_ext or f"-{b}" in name_no_ext:
                band_map[b] = f
            elif b in name_upper:
                if b == "B08" and "B8A" in name_upper: continue
                if b not in band_map: band_map[b] = f
    return band_map

def load_band_array(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    with rasterio.open(tmp_path) as src:
        arr = src.read(1).astype(np.float32)
    os.unlink(tmp_path)
    return arr

def build_fusion_tensor(s1_arrays: dict, s2_arrays: dict):
    s1 = np.stack([s1_arrays[b] for b in S1_BANDS], axis=0)
    s2 = np.stack([s2_arrays[b] for b in S2_BANDS_V020], axis=0)
    stacked = np.concatenate([s1, s2], axis=0)
    stacked = (stacked - fusion_mean[:, None, None]) / fusion_std[:, None, None]
    return torch.from_numpy(stacked).unsqueeze(0).float().to(device)

def run_vqa(question: str, img_tensor: torch.Tensor):
    q_ids = torch.tensor([tokenizer.encode(question, max_length=32, padding="max_length", truncation=True)]).to(device)
    with torch.no_grad():
        logits = model((img_tensor, q_ids))
    probs = torch.softmax(logits, dim=-1)[0]
    
    top5_probs, top5_idx = torch.topk(probs, min(5, len(probs)))
    top5_results = []
    for i in range(len(top5_probs)):
        idx = int(top5_idx[i].item())
        conf = float(top5_probs[i].item())
        ans = id_to_answer.get(idx, f"class_{idx}") if id_to_answer else f"class_{idx}"
        top5_results.append({"Answer": str(ans).capitalize(), "Confidence": conf})
        
    return top5_results[0]["Answer"], top5_results[0]["Confidence"], top5_results

# ----------------------------------------------------------------------------
# UI Layout
# ----------------------------------------------------------------------------
tab_vqa, tab_batch, tab_change, tab_about = st.tabs([
    "💬 Ask a Question", "📂 Batch Analysis", "🔄 Change Detection", "ℹ️ About"
])

# --- Tab 1: Single-Patch VQA ---
with tab_vqa:
    st.subheader("Multimodal Visual Question Answering")
    st.write("Upload a co-registered Sentinel-1 (SAR) and Sentinel-2 (Optical) patch.")

    col1, col2 = st.columns(2)
    with col1: s1_files = st.file_uploader("Sentinel-1 SAR bands (VV, VH)", type=["tif", "tiff"], accept_multiple_files=True, key="vqa_s1")
    with col2: s2_files = st.file_uploader("Sentinel-2 optical bands (B02–B12)", type=["tif", "tiff"], accept_multiple_files=True, key="vqa_s2")
    
    st.markdown("### 📝 Query the Model")
    
    # Categorized Reference Questions matching the ISRO PS Guide
    SUGGESTED_QUESTIONS = [
        "Is there arable land in this image?",
        "Is there any inland water in this patch?",
        "Is there a forested area visible in this scene?",
        "Is there more arable land than pasture in this image?",
        "Would you say that any arable land lies next to inland water in this image?",
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "Write a custom query..."
    ]
    selected_q = st.selectbox("Select a query aligned with PS reference categories:", SUGGESTED_QUESTIONS)
    user_query = st.text_input("Enter your custom query:") if selected_q == "Write a custom query..." else selected_q

    if st.button("🚀 Run AI Analysis", type="primary", key="vqa_run"):
        if not s1_files or not s2_files:
            st.warning("⚠️ Upload both Sentinel-1 and Sentinel-2 bands first.")
        else:
            s1_map = read_bands_from_uploads(s1_files, S1_BANDS)
            s2_map = read_bands_from_uploads(s2_files, S2_BANDS_V020)
            
            if len(s1_map) < 2 or len(s2_map) < 10:
                st.error(f"❌ Missing bands — Found {len(s1_map)}/2 SAR bands and {len(s2_map)}/10 Optical bands.")
            else:
                with st.spinner("Executing multimodal fusion inference..."):
                    s1_arrays = {b: load_band_array(s1_map[b]) for b in S1_BANDS}
                    s2_arrays = {b: load_band_array(s2_map[b]) for b in S2_BANDS_V020}
                    img_tensor = build_fusion_tensor(s1_arrays, s2_arrays)
                    
                    answer_text, confidence, top5_results = run_vqa(user_query, img_tensor)

                st.markdown(f"""<div class="answer-banner">
                    <b>Q:</b> {user_query}<br><b>A:</b> <span style='color: #0b3d91; font-weight: bold;'>{answer_text}</span>
                    </div>""", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Top Confidence", f"{confidence*100:.1f}%")
                c2.metric("Input Fusion", "12 Channels (VV, VH, Optical)")
                c3.metric("Vocabulary Size", f"{num_classes} Classes")
                
                st.markdown("### 📊 Inference Confidence Distribution")
                df_probs = pd.DataFrame(top5_results).sort_values("Confidence", ascending=True)
                fig = px.bar(df_probs, x="Confidence", y="Answer", orientation='h', 
                             color="Confidence", color_continuous_scale="Teal",
                             title="Top 5 Candidate Classes")
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0), height=280)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🗺️ Earth Observation Visualizations")
                vc1, vc2, vc3, vc4 = st.columns(4)
                vc1.image(get_rgb(s2_arrays), caption="True Color RGB", use_container_width=True)
                vc2.image(get_false_color(s2_arrays), caption="Color Infrared (Vegetation)", use_container_width=True)
                vc3.image(get_sar_composite(s1_arrays), caption="SAR Radar Composite", use_container_width=True)
                
                fig_ndvi, ax = plt.subplots(figsize=(3,3))
                ax.imshow(get_ndvi(s2_arrays), cmap="RdYlGn", vmin=-1, vmax=1)
                ax.axis("off")
                fig_ndvi.subplots_adjust(left=0, right=1, bottom=0, top=1)
                vc4.pyplot(fig_ndvi, use_container_width=True)
                vc4.caption("NDVI Heatmap")

# --- Tab 2: Batch Analysis (Flexible Pathing) ---
with tab_batch:
    st.subheader("Batch Analysis across Multiple Patches")
    st.write("Pass a path containing patch subfolders (e.g. `test_patches/`) or a single patch folder.")

    batch_root = st.text_input("Folder path containing patch subfolders:", "test_patches")
    batch_query = st.text_input("Question to execute across all patches:", "Is there arable land in this image?", key="batch_q")

    if st.button("Run Batch Analysis", key="batch_run"):
        if not os.path.exists(batch_root):
            st.error(f"❌ Path not found: `{batch_root}`")
        else:
            # Flexible path resolver: check if folder contains subfolders or direct .tif files
            subdirs = [os.path.join(batch_root, d) for d in os.listdir(batch_root) if os.path.isdir(os.path.join(batch_root, d))]
            direct_tifs = [f for f in os.listdir(batch_root) if f.endswith(('.tif', '.tiff'))]
            
            patch_folders = subdirs if len(subdirs) > 0 else ([batch_root] if len(direct_tifs) >= 10 else [])
            
            if not patch_folders:
                st.warning("No patch subfolders or complete `.tif` band sets found in the path.")
            else:
                results = []
                progress = st.progress(0.0, text="Processing patches...")
                
                for i, pfolder in enumerate(patch_folders):
                    pname = os.path.basename(pfolder) if pfolder != batch_root else "single_patch"
                    try:
                        files_in_dir = [os.path.join(pfolder, f) for f in os.listdir(pfolder) if f.endswith(('.tif', '.tiff'))]
                        
                        class MockFile:
                            def __init__(self, path): self.name, self.path = os.path.basename(path), path
                            def getbuffer(self): return open(self.path, "rb").read()

                        mock_files = [MockFile(p) for p in files_in_dir]
                        s1_map = read_bands_from_uploads(mock_files, S1_BANDS)
                        s2_map = read_bands_from_uploads(mock_files, S2_BANDS_V020)

                        s1_arrays = {b: load_band_array(s1_map[b]) for b in S1_BANDS}
                        s2_arrays = {b: load_band_array(s2_map[b]) for b in S2_BANDS_V020}
                        img_tensor = build_fusion_tensor(s1_arrays, s2_arrays)

                        ans, conf, _ = run_vqa(batch_query, img_tensor)
                        results.append({"Patch": pname, "Answer": ans, "Confidence": f"{conf*100:.1f}%"})
                    except Exception as e:
                        results.append({"Patch": pname, "Answer": f"Error: {e}", "Confidence": "N/A"})
                    
                    progress.progress((i + 1) / len(patch_folders))

                st.dataframe(pd.DataFrame(results), use_container_width=True)

# --- Tab 3: Bi-Temporal Change Detection ---
with tab_change:
    st.subheader("Bi-Temporal Change Detection")
    st.write("Upload Sentinel-2 optical bands from two dates of the same area to analyze land-use shifts.")

    c1, c2 = st.columns(2)
    with c1: t1_files = st.file_uploader("Time 1 (Earlier Date)", type=["tif", "tiff"], accept_multiple_files=True, key="cd_t1")
    with c2: t2_files = st.file_uploader("Time 2 (Later Date)", type=["tif", "tiff"], accept_multiple_files=True, key="cd_t2")

    if st.button("🔄 Analyze Changes", type="primary", key="change_run"):
        if not t1_files or not t2_files:
            st.warning("⚠️ Upload Sentinel-2 bands for both dates.")
        else:
            t1_map = read_bands_from_uploads(t1_files, S2_BANDS_V020)
            t2_map = read_bands_from_uploads(t2_files, S2_BANDS_V020)
            
            if len(t1_map) < 10 or len(t2_map) < 10:
                st.error("❌ Missing required optical bands for one or both dates.")
            else:
                with st.spinner("Computing bi-temporal spectral divergence..."):
                    s2_t1 = {b: load_band_array(t1_map[b]) for b in S2_BANDS_V020}
                    s2_t2 = {b: load_band_array(t2_map[b]) for b in S2_BANDS_V020}
                    
                    ndvi_diff = get_ndvi(s2_t2) - get_ndvi(s2_t1)

                cc1, cc2, cc3 = st.columns(3)
                cc1.image(get_rgb(s2_t1), caption="Time 1 (RGB)", use_container_width=True)
                cc2.image(get_rgb(s2_t2), caption="Time 2 (RGB)", use_container_width=True)
                
                fig_cd, ax_cd = plt.subplots(figsize=(3,3))
                ax_cd.imshow(ndvi_diff, cmap="coolwarm_r", vmin=-0.5, vmax=0.5)
                ax_cd.axis("off")
                fig_cd.subplots_adjust(left=0, right=1, bottom=0, top=1)
                cc3.pyplot(fig_cd, use_container_width=True)
                cc3.caption("NDVI Divergence (Red = Loss, Blue = Gain)")
                
                loss_pct = np.mean(ndvi_diff < -0.15) * 100
                gain_pct = np.mean(ndvi_diff > 0.15) * 100
                st.info(f"**Vegetation Analytics:** Detected **{loss_pct:.1f}%** vegetation loss and **{gain_pct:.1f}%** vegetation gain.")

# --- Tab 4: About ---
with tab_about:
    st.subheader("About SatQuery AI")
    st.markdown("""
    Built for **ISRO SIH PS 26167** — an agentic multimodal vision-language assistant for Earth observation analysis.
    
    * **Vision-Language Model:** `ConfigILM` (ResNet-50 + BERT-tiny).
    * **Sensor Modality:** Optical-SAR Early Fusion (Sentinel-1 VV/VH + Sentinel-2 10-Band Optical).
    * **Capabilities:** Multimodal VQA, Bi-temporal change detection, and multi-patch batch analytics.
    """)
    st.metric("Vocabulary Size", f"{num_classes} Classes")
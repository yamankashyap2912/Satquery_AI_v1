import os
import json
import tempfile
import torch
import numpy as np
import rasterio
import streamlit as st
from configilm.ConfigILM import ILMConfiguration, ConfigILM, ILMType
from configilm.util import get_default_tokenizer
from configilm.extra.BENv2_utils import band_combi_to_mean_std

# ----------------------------------------------------------------------------
# Page config + light theming
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SatQuery AI — ISRO SIH 26167", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
    .main > div { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px; border-radius: 8px 8px 0 0; font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a4d8f15, #1a4d8f05);
        border: 1px solid #1a4d8f30; border-radius: 12px; padding: 16px; margin-bottom: 8px;
    }
    .answer-banner {
        background: linear-gradient(135deg, #0f9d5810, #0f9d5805);
        border-left: 4px solid #0f9d58; border-radius: 8px; padding: 16px 20px; font-size: 1.1rem;
    }
    div[data-testid="stExpander"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ SatQuery AI")
st.caption("Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis  ·  ISRO SIH PS 26167")

device = "cuda" if torch.cuda.is_available() else "cpu"
S2_BANDS_V020 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]
BEN19_CLASSES = [
    'Agro-forestry areas', 'Arable land', 'Beaches, dunes, sands', 'Broad-leaved forest',
    'Coastal wetlands', 'Complex cultivation patterns', 'Coniferous forest',
    'Industrial or commercial units', 'Inland waters', 'Inland wetlands',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Marine waters', 'Mixed forest', 'Moors, heathland and sclerophyllous vegetation',
    'Natural grassland and sparsely vegetated areas', 'Pastures', 'Permanent crops',
    'Transitional woodland, shrub', 'Urban fabric'
]

# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource
def load_answer_vocab():
    vocab_path = "answer_vocab.json"
    if not os.path.exists(vocab_path):
        st.warning("answer_vocab.json not found — predictions will show raw class IDs instead of answer text. "
                   "Save `answer_to_id` as answer_vocab.json next to app.py after training (see chat for the snippet).")
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
    
    # ✅ FIX: Call these separately since ConfigILM's .to() returns None
    model.to(device)
    model.eval()
    
    return model

@st.cache_resource
def load_tokenizer():
    return get_default_tokenizer()

@st.cache_resource
def load_norm_stats():
    fusion_mean, fusion_std = band_combi_to_mean_std(12)
    s2_mean, s2_std = band_combi_to_mean_std(10)
    return (np.array(fusion_mean), np.array(fusion_std)), (np.array(s2_mean), np.array(s2_std))

answer_to_id, id_to_answer = load_answer_vocab()
num_classes = len(answer_to_id) if answer_to_id else 25
tokenizer = load_tokenizer()
(fusion_mean, fusion_std), (s2_mean, s2_std) = load_norm_stats()

with st.spinner("Loading SatQuery AI model weights..."):
    model = load_vqa_model(num_classes)

# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def read_bands_from_uploads(files, band_names):
    band_map = {}
    for f in files:
        for b in band_names:
            if b.upper() in f.name.upper():
                band_map[b] = f
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
    top_idx = int(torch.argmax(probs).item())
    top_conf = float(probs[top_idx].item())
    answer_text = id_to_answer.get(top_idx, f"class_{top_idx}") if id_to_answer else f"class_{top_idx}"
    return answer_text, top_conf, logits

def render_uploader_pair(key_prefix):
    col1, col2 = st.columns(2)
    with col1:
        s1_files = st.file_uploader("Sentinel-1 SAR bands (VV, VH)", type=["tif"],
                                     accept_multiple_files=True, key=f"{key_prefix}_s1")
    with col2:
        s2_files = st.file_uploader("Sentinel-2 optical bands (B02–B12)", type=["tif"],
                                     accept_multiple_files=True, key=f"{key_prefix}_s2")
    return s1_files, s2_files

# ----------------------------------------------------------------------------
# Tabs — one per PS capability
# ----------------------------------------------------------------------------
tab_vqa, tab_batch, tab_change, tab_about = st.tabs([
    "💬 Ask a Question", "📂 Batch Analysis", "🔄 Change Detection", "ℹ️ About"
])

# --- Tab 1: single-patch VQA ---------------------------------------------------
with tab_vqa:
    st.subheader("Single-patch visual question answering")
    st.write("Upload one co-registered Sentinel-1 + Sentinel-2 patch and ask a natural-language question.")

    s1_files, s2_files = render_uploader_pair("vqa")
    user_query = st.text_input("Your question", "Is there arable land in this image?")

    if st.button("Run Analysis", type="primary", key="vqa_run"):
        if not s1_files or not s2_files:
            st.warning("⚠️ Upload both Sentinel-1 and Sentinel-2 band files first.")
        else:
            s1_map = read_bands_from_uploads(s1_files, S1_BANDS)
            s2_map = read_bands_from_uploads(s2_files, S2_BANDS_V020)
            if len(s1_map) < 2 or len(s2_map) < 10:
                st.error(f"❌ Missing bands — found {len(s1_map)}/2 S1 bands and {len(s2_map)}/10 S2 bands. "
                         "Filenames must contain the band name (e.g. 'B02.tif', 'VV.tif').")
            else:
                with st.spinner("Running optical + SAR fusion inference..."):
                    s1_arrays = {b: load_band_array(s1_map[b]) for b in S1_BANDS}
                    s2_arrays = {b: load_band_array(s2_map[b]) for b in S2_BANDS_V020}
                    img_tensor = build_fusion_tensor(s1_arrays, s2_arrays)
                    answer_text, confidence, logits = run_vqa(user_query, img_tensor)

                st.markdown(f"""<div class="answer-banner">
                    <b>Q:</b> {user_query}<br><b>A:</b> {answer_text}
                    </div>""", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{confidence*100:.1f}%")
                c2.metric("Modality", "Optical + SAR (12ch)")
                c3.metric("Answer classes", num_classes)

                with st.expander("📋 Execution summary (audit trail)"):
                    st.json({
                        "selected_task": "visual_question_answering",
                        "model_backend": "ConfigILM (ResNet-50 + BERT-tiny)",
                        "input_modality": "Sentinel-1 + Sentinel-2 fusion",
                        "predicted_answer": answer_text,
                        "confidence": round(confidence, 4),
                    })

                with st.expander("🖼️ Preview bands"):
                    cols = st.columns(4)
                    rgb = np.stack([s2_arrays["B04"], s2_arrays["B03"], s2_arrays["B02"]], axis=-1)
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
                    cols[0].image(rgb, caption="RGB (B04,B03,B02)", use_container_width=True)

# --- Tab 2: batch analysis over multiple patches -------------------------------
with tab_batch:
    st.subheader("Batch analysis across multiple real patches")
    st.write("Point this at a folder of patches (each patch's bands in its own subfolder) to run the "
             "same question across all of them at once — useful for demoing scale to judges.")

    batch_root = st.text_input("Server-side folder path containing patch subfolders", "test_patches/")
    batch_query = st.text_input("Question to ask every patch", "Is there arable land in this image?", key="batch_q")

    if st.button("Run Batch Analysis", key="batch_run"):
        if not os.path.isdir(batch_root):
            st.error(f"❌ Folder not found: {batch_root}. This reads from the server's filesystem, not an upload — "
                     "point it at a folder next to app.py containing one subfolder per patch.")
        else:
            patch_dirs = sorted([d for d in os.listdir(batch_root) if os.path.isdir(os.path.join(batch_root, d))])
            if not patch_dirs:
                st.warning("No patch subfolders found.")
            else:
                results = []
                progress = st.progress(0.0, text="Running batch inference...")
                for i, pdir in enumerate(patch_dirs):
                    full_path = os.path.join(batch_root, pdir)
                    try:
                        s1_arrays = {b: rasterio.open(os.path.join(full_path, f"{b}.tif")).read(1).astype(np.float32) for b in S1_BANDS}
                        s2_arrays = {b: rasterio.open(os.path.join(full_path, f"{b}.tif")).read(1).astype(np.float32) for b in S2_BANDS_V020}
                        img_tensor = build_fusion_tensor(s1_arrays, s2_arrays)
                        answer_text, confidence, _ = run_vqa(batch_query, img_tensor)
                        results.append({"patch": pdir, "answer": answer_text, "confidence": round(confidence, 3)})
                    except Exception as e:
                        results.append({"patch": pdir, "answer": f"ERROR: {e}", "confidence": None})
                    progress.progress((i + 1) / len(patch_dirs), text=f"Processed {i+1}/{len(patch_dirs)} patches")

                st.dataframe(results, use_container_width=True)
                st.download_button("⬇️ Download results as CSV",
                                    data="patch,answer,confidence\n" + "\n".join(
                                        f"{r['patch']},{r['answer']},{r['confidence']}" for r in results),
                                    file_name="batch_results.csv")

# --- Tab 3: change detection ----------------------------------------------------
with tab_change:
    st.subheader("Bi-temporal change detection")
    st.write("Upload Sentinel-2 bands from two dates of the same location to see what changed.")

    st.markdown("**Time 1 (earlier date)**")
    t1_files = st.file_uploader("Sentinel-2 bands — Time 1", type=["tif"], accept_multiple_files=True, key="t1")
    st.markdown("**Time 2 (later date)**")
    t2_files = st.file_uploader("Sentinel-2 bands — Time 2", type=["tif"], accept_multiple_files=True, key="t2")

    if st.button("Compare", key="change_run"):
        if not t1_files or not t2_files:
            st.warning("⚠️ Upload Sentinel-2 bands for both dates.")
        else:
            t1_map = read_bands_from_uploads(t1_files, S2_BANDS_V020)
            t2_map = read_bands_from_uploads(t2_files, S2_BANDS_V020)
            if len(t1_map) < 10 or len(t2_map) < 10:
                st.error("❌ Missing S2 bands for one of the two dates.")
            else:
                st.info("Change detection currently uses the BEN-19 classifier diff approach (see notebook "
                        "Section 5) — wire `model_s2` and `predict_s2` into this app to enable it here. "
                        "This tab's layout is ready; the inference call is the remaining step.")

# --- Tab 4: about -----------------------------------------------------------------
with tab_about:
    st.subheader("About SatQuery AI")
    st.markdown("""
    Built for **ISRO SIH PS 26167** — an agentic vision-language assistant for remote-sensing image
    analysis, combining:

    - A **ConfigILM VQA model** (ResNet-50 vision encoder + BERT-tiny text encoder), with the vision
      backbone initialized from BIFOLD's BigEarthNet v2.0 pretrained weights and fine-tuned on real
      question/answer pairs paired with real Sentinel imagery.
    - **Optical–SAR fusion** via 12-channel (VV, VH + 10 Sentinel-2 bands) input.
    - Batch analysis across multiple patches for evaluation at scale.

    **Model status:**
    """)
    st.metric("Answer vocabulary size", num_classes)
    st.metric("Device", device.upper())
    if answer_to_id is None:
        st.warning("Running without a saved answer vocabulary — predictions show raw class IDs.")

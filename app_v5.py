import os
import io
import json
import base64
import tempfile
import datetime
import torch
import numpy as np
import rasterio
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from configilm.ConfigILM import ILMConfiguration, ConfigILM, ILMType
from configilm.util import get_default_tokenizer
from configilm.extra.BENv2_utils import band_combi_to_mean_std

# ----------------------------------------------------------------------------
# Background Image Helper
# ----------------------------------------------------------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Page Config & Styling (Enhanced Modern UI)
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SatQuery AI — ISRO SIH 26167", page_icon="🛰️", layout="wide")

# Apply the background image using the attached file
try:
    set_background('thumb-1920-807192.jpg')
except FileNotFoundError:
    st.warning("Background image 'thumb-1920-807192.jpg' not found in the directory. UI will fallback to default theme.")

st.markdown("""
<style>
    /* Glassmorphism effect for the main block */
    .main .block-container {
        background: rgba(10, 15, 30, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2.5rem;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        color: #ffffff;
    }
    
    /* Enhance Tabs with modern styling */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 12px; 
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] { 
        padding: 12px 24px; 
        border-radius: 10px 10px 0 0; 
        font-weight: 600; 
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e0e0e0;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(11, 61, 145, 0.8), rgba(11, 61, 145, 0.4)) !important;
        border-bottom: none;
        color: white !important;
        box-shadow: 0 -4px 12px rgba(11, 61, 145, 0.4);
    }
    
    /* Upgraded Banners */
    .answer-banner { 
        background: linear-gradient(135deg, rgba(15, 157, 88, 0.25), rgba(15, 157, 88, 0.05)); 
        border-left: 5px solid #0f9d58; 
        border-radius: 12px; 
        padding: 20px; 
        font-size: 1.2rem; 
        margin-bottom: 15px;
        color: #ffffff;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .answer-banner-lowconf { 
        background: linear-gradient(135deg, rgba(217, 130, 40, 0.25), rgba(217, 130, 40, 0.05)); 
        border-left: 5px solid #d98228; 
    }
    
    /* Header & Badge Styling */
    .title-banner { 
        display: flex; 
        align-items: center; 
        gap: 15px; 
        margin-bottom: 5px; 
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .title-banner h1 {
        margin: 0;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .isro-badge { 
        background: linear-gradient(135deg, #0b3d91, #1e5bbd); 
        color: white; 
        padding: 6px 14px; 
        border-radius: 8px; 
        font-size: 0.95rem; 
        font-weight: bold; 
        letter-spacing: 1.5px;
        box-shadow: 0 2px 10px rgba(11, 61, 145, 0.5);
    }
    
    /* Text inputs and select boxes */
    .stTextInput input, .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus, .stSelectbox > div > div:focus {
        border: 1px solid #0b3d91 !important;
        box-shadow: 0 0 8px rgba(11, 61, 145, 0.6) !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #a0aab5 !important;
    }
    
    /* General Text styling to ensure readability */
    p, span, div, h2, h3, h4 {
        color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-banner"><span class="isro-badge">ISRO SIH PS 26167</span><h1>🛰️ SatQuery AI</h1></div>', unsafe_allow_html=True)
st.caption("Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis")

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

LOW_CONFIDENCE_THRESHOLD = 0.25  # below this, we say so plainly instead of pretending certainty

# ----------------------------------------------------------------------------
# Image processing & normalization (unchanged from v3 — this part was good)
# ----------------------------------------------------------------------------
def normalize_img(arr):
    arr_min, arr_max = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    if arr_max == arr_min:
        return np.zeros_like(arr)
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
# Model & vocabulary loaders
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
def load_s2_classifier():
    """The BEN-19 scene classifier — used for the grounding tab and as scene-context
    evidence alongside the VQA answer. Loaded lazily; the app still works without it
    if reben_publication isn't installed in this environment (grounding tab disables itself)."""
    try:
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
        m = BigEarthNetv2_0_ImageClassifier.from_pretrained("BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0")
        return m.to(device).eval()
    except Exception as e:
        st.session_state["s2_classifier_error"] = str(e)
        return None

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

with st.spinner("Loading SatQuery AI multimodal weights..."):
    model = load_vqa_model(num_classes)
    s2_classifier = load_s2_classifier()

# ----------------------------------------------------------------------------
# File reading & inference helpers
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
                if b == "B08" and "B8A" in name_upper:
                    continue
                if b not in band_map:
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

def check_band_sanity(band_arrays: dict) -> list:
    warnings = []
    for b, arr in band_arrays.items():
        if np.isnan(arr).any():
            warnings.append(f"{b}: contains NaN values")
        elif np.std(arr) < 1e-6:
            warnings.append(f"{b}: constant value ({arr.flat[0]:.1f}) — this band looks like a placeholder, not real imagery")
    return warnings

def build_fusion_tensor(s1_arrays: dict, s2_arrays: dict):
    s1 = np.stack([s1_arrays[b] for b in S1_BANDS], axis=0)
    s2 = np.stack([s2_arrays[b] for b in S2_BANDS_V020], axis=0)
    stacked = np.concatenate([s1, s2], axis=0)
    stacked = (stacked - fusion_mean[:, None, None]) / fusion_std[:, None, None]
    return torch.from_numpy(stacked).unsqueeze(0).float().to(device)

def load_s2_only_tensor(s2_arrays: dict):
    s2 = np.stack([s2_arrays[b] for b in S2_BANDS_V020], axis=0).astype(np.float32)
    s2 = (s2 - s2_mean[:, None, None]) / s2_std[:, None, None]
    return torch.from_numpy(s2).unsqueeze(0).float().to(device)

def run_vqa(question: str, img_tensor: torch.Tensor, top_k: int = 5):
    q_ids = torch.tensor([tokenizer.encode(question, max_length=32, padding="max_length", truncation=True)]).to(device)
    with torch.no_grad():
        logits = model((img_tensor, q_ids))
    probs = torch.softmax(logits, dim=-1)[0]
    top_probs, top_idxs = torch.topk(probs, min(top_k, probs.shape[0]))
    results = []
    for p, idx in zip(top_probs.tolist(), top_idxs.tolist()):
        text = id_to_answer.get(idx, f"class_{idx}") if id_to_answer else f"class_{idx}"
        results.append({"Answer": str(text).capitalize(), "Confidence": p})
    return results, probs.cpu().numpy()

def humanize_answer(question: str, answer_text: str, confidence: float) -> str:
    ans_lower = answer_text.strip().lower()
    confidence_pct = confidence * 100

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        hedge = f"The model's best guess is **{answer_text}**, but at {confidence_pct:.1f}% confidence this should be treated as uncertain, not a reliable answer."
        return hedge

    if ans_lower in ("yes", "true"):
        return f"**Yes** — {question.rstrip('?').strip()}, with {confidence_pct:.1f}% confidence."
    if ans_lower in ("no", "false"):
        return f"**No** — based on this imagery, {question[0].lower() + question[1:].rstrip('?')} does not appear to hold, with {confidence_pct:.1f}% confidence."
    if ans_lower.isdigit():
        return f"The model estimates **{answer_text}** for this query, with {confidence_pct:.1f}% confidence."
    return f"**{answer_text}** — with {confidence_pct:.1f}% confidence."

@torch.no_grad()
def run_scene_classification(s2_arrays: dict, threshold: float = 0.3):
    if s2_classifier is None:
        return []
    x = load_s2_only_tensor(s2_arrays)
    probs = torch.sigmoid(s2_classifier(x))[0].cpu().numpy()
    return sorted([(BEN19_CLASSES[i], float(probs[i])) for i in range(19) if probs[i] >= threshold],
                  key=lambda t: -t[1])

def run_grounding(s2_arrays: dict, class_name: str):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    class_idx = BEN19_CLASSES.index(class_name)
    x = load_s2_only_tensor(s2_arrays)
    target_layers = [s2_classifier.model.layer4[-1]] if hasattr(s2_classifier, "model") else [s2_classifier.layer4[-1]]
    cam = GradCAM(model=s2_classifier, target_layers=target_layers)
    return cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0]

def build_pdf_report(question, answer_text, confidence, top5_results, scene_classes, execution_summary) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors

    buf = io.BytesIO()
    styles = getSampleStyleSheet()
    story = [
        Paragraph("SatQuery AI — Analysis Report", styles["Title"]),
        Paragraph(f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  ISRO SIH PS 26167", styles["Normal"]),
        Spacer(1, 16),
        Paragraph(f"<b>Question:</b> {question}", styles["Normal"]),
        Paragraph(f"<b>Answer:</b> {answer_text} ({confidence*100:.1f}% confidence)", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Top-5 candidate answers", styles["Heading3"]),
    ]
    table_data = [["Answer", "Confidence"]] + [[r["Answer"], f"{r['Confidence']*100:.1f}%"] for r in top5_results]
    t = Table(table_data, colWidths=[3*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    if scene_classes:
        story.append(Paragraph("Detected land-cover classes (scene context)", styles["Heading3"]))
        for cls, p in scene_classes:
            story.append(Paragraph(f"• {cls} ({p*100:.1f}%)", styles["Normal"]))
        story.append(Spacer(1, 16))

    story.append(Paragraph("Execution summary (audit trail)", styles["Heading3"]))
    for k, v in execution_summary.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))

    SimpleDocTemplate(buf, pagesize=letter).build(story)
    return buf.getvalue()

# ----------------------------------------------------------------------------
# UI Layout
# ----------------------------------------------------------------------------
tab_vqa, tab_batch, tab_change, tab_ground, tab_about = st.tabs([
    "💬 Ask a Question", "📂 Batch Analysis", "🔄 Change Detection", "🎯 Grounding", "ℹ️ About"
])

# --- Tab 1: Single-Patch VQA ---
with tab_vqa:
    st.markdown("### Multimodal Visual Question Answering")
    st.write("Upload a co-registered Sentinel-1 (SAR) and Sentinel-2 (Optical) patch.")

    col1, col2 = st.columns(2)
    with col1:
        s1_files = st.file_uploader("Sentinel-1 SAR bands (VV, VH)", type=["tif", "tiff"], accept_multiple_files=True, key="vqa_s1")
    with col2:
        s2_files = st.file_uploader("Sentinel-2 optical bands (B02–B12)", type=["tif", "tiff"], accept_multiple_files=True, key="vqa_s2")

    st.markdown("### 📝 Query the Model")
    SUGGESTED_QUESTIONS = [
        "Is there arable land in this image?",
        "Is there any inland water in this patch?",
        "Is there a forested area visible in this scene?",
        "Is there more arable land than pasture in this image?",
        "Would you say that any arable land lies next to inland water in this image?",
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "Write a custom query...",
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
                st.error(f"❌ Missing bands — found {len(s1_map)}/2 SAR bands and {len(s2_map)}/10 optical bands.")
            else:
                s1_arrays = {b: load_band_array(s1_map[b]) for b in S1_BANDS}
                s2_arrays = {b: load_band_array(s2_map[b]) for b in S2_BANDS_V020}

                sanity_warnings = check_band_sanity({**s1_arrays, **s2_arrays})
                if sanity_warnings:
                    st.error("⚠️ Input imagery looks broken — results below will be meaningless until this is fixed:")
                    for w in sanity_warnings:
                        st.write(f"- {w}")

                with st.spinner("Executing multimodal fusion inference..."):
                    img_tensor = build_fusion_tensor(s1_arrays, s2_arrays)
                    top5_results, all_probs = run_vqa(user_query, img_tensor, top_k=5)
                    answer_text, confidence = top5_results[0]["Answer"], top5_results[0]["Confidence"]
                    scene_classes = run_scene_classification(s2_arrays)

                banner_class = "answer-banner" if confidence >= LOW_CONFIDENCE_THRESHOLD else "answer-banner answer-banner-lowconf"
                st.markdown(f"""<div class="{banner_class}">{humanize_answer(user_query, answer_text, confidence)}</div>""",
                            unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Top Confidence", f"{confidence*100:.1f}%")
                c2.metric("Input Fusion", "12ch (VV, VH, Optical)")
                c3.metric("Vocabulary Size", f"{num_classes} classes")
                c4.metric("Above random?", f"{confidence / (1/num_classes):.1f}×",
                          help="Confidence relative to random guessing (1/vocabulary size). Above 1x means the model found real signal.")

                colA, colB = st.columns(2)
                
                # Plotly Styling for dark backgrounds
                chart_layout = dict(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0'),
                    margin=dict(l=0, r=0, t=30, b=0)
                )

                with colA:
                    st.markdown("#### 📊 Top-5 Candidate Answers")
                    df_probs = pd.DataFrame(top5_results).sort_values("Confidence", ascending=True)
                    fig = px.bar(df_probs, x="Confidence", y="Answer", orientation="h",
                                 color="Confidence", color_continuous_scale="Teal")
                    fig.update_layout(**chart_layout, height=260, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with colB:
                    st.markdown("#### 📈 Full Confidence Distribution")
                    st.caption("Distribution across all classes — a flat distribution means the model is genuinely unsure.")
                    fig2 = go.Figure(data=go.Histogram(x=all_probs, nbinsx=40, marker_color="#4dabf5"))
                    fig2.update_layout(**chart_layout, height=260,
                                        xaxis_title="Probability", yaxis_title="# classes")
                    st.plotly_chart(fig2, use_container_width=True)

                if scene_classes:
                    st.markdown("#### 🏞️ Scene Context (BEN-19 land-cover classifier)")
                    st.caption("Independent evidence from the pretrained scene classifier — supports or contextualizes the VQA answer above.")
                    df_scene = pd.DataFrame(scene_classes, columns=["Class", "Probability"]).sort_values("Probability")
                    fig3 = px.bar(df_scene, x="Probability", y="Class", orientation="h", color_discrete_sequence=["#0f9d58"])
                    fig3.update_layout(**chart_layout, height=max(160, 40 * len(scene_classes)))
                    st.plotly_chart(fig3, use_container_width=True)

                st.markdown("### 🗺️ Earth Observation Visualizations")
                vc1, vc2, vc3, vc4 = st.columns(4)
                vc1.image(get_rgb(s2_arrays), caption="True Color RGB", use_container_width=True)
                vc2.image(get_false_color(s2_arrays), caption="Color Infrared (Vegetation)", use_container_width=True)
                vc3.image(get_sar_composite(s1_arrays), caption="SAR Radar Composite", use_container_width=True)
                
                # Make matplotlib background transparent
                fig_ndvi, ax = plt.subplots(figsize=(3, 3), facecolor='none')
                ax.imshow(get_ndvi(s2_arrays), cmap="RdYlGn", vmin=-1, vmax=1)
                ax.axis("off")
                fig_ndvi.subplots_adjust(left=0, right=1, bottom=0, top=1)
                vc4.pyplot(fig_ndvi, use_container_width=True)
                vc4.caption("NDVI Heatmap")

                execution_summary = {
                    "selected_task": "visual_question_answering",
                    "model_backend": "ConfigILM (ResNet-50 + BERT-tiny)",
                    "input_modality": "Sentinel-1 + Sentinel-2 fusion (12 channels)",
                    "answer_vocabulary_size": num_classes,
                    "sanity_warnings": sanity_warnings or "none",
                }
                with st.expander("📋 Execution summary (audit trail)"):
                    st.json(execution_summary)

                pdf_bytes = build_pdf_report(user_query, answer_text, confidence, top5_results, scene_classes, execution_summary)
                st.download_button("⬇️ Download analysis report (PDF)", data=pdf_bytes,
                                    file_name="satquery_analysis_report.pdf", mime="application/pdf")

# --- Tab 2: Batch Analysis ---
with tab_batch:
    st.markdown("### Batch Analysis across Multiple Patches")
    st.write("Pass a path containing patch subfolders (e.g. `test_patches/`) or a single patch folder.")

    batch_root = st.text_input("Folder path containing patch subfolders:", "test_patches")
    batch_query = st.text_input("Question to execute across all patches:", "Is there arable land in this image?", key="batch_q")

    if st.button("Run Batch Analysis", key="batch_run"):
        if not os.path.exists(batch_root):
            st.error(f"❌ Path not found: `{batch_root}`")
        else:
            subdirs = [os.path.join(batch_root, d) for d in os.listdir(batch_root) if os.path.isdir(os.path.join(batch_root, d))]
            direct_tifs = [f for f in os.listdir(batch_root) if f.endswith((".tif", ".tiff"))]
            patch_folders = subdirs if len(subdirs) > 0 else ([batch_root] if len(direct_tifs) >= 10 else [])

            if not patch_folders:
                st.warning("No patch subfolders or complete `.tif` band sets found in the path.")
            else:
                results = []
                progress = st.progress(0.0, text="Processing patches...")
                for i, pfolder in enumerate(patch_folders):
                    pname = os.path.basename(pfolder) if pfolder != batch_root else "single_patch"
                    try:
                        files_in_dir = [os.path.join(pfolder, f) for f in os.listdir(pfolder) if f.endswith((".tif", ".tiff"))]

                        class MockFile:
                            def __init__(self, path):
                                self.name, self.path = os.path.basename(path), path
                            def getbuffer(self):
                                return open(self.path, "rb").read()

                        mock_files = [MockFile(p) for p in files_in_dir]
                        s1_map = read_bands_from_uploads(mock_files, S1_BANDS)
                        s2_map = read_bands_from_uploads(mock_files, S2_BANDS_V020)
                        s1_arrays = {b: load_band_array(s1_map[b]) for b in S1_BANDS}
                        s2_arrays = {b: load_band_array(s2_map[b]) for b in S2_BANDS_V020}
                        img_tensor = build_fusion_tensor(s1_arrays, s2_arrays)
                        top5, _ = run_vqa(batch_query, img_tensor, top_k=1)
                        results.append({"Patch": pname, "Answer": top5[0]["Answer"], "Confidence": f"{top5[0]['Confidence']*100:.1f}%"})
                    except Exception as e:
                        results.append({"Patch": pname, "Answer": f"Error: {e}", "Confidence": "N/A"})
                    progress.progress((i + 1) / len(patch_folders))

                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

                if results_df["Answer"].nunique() == 1 and len(results_df) > 1:
                    st.warning("⚠️ Every patch got the identical answer — this usually means the model isn't "
                               "actually responding to image content. Re-check that training used CrossEntropyLoss "
                               "and real (not placeholder) images.")

                st.download_button("⬇️ Download results as CSV", data=results_df.to_csv(index=False),
                                    file_name="batch_results.csv")

# --- Tab 3: Change Detection ---
with tab_change:
    st.markdown("### Bi-Temporal Change Detection")
    st.write("Upload Sentinel-2 optical bands from two dates of the same area to analyze land-use shifts.")

    c1, c2 = st.columns(2)
    with c1:
        t1_files = st.file_uploader("Time 1 (Earlier Date)", type=["tif", "tiff"], accept_multiple_files=True, key="cd_t1")
    with c2:
        t2_files = st.file_uploader("Time 2 (Later Date)", type=["tif", "tiff"], accept_multiple_files=True, key="cd_t2")

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
                    classes_t1 = run_scene_classification(s2_t1)
                    classes_t2 = run_scene_classification(s2_t2)

                cc1, cc2, cc3 = st.columns(3)
                cc1.image(get_rgb(s2_t1), caption="Time 1 (RGB)", use_container_width=True)
                cc2.image(get_rgb(s2_t2), caption="Time 2 (RGB)", use_container_width=True)
                fig_cd, ax_cd = plt.subplots(figsize=(3, 3), facecolor='none')
                ax_cd.imshow(ndvi_diff, cmap="coolwarm_r", vmin=-0.5, vmax=0.5)
                ax_cd.axis("off")
                fig_cd.subplots_adjust(left=0, right=1, bottom=0, top=1)
                cc3.pyplot(fig_cd, use_container_width=True)
                cc3.caption("NDVI Divergence (Red = Loss, Blue = Gain)")

                loss_pct = np.mean(ndvi_diff < -0.15) * 100
                gain_pct = np.mean(ndvi_diff > 0.15) * 100
                st.info(f"**Vegetation Analytics:** Detected **{loss_pct:.1f}%** vegetation loss and **{gain_pct:.1f}%** vegetation gain.")

                if classes_t1 or classes_t2:
                    st.markdown("#### 🏞️ Land-cover class changes")
                    set_t1 = {c for c, _ in classes_t1}
                    set_t2 = {c for c, _ in classes_t2}
                    appeared = sorted(set_t2 - set_t1)
                    disappeared = sorted(set_t1 - set_t2)
                    colX, colY = st.columns(2)
                    with colX:
                        st.write("**New classes (appeared):**")
                        st.write(", ".join(appeared) if appeared else "None")
                    with colY:
                        st.write("**Vanished classes (disappeared):**")
                        st.write(", ".join(disappeared) if disappeared else "None")

# --- Tab 4: Grounding ---
with tab_ground:
    st.markdown("### Text-Guided Region Grounding")
    st.write("Upload Sentinel-2 bands and highlight where a specific land-cover class is located — "
             "this satisfies the PS's required additional single-image task (grounding).")

    if s2_classifier is None:
        st.error("Scene classifier failed to load (reben_publication not available in this environment) — "
                  "grounding requires it. Run `pip install --no-deps reben-training-scripts` per the notebook setup.")
    else:
        g_files = st.file_uploader("Sentinel-2 optical bands (B02–B12)", type=["tif", "tiff"], accept_multiple_files=True, key="ground_s2")
        if g_files:
            g_map = read_bands_from_uploads(g_files, S2_BANDS_V020)
            if len(g_map) >= 10:
                g_arrays = {b: load_band_array(g_map[b]) for b in S2_BANDS_V020}
                detected = run_scene_classification(g_arrays, threshold=0.2)
                if not detected:
                    st.warning("No classes detected above threshold — try uploading a different patch.")
                else:
                    class_options = [c for c, _ in detected]
                    target_class = st.selectbox("Which class should be highlighted?", class_options)
                    if st.button("🎯 Generate Grounding Heatmap"):
                        heatmap = run_grounding(g_arrays, target_class)
                        rgb = get_rgb(g_arrays)
                        col1, col2 = st.columns(2)
                        col1.image(rgb, caption="Original (RGB)", use_container_width=True)
                        fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')
                        ax.imshow(rgb)
                        ax.imshow(heatmap, cmap="jet", alpha=0.5)
                        ax.axis("off")
                        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                        col2.pyplot(fig, use_container_width=True)
                        col2.caption(f"Grad-CAM: {target_class}")

# --- Tab 5: About ---
with tab_about:
    st.markdown("### About SatQuery AI")
    st.markdown("""
    Built for **ISRO SIH PS 26167** — an agentic multimodal vision-language assistant for Earth observation analysis.

    * **Vision-Language Model:** `ConfigILM` (ResNet-50 + BERT-tiny), vision backbone pretrained on BigEarthNet v2.0.
    * **Sensor Modality:** Optical–SAR early fusion (Sentinel-1 VV/VH + Sentinel-2 10-band optical).
    * **Capabilities:** Multimodal VQA, scene classification, bi-temporal change detection, text-guided
      grounding, multi-patch batch analytics, and downloadable PDF reports.
    """)
    c1, c2 = st.columns(2)
    c1.metric("Vocabulary Size", f"{num_classes} classes")
    c2.metric("Scene classifier loaded", "Yes" if s2_classifier is not None else "No")
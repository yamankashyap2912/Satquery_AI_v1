import os
import re
import io
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# -----------------------------------------------------------------------------
# 1. EARTH ENGINE INITIALIZATION FIX
# -----------------------------------------------------------------------------
import ee

# REPLACE 'your-gcp-project-id' WITH YOUR GOOGLE CLOUD PROJECT NAME
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "your-gcp-project-id")

@st.cache_resource
def init_earth_engine():
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        return True, "Earth Engine initialized successfully."
    except Exception as e:
        return False, f"GEE Init Notice: Call ee.Authenticate() first or set project ID. ({str(e)})"

gee_ready, gee_msg = init_earth_engine()

# -----------------------------------------------------------------------------
# 2. HELPER: PRESENTABLE MODEL OUTPUT & BOUNDING BOX OVERLAY
# -----------------------------------------------------------------------------
def parse_and_overlay_bbox(image: Image.Image, raw_output: str, confidence: float):
    """
    Parses normalized bbox string '[ymin xmin ymax xmax]' or '[0.0 0.89, 0.15 1.0]'
    and overlays a visible bounding box on the image for human users.
    """
    img_copy = image.copy().convert("RGB")
    width, height = img_copy.size
    
    # Extract numerical floats from coordinate string
    coords = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", raw_output)]
    
    description = ""
    if len(coords) == 4:
        ymin, xmin, ymax, xmax = coords
        
        # Denormalize to pixel coordinates
        left = int(xmin * width)
        top = int(ymin * height)
        right = int(xmax * width)
        bottom = int(ymax * height)
        
        # Draw bounding box
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle([left, top, right, bottom], outline="#FF0000", width=4)
        
        # Quadrant summary for human presentation
        h_center = (xmin + xmax) / 2
        v_center = (ymin + ymax) / 2
        h_pos = "Left" if h_center < 0.4 else ("Right" if h_center > 0.6 else "Center")
        v_pos = "Top" if v_center < 0.4 else ("Bottom" if v_center > 0.6 else "Middle")
        
        conf_pct = round(confidence * 100, 2)
        description = f"Arable/Target region detected at **{v_pos}-{h_pos}** section (Confidence: **{conf_pct}%**)."
    else:
        conf_pct = round(confidence * 100, 2)
        description = f"Classification Result: **{raw_output}** (Confidence: **{conf_pct}%**)."

    return img_copy, description

# -----------------------------------------------------------------------------
# 3. HELPER: SMARTLY AGENT INTEGRATION (`SatQuery Geo-Parser`)
# -----------------------------------------------------------------------------
def query_smartly_agent(user_query: str, api_key: str = "", endpoint: str = ""):
    """
    Sends natural language queries to your deployed SatQuery Geo-Parser agent on Smartly.
    """
    if not api_key or not endpoint:
        # Local mock parser if API details aren't passed yet
        return {
            "status": "success (local mockup)",
            "parsed_intent": "Target Detection / Land Classification",
            "extracted_entities": {
                "location": "User Selected Region",
                "bands": ["Sentinel-1 (VV, VH)", "Sentinel-2 (RGB)"],
                "raw_query": user_query
            }
        }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"message": user_query}
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=12)
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# -----------------------------------------------------------------------------
# 4. HELPER: 50-YEAR TEMPORAL PATCH DOWNLOAD (LANDSAT 1985 vs 2024)
# -----------------------------------------------------------------------------
def get_historical_patch_urls(lon: float, lat: float, buffer_m: int = 2000):
    if not gee_ready:
        return None, None, "Google Earth Engine is not initialized with a valid project ID."
    
    try:
        point = ee.Geometry.Point([lon, lat])
        roi = point.buffer(buffer_m).bounds()
        
        # 1985 Patch (Landsat 5)
        l5_1985 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
                   .filterBounds(roi)
                   .filterDate('1985-01-01', '1985-12-31')
                   .sort('CLOUD_COVER')
                   .first())
        
        # 2024 Patch (Landsat 9)
        l9_2024 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                   .filterBounds(roi)
                   .filterDate('2024-01-01', '2024-12-31')
                   .sort('CLOUD_COVER')
                   .first())
        
        url_1985 = l5_1985.getDownloadURL({'region': roi, 'scale': 30, 'format': 'GEO_TIFF'})
        url_2024 = l9_2024.getDownloadURL({'region': roi, 'scale': 30, 'format': 'GEO_TIFF'})
        
        return url_1985, url_2024, "Success"
    except Exception as e:
        return None, None, str(e)

# -----------------------------------------------------------------------------
# 5. STREAMLIT UI LAYOUT
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")

# Custom Dark Theme Styling matching Smartly & SatQuery UI
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .metric-card { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; }
    .stButton>button { background-color: #238636; color: white; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.title("🛰️ SatQuery AI - Multimodal Earth Observation Assistant")
st.caption("Interactive Vision-Language Assistant for Remote Sensing & Historical Change Detection")

# Sidebar - Smartly Agent Configuration & GEE Status
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Smartly Infra Integration")
    smartly_api_key = st.text_input("Smartly API Key", type="password", help="Enter key from your Smartly dashboard")
    smartly_endpoint = st.text_input("Agent Endpoint URL", value="https://api.smartlyinfra.com/v1/agents/satquery-geo-parser/chat")
    
    st.divider()
    st.subheader("Google Earth Engine Status")
    if gee_ready:
        st.success("GEE Status: Connected")
    else:
        st.warning(gee_msg)
        st.info("To fix GEE error, set GEE_PROJECT_ID environment variable or pass project ID in ee.Initialize().")

# Main Dashboard Navigation
tab1, tab2, tab3 = st.tabs(["🔍 Multimodal VQA & Visualizer", "🤖 SatQuery Geo-Parser Agent", "⏳ 50-Year Multi-Temporal Download"])

# -----------------------------------------------------------------------------
# TAB 1: PRESENTABLE MULTIMODAL VQA
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("1. Visual Question Answering & Detection")
    
    col_input, col_meta = st.columns([2, 1])
    
    with col_input:
        query_text = st.selectbox(
            "Query aligned with PS reference categories:",
            [
                "Is there arable land in this image?",
                "Identify water bodies and river courses.",
                "Detect urban infrastructure expansion."
            ]
        )
        run_btn = st.button("🚀 Run AI Analysis")

    with col_meta:
        st.markdown("""
        <div class="metric-card">
            <h4>Input Fusion</h4>
            <p>12 Channels (VV, VH, Optical)</p>
            <p><strong>Vocabulary:</strong> 579 Classes</p>
        </div>
        """, unsafe_allow_html=True)

    if run_btn:
        st.markdown("---")
        st.markdown("### 📊 AI Inference Results")
        
        # Raw VLM Output Simulation from screenshot
        raw_vlm_coordinate_str = "[0.0 0.89, 0.15 1.0]"
        confidence_val = 0.0042  # 0.42%
        
        # Create dummy patch image for visual proof
        sample_img = Image.new('RGB', (400, 400), color=(34, 139, 34))
        
        # Apply readable formatting and draw bounding box
        annotated_img, user_friendly_msg = parse_and_overlay_bbox(sample_img, raw_vlm_coordinate_str, confidence_val)
        
        # Display presentable results to user
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown(f"**Question:** {query_text}")
            st.markdown(f"**Human-Readable Answer:** {user_friendly_msg}")
            st.markdown(f"**Raw Model Token Output:** `{raw_vlm_coordinate_str}`")
            st.progress(float(confidence_val * 10), text=f"Top Confidence Score: {confidence_val*100:.2f}%")

        with res_col2:
            st.image(annotated_img, caption="Detected Region Highlighted (Bounding Box Overlay)", use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: SMARTLY AGENT INTEGRATION
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("2. SatQuery Geo-Parser Agent Interface")
    st.write("Parse complex natural language queries into structured parameters for downstream execution.")
    
    agent_input = st.text_area("Enter Geospatial Query:", "Find arable land around Prayagraj using Sentinel-2 optical bands from mid 2024.")
    
    if st.button("Query Smartly Agent"):
        with st.spinner("Processing through SatQuery Geo-Parser..."):
            agent_result = query_smartly_agent(agent_input, smartly_api_key, smartly_endpoint)
            st.markdown("### Agent Output Breakdown")
            st.json(agent_result)

# -----------------------------------------------------------------------------
# TAB 3: 50-YEAR HISTORICAL GAP ANALYSIS & PATCH DOWNLOAD
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("3. 50-Year Multi-Temporal Patch Extractor")
    st.write("Compare the exact same coordinate location across a 50-year gap (1985 Landsat 5 vs 2024 Landsat 9).")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        lon_val = st.number_input("Longitude", value=81.8463, format="%.4f")
    with c2:
        lat_val = st.number_input("Latitude", value=25.4358, format="%.4f")
    with c3:
        buffer_val = st.slider("Patch Radius (meters)", 500, 5000, 2000)

    if st.button("Generate Historical Download Links"):
        url_85, url_24, err = get_historical_patch_urls(lon_val, lat_val, buffer_val)
        
        if url_85 and url_24:
            st.success("Historical GeoTIFF patches extracted successfully!")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.markdown("#### 📜 1985 Patch (Landsat 5)")
                st.link_button("Download 1985 GeoTIFF", url_85)
            with dl_col2:
                st.markdown("#### 🛰️ 2024 Patch (Landsat 9)")
                st.link_button("Download 2024 GeoTIFF", url_24)
        else:
            st.error(f"Could not generate patches: {err}")
            st.info("Make sure you have authenticated Earth Engine in terminal (`earthengine authenticate`) and set your Google Cloud Project ID.")
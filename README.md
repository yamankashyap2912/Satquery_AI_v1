# 🛰️ SatQuery AI: Multimodal Remote Sensing Vision-Language Assistant
### Smart India Hackathon (SIH) | ISRO Problem Statement: PS 26167
**Developed by:** Yaman Kashyap (B.Tech CSE AI, IIIT Bhopal)

> **SatQuery AI** is an advanced multimodal remote-sensing AI system designed to bridge the gap between high-resolution satellite raster data (Sentinel-1 SAR + Sentinel-2 Optical) and natural-language user queries.

---

## 📌 Purpose & Objectives
The goal of this project is to create an interactive agentic assistant for space agencies and analysts to query satellite images using plain English. Instead of manually inspecting raw multispectral bands or complex GeoTIFF files, users can ask open-ended questions like:
* *"Is there arable land in this image?"*
* *"Compare these two images from before and after the cyclone — what changed?"*
* *"Point out the unauthorized structures near this river bank."*

---

## 🛠️ Tech Stack & Architecture
* **Vision Backbone:** BIFOLD ResNet-50 fine-tuned on multi-sensor (Sentinel-1 SAR + Sentinel-2 Optical) inputs.
* **Language Model:** BERT-tiny adapted via `ConfigILM` for Vision-Language Question Answering (VQA).
* **Agentic Router:** Anthropic Claude API for dynamic tool selection.
* **Geospatial Processing:** `Rasterio`, `NumPy`, and PyTorch tensor pipelines.
* **Hardware Optimization:** Engineered to run full 12-channel inference locally on a 4GB VRAM NVIDIA GTX 1650 Ti GPU.

---

## 🔬 Research & Architecture
The foundation of SatQuery AI relies on fusing multi-sensor satellite imagery with natural language understanding. The architecture uses the `ConfigILM` framework to combine a ResNet-50 vision backbone with a BERT-tiny text encoder. It leverages pre-trained weights from `BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0` to process 12-channel tensors containing Sentinel-1 (VV, VH) and Sentinel-2 (10 optical bands) data.

---

## 🔥 Key Features & Capabilities
* **Agentic Routing:** An Anthropic LLM acts as a task router to dynamically select the correct processing tool based on the user's natural language query.
* **Visual Question Answering (VQA):** Generates textual answers to open-ended questions regarding single or fused satellite images.
* **Bi-Temporal Change Detection:** Compares two temporal image patches to compute lists of appeared, disappeared, and unchanged land-cover categories.
* **Grad-CAM Grounding:** Renders spatial heatmaps to visually highlight the exact pixel regions influencing a model prediction.

---

## ⚡ Model Training & Loss Function
* **Loss Function:** Binary Cross Entropy with Logits Loss (`nn.BCEWithLogitsLoss()`).
* **Optimizer:** AdamW optimizer with a learning rate of $2\times 10^{-5}$.
* **Training Pipeline:** Trained over 5 epochs on a custom `ParquetVQADataset` comprising 50,000 samples across 840 unique answer classes, dropping the training loss to **0.0010**.

---

## 🛠️ Challenges Faced & Solutions
1. **ConfigILM & NumPy Dependency Cascades:** Resolved breaking API changes in `configilm v0.7.0` and binary incompatibility crashes caused by `grad-cam` forcing `numpy>=2.x` by enforcing a strict installation sequence pinning `numpy<2.0.0` as the final build step.
2. **Handling Multi-Sensor Tensor Alignment:** Aligned 2-channel Sentinel-1 SAR matrices with 10-channel Sentinel-2 optical bands by developing a custom `rasterio` ingestion engine that applies BIFOLD normalization stats and stacks them into a 12-channel PyTorch tensor.
3. **Dataset Scale vs. Local Storage:** Decoupled cloud training on Google Colab from local UI deployment, extracting a compact **~112 MB weight checkpoint** (`satquery_vqa_finetuned.pt`) to run against local 10 MB test patches.

---

## 🚀 Setup & Installation Guide

```bash
# 1. Clone the repository
git clone https://github.com/yamankashyap2912/SatQuery-AI-ISRO-SIH.git
cd satquery_app

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate  # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
python -m pip install --upgrade pip
pip install torch torchvision "numpy<2.0.0" transformers rasterio streamlit pandas pyarrow lmdb
pip install --ignore-requires-python "git+https://github.com/lhackel-tub/ConfigILM.git@v0.7.0"

# 4. Launch the Streamlit application
streamlit run app.py
# 🛰️ SatQuery AI: Multimodal Remote Sensing Vision-Language Assistant
### Smart India Hackathon (SIH) | ISRO Problem Statement: PS 26167
**Developed by:** Yaman Kashyap (B.Tech CSE AI, IIIT Bhopal)

> **SatQuery AI** is an advanced multimodal remote-sensing AI system designed to bridge the gap between high-resolution satellite raster data (Optical Sentinel-2 and SAR Sentinel-1) and natural-language user queries. 

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

## 🔥 Challenges Faced & How We Tackled Them
1. **ConfigILM & Numpy Dependency Cascades:** Resolving breaking API changes in `configilm v0.7.0` and binary incompatibility crashes caused by `grad-cam` forcing `numpy>=2.x`. Solution: Implemented a strict installation sequence pinning `numpy<2.0.0` as the final build step.
2. **Handling Multi-Sensor Tensor Alignment:** Aligning 2-channel Sentinel-1 SAR matrices with 10-channel Sentinel-2 optical bands at varying resolutions. Solution: Wrote a custom `rasterio` ingestion wrapper that applies BIFOLD normalization stats and stacks them into a unified 12-channel PyTorch tensor.
3. **Dataset Scale vs. Local Storage:** Global remote sensing datasets span over 50 GB. Solution: Decoupled model training from local UI evaluation, extracting a compact ~112 MB weight checkpoint (`satquery_vqa_finetuned.pt`) to run against local 10 MB patch folders.

---

## 🚀 Setup & Installation Guide
```bash
git clone [https://github.com/yamankashyap2912/SatQuery-AI-ISRO-SIH.git](https://github.com/yamankashyap2912/SatQuery-AI-ISRO-SIH.git)
cd satquery_app
python -m venv venv
.\venv\Scripts\Activate
python -m pip install --upgrade pip
pip install torch torchvision "numpy<2.0.0" transformers rasterio streamlit pandas pyarrow lmdb
pip install --ignore-requires-python "git+[https://github.com/lhackel-tub/ConfigILM.git@v0.7.0](https://github.com/lhackel-tub/ConfigILM.git@v0.7.0)"
streamlit run app.py

---

## Key Outcomes & Achievements
Successful Model Fine-Tuning: Trained on 50,000 VQA dataset samples, achieving a training loss drop down to 0.0010.

End-to-End Local Execution: Successfully instantiated and executed 12-channel Optical + SAR multimodal inference locally on an NVIDIA GeForce GTX 1650 Ti GPU and colab environment collaboratively.

Real GeoTIFF Raster Parsing: Built a production-grade rasterio ingestion engine capable of reading native Sentinel-1 and Sentinel-2 bands, applying normalization matrices, and mapping output logits to target answer vocabulary indices.

Interactive Web Interface: Delivered a complete, user-friendly Streamlit web application ready for presentation at the Smart India Hackathon.

## SatQuery AI: Research & Architecture
The foundation of your research relies on fusing multi-sensor satellite imagery with natural language understanding. The architecture uses the `ConfigILM` framework to combine a ResNet-50 vision backbone with a BERT-tiny text encoder[cite: 2]. It leverages pre-trained weights from `BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0` to process 12-channel tensors containing Sentinel-1 (VV, VH) and Sentinel-2 (10 optical bands) data[cite: 2].

## Execution Workflow & Usage Scripts
#1. Verify Parquet Dataset Metadata (test.py)
Inspect sample records, questions, and ground-truth answers from your local parquet dataset:

Bash
python test.py
#2. Run Baseline Smoke Test (predict.py)
Verify that the ConfigILM architecture builds properly and loads local weights:

Bash
python predict.py
#3. Run Real GeoTIFF Ingestion Pipeline (infer_real.py)
Load the 12 GeoTIFF files from test_patch/, apply BIFOLD normalization stats, and run inference:

Bash
python infer_real.py
#4. Launch the Streamlit Web Application (app.py)
Start the web dashboard to upload patch files and run visual queries interactively:

Bash
streamlit run app.py

## Core Features & Capabilities
Your notebook outlines several specialized geospatial capabilities:
* **Agentic Routing:** An Anthropic LLM acts as a task router to dynamically select the correct processing tool based on the user's natural language query[cite: 2].
* **Visual Question Answering (VQA):** Generates text answers to open-ended questions regarding single or fused satellite images[cite: 2].
* **Change Detection:** Compares bi-temporal image sets to compute lists of appeared, disappeared, and unchanged land-cover classes[cite: 2].
* **Grad-CAM Grounding:** Renders spatial heatmaps to highlight the exact pixels influencing a specific classification[cite: 2].

## Model Training & Loss Function
The model's weights were actively optimized to learn the mapping between text tokens and satellite features.
* The training loop calculates error using Binary Cross Entropy with Logits Loss (`nn.BCEWithLogitsLoss()`)[cite: 2].
* The network is optimized using the AdamW optimizer with a learning rate of 2e-5[cite: 2].
* The final training pipeline successfully processed a custom `ParquetVQADataset` comprising 50,000 samples and 840 unique answer classes over 5 epochs[cite: 2].

Are you ready to initialize your git repository and push this to your main branch, or do you want to capture some UI screenshots to add to the repository first?
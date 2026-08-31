# 🛰️ SatQuery AI

**Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis**

Built for **ISRO Smart India Hackathon — Problem Statement 26167**, under the theme *Space Technology*.

SatQuery AI lets a user ask plain-English questions about a satellite image and get an
evidence-grounded answer — with confidence scores, supporting visuals, scene context, and a
downloadable report — instead of a single opaque prediction.

---

## ✨ Features

The app is organized into five tabs, each covering one of the problem statement's required
capabilities:

### 💬 Ask a Question (single-image VQA)
- Upload a co-registered Sentinel‑1 (SAR) + Sentinel‑2 (optical) patch.
- Pick from a set of PS-aligned reference questions, or write a custom query.
- Returns a plain-language answer, a confidence score, and a top‑5 candidate answer chart.
- Shows the full confidence distribution across the answer vocabulary, so a genuinely uncertain
  prediction is visible rather than hidden behind one number.
- Low-confidence answers are explicitly flagged as uncertain rather than stated as fact.
- Surfaces independent **scene-context evidence** from a BEN‑19 land-cover classifier alongside
  the VQA answer.
- Renders four Earth-observation views side by side: true-color RGB, color-infrared (vegetation),
  SAR radar composite, and an NDVI heatmap.
- Includes a sanity check on uploaded bands (flags NaNs or suspiciously constant/placeholder data)
  before running inference.
- Every run produces an **execution summary / audit trail** (task type, model backend, input
  modality, vocabulary size, warnings) and a **downloadable PDF report**.

### 📂 Batch Analysis
- Point the app at a folder of patch subfolders (or a single patch) and run one question across
  all of them.
- Progress bar over the batch; results shown in a table and exportable as CSV.
- Flags the case where every patch returns the identical answer — a signal the model isn't
  responding to image content and needs investigating before trusting the results.

### 🔄 Change Detection (bi-temporal)
- Upload Sentinel‑2 optical bands from two dates of the same area.
- Computes NDVI divergence between the two dates and visualizes it (red = vegetation loss,
  blue = vegetation gain), alongside the true-color image for each date.
- Reports the percentage of the scene showing significant vegetation loss/gain.
- Diffs the BEN‑19 land-cover classes detected at each date to show which classes appeared or
  disappeared between the two time points.

### 🎯 Grounding
- Upload a Sentinel‑2 patch, detect which BEN‑19 land-cover classes are present, and pick one to
  visually localize.
- Produces a Grad-CAM heatmap overlaid on the true-color image, showing *where* the selected class
  was detected rather than just naming it — satisfying the PS's grounding requirement.
- Degrades gracefully: if the scene classifier isn't available in the environment, the tab
  explains why and what to install rather than crashing.

### ℹ️ About
- Summarizes the model architecture, sensor modality, and full capability list for anyone opening
  the app cold (e.g. judges).

---

## 🧠 How it works

| Component | Details |
|---|---|
| **Vision-Language model** | [`ConfigILM`](https://github.com/lhackel-tub/ConfigILM) joining a ResNet‑50 vision encoder and a BERT-tiny text encoder into one VQA classification model |
| **Vision backbone pretraining** | Initialized from BIFOLD's pretrained BigEarthNet v2.0 weights |
| **Sensor fusion** | Optical–SAR early fusion: 2 Sentinel‑1 channels (VV, VH) + 10 Sentinel‑2 bands (B02–B12 minus B01) = 12 input channels |
| **Scene classifier** | Pretrained BEN‑19 land-cover classifier (`reben_publication`), used for scene context and to power grounding |
| **Grounding** | Grad-CAM over the scene classifier's final conv layer |
| **Answer format** | Fixed-vocabulary multiple choice (softmax over the answer vocabulary), not free-form text |
| **UI** | Streamlit, with a custom glassmorphism theme over a space-themed background |
| **Reporting** | `reportlab`-generated PDF per analysis run, plus CSV export for batch runs |

This satisfies the PS's core requirements: single-image VQA, optical–SAR fusion, bi-temporal
change detection, and grounding — delivered through one agentic, task-routing interface rather
than a single general-purpose model.

---

## 📁 Project structure

```
satquery_app/
├── app_v5.py                    # Final Streamlit app (this README describes this version)
├── agent.py                     # Query-routing / agent logic
├── answer_vocab.json            # VQA answer vocabulary (class ID ↔ answer text)
├── satquery_vqa_finetuned.pt    # Fine-tuned VQA model checkpoint
├── patch_image_cache.pkl        # Cached real Sentinel-1/2 patches (by patch_id)
├── BigEarthNet-VQA.parquet      # Training dataset (question, answer, patch_id, metadata)
├── download.py                  # Fetches real imagery via Google Earth Engine
├── extract_real_patch.py        # Extracts a single real patch for testing/demoing
├── infer_real.py                # Standalone inference script (non-UI)
├── predict.py                   # Prediction helper utilities
├── test_patches/                # Sample patches for the Batch Analysis tab
├── reben-training-scripts/      # Supporting training scripts from BIFOLD/reben_publication
├── SatQuery_AI_FINAL_colab.ipynb  # Training notebook (Colab)
├── thumb-1920-807192.jpg        # App background image
└── requirements.txt
```

---

## 🚀 Running the app

```bash
pip install -r requirements.txt
# reben_publication (needed for scene classification + grounding):
pip install --no-deps reben-training-scripts

streamlit run app_v5.py
```

**Required files in the working directory:** `answer_vocab.json`, `satquery_vqa_finetuned.pt`,
and `thumb-1920-807192.jpg` (the app falls back to a default theme if the background image is
missing, and shows raw class IDs if the vocabulary file is missing).

**Input format:** Sentinel‑1 and Sentinel‑2 bands as individual GeoTIFF (`.tif`/`.tiff`) files,
named or containing the band code (e.g. `B04.tif`, `patch_VV.tif`).

---

## 🎯 Alignment with PS 26167 requirements

| Requirement | Where it's implemented |
|---|---|
| Answer questions about a single satellite image | Ask a Question tab |
| Detect changes between two images over time | Change Detection tab |
| Combine optical + radar (fusion) | 12-channel fusion tensor used by the VQA model |
| Grounding / visual evidence | Grad-CAM heatmaps in the Grounding tab |
| Confidence information | Per-answer confidence + full distribution chart |
| Execution summaries | Audit-trail JSON shown after every VQA run |
| Downloadable reports | PDF export (single run) and CSV export (batch) |

---


---

## 🛣️ Status

The interactive application (all five tabs) is built and functional end-to-end. Model prediction
quality is the current focus area — see the team lead for the latest training status before a
live demo.
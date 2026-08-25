import torch
import numpy as np
import os
import rasterio
from configilm.ConfigILM import ILMConfiguration, ConfigILM, ILMType
from configilm.util import get_default_tokenizer
from configilm.extra.BENv2_utils import band_combi_to_mean_std

# 1. Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running inference on: {device}")

# 2. Band Definitions (Matching BigEarthNet v2.0 / reBEN standard)
S2_BANDS_V020 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]

# Load standard normalization parameters
s2_mean, s2_std = band_combi_to_mean_std(10)
fusion_mean, fusion_std = band_combi_to_mean_std(12) # 2 SAR + 10 Optical bands
fusion_mean, fusion_std = np.array(fusion_mean), np.array(fusion_std)

def load_real_fusion_patch(s1_file_map: dict, s2_file_map: dict) -> torch.Tensor:
    """Reads real GeoTIFF bands via rasterio, normalizes them, and stacks into a 12-channel tensor."""
    s1_arrays = []
    for b in S1_BANDS:
        with rasterio.open(s1_file_map[b]) as src:
            s1_arrays.append(src.read(1).astype(np.float32))
            
    s2_arrays = []
    for b in S2_BANDS_V020:
        with rasterio.open(s2_file_map[b]) as src:
            s2_arrays.append(src.read(1).astype(np.float32))
            
    s1 = np.stack(s1_arrays, axis=0)
    s2 = np.stack(s2_arrays, axis=0)
    
    # Concatenate SAR and Optical into 12 channels
    stacked = np.concatenate([s1, s2], axis=0)
    
    # Apply standard normalization
    stacked = (stacked - fusion_mean[:, None, None]) / fusion_std[:, None, None]
    
    # Add batch dimension -> shape: [1, 12, H, W]
    return torch.from_numpy(stacked).unsqueeze(0).float()

# 3. Rebuild model & load your downloaded weights
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
model.load_state_dict(torch.load("satquery_vqa_finetuned.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()

# 4. Define paths to your local sample GeoTIFFs
# (Update these paths to point to a real test patch folder on your machine)
PATCH_DIR = "test_patch"

sample_s1_paths = {
    "VV": os.path.join(PATCH_DIR, "VV.tif"),
    "VH": os.path.join(PATCH_DIR, "VH.tif")
}

sample_s2_paths = {b: os.path.join(PATCH_DIR, f"{b}.tif") for b in S2_BANDS_V020}

# 5. Run prediction with a real question
tokenizer = get_default_tokenizer()
question = "Is there arable land in this image?"
q_ids = torch.tensor([tokenizer.encode(question, max_length=32, padding="max_length", truncation=True)]).to(device)

try:
    # Load real image tensors
    real_img_tensor = load_real_fusion_patch(sample_s1_paths, sample_s2_paths).to(device)
    
    with torch.no_grad():
        logits = model((real_img_tensor, q_ids))
        predicted_idx = torch.argmax(logits, dim=-1).item()
        
    print(f"\nQuestion: '{question}'")
    print(f"Predicted Class ID from Real GeoTIFF: {predicted_idx}")
    
except Exception as e:
    print(f"\n[Note] Real file paths need to be updated: {e}")
    print("Once you point the dictionary to your local .tif files, this script will process real satellite pixels!")
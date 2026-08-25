import torch
from configilm.ConfigILM import ILMConfiguration, ConfigILM, ILMType
from configilm.util import get_default_tokenizer

# 1. Setup device to utilize your local GPU (GTX 1650 Ti)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# 2. Rebuild the exact model architecture matching your Colab training
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

print("Initializing model architecture...")
model = ConfigILM(vqa_config)

# 3. Load your fine-tuned weights
print("Loading fine-tuned weights from satquery_vqa_finetuned.pt...")
model.load_state_dict(torch.load("satquery_vqa_finetuned.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()
print("Model weights loaded successfully!")

# 4. Test forward pass with a dummy input
tokenizer = get_default_tokenizer()
question = "Is there urban fabric in this image?"
q_ids = torch.tensor([tokenizer.encode(question, max_length=32, padding="max_length", truncation=True)]).to(device)

# Simulating a 12-channel GeoTIFF tensor (Sentinel-1 SAR + Sentinel-2 Optical)
dummy_img = torch.rand(1, 12, 120, 120).to(device)

with torch.no_grad():
    logits = model((dummy_img, q_ids))
    predicted_idx = torch.argmax(logits, dim=-1).item()

print(f"\nTest Question: '{question}'")
print(f"Model Output Shape: {logits.shape}")
print(f"Predicted Answer Class ID: {predicted_idx}")
print("🎉 Local virtual environment test passed successfully!")
import os
import cv2
import ee
import numpy as np
import pandas as pd
import rasterio

# 1. Initialize Google Earth Engine
GEE_PROJECT_ID = "gen-lang-client-0849351832"

try:
    ee.Initialize(project=GEE_PROJECT_ID)
    print("Google Earth Engine initialized successfully.")
except Exception:
    print("Authenticating Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT_ID)

# 2. Load Local Parquet File
parquet_path = "BigEarthNet-VQA.parquet"
if not os.path.exists(parquet_path):
    raise FileNotFoundError(
        f"Could not find '{parquet_path}' in the current directory. "
        "Make sure the parquet file is in your satquery_app folder."
    )

print("Loading Parquet dataset...")
df = pd.read_parquet(parquet_path)

# Sample 15 unique coordinates
unique_patches = df[["patch_id", "latitude", "longitude", "season"]].drop_duplicates("patch_id").sample(n=15, random_state=42)

# 3. Define Sensor Bands
S1_BANDS = ["VV", "VH"]
S2_BANDS_GEE = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

def fetch_and_save_patch(lat: float, lon: float, season: str, patch_folder: str, target_size: int = 120):
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(1200 / 2).bounds()
    season_months = {"Winter": (12, 2), "Spring": (3, 5), "Summer": (6, 8), "Autumn": (9, 11), "Fall": (9, 11)}
    start_m, end_m = season_months.get(season, (6, 8))

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filter(ee.Filter.calendarRange(start_m, end_m, "month"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .first()
    )

    try:
        # Optical Bands (Sentinel-2)
        s2_sample = s2.select(S2_BANDS_GEE).sampleRectangle(region=region, defaultValue=0).getInfo()
        for b in S2_BANDS_GEE:
            arr = np.array(s2_sample["properties"][b], dtype=np.float32)
            if arr.shape != (target_size, target_size):
                arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
            # Map single digit names to two digits for Streamlit matching (e.g., B2 -> B02)
            band_filename = b.replace("B", "B0") if b in ["B2", "B3", "B4", "B5", "B6", "B7", "B8"] else b
            
            out_file = os.path.join(patch_folder, f"{band_filename}.tif")
            with rasterio.open(out_file, 'w', driver='GTiff', height=target_size, width=target_size, count=1, dtype='float32') as dst:
                dst.write(arr, 1)

        # Radar Bands (Sentinel-1)
        s1_sample = s1.select(S1_BANDS).sampleRectangle(region=region, defaultValue=0).getInfo()
        for b in S1_BANDS:
            arr = np.array(s1_sample["properties"][b], dtype=np.float32)
            if arr.shape != (target_size, target_size):
                arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
            out_file = os.path.join(patch_folder, f"{b}.tif")
            with rasterio.open(out_file, 'w', driver='GTiff', height=target_size, width=target_size, count=1, dtype='float32') as dst:
                dst.write(arr, 1)

        return True
    except Exception as e:
        print(f"Skipping patch at ({lat}, {lon}): {e}")
        return False

# 4. Execute Downloads
output_base = "test_patches"
os.makedirs(output_base, exist_ok=True)

print(f"Downloading 15 patches from Earth Engine into ./{output_base}/ ...")

successful_count = 0
for i, row in enumerate(unique_patches.itertuples()):
    folder_path = os.path.join(output_base, f"patch_{i+1:02d}")
    os.makedirs(folder_path, exist_ok=True)
    
    if fetch_and_save_patch(row.latitude, row.longitude, row.season, folder_path):
        successful_count += 1
        print(f"Saved patch_{i+1:02d} ({successful_count}/15)")

print(f"\nCompleted! Saved {successful_count} patches into ./{output_base}/")
import rasterio

# Pick the first cached patch
sample_pid = list(patch_cache.keys())[0]
sample_patch = patch_cache[sample_pid]  # shape (14, 120, 120)

band_names = ["VV", "VH", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

# Export each band as a real GeoTIFF file
for i, b in enumerate(band_names):
    filename = f"{b}.tif"
    with rasterio.open(
        filename, 'w',
        driver='GTiff', height=120, width=120, count=1,
        dtype=sample_patch[i].dtype
    ) as dst:
        dst.write(sample_patch[i], 1)

from google.colab import files
for b in ["VV", "VH", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]:
    # Rename B2 -> B02 for Streamlit uploader matching
    orig_b = b.replace("B0", "B") if b.startswith("B0") else b
    files.download(f"{orig_b}.tif")
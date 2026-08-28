import ee

# Initialize Earth Engine (Run ee.Authenticate() first time)
ee.Initialize()

# Define your exact bounding box or coordinate point (Longitude, Latitude)
# Example: Prayagraj region
point = ee.Geometry.Point([81.8463, 25.4358])
roi = point.buffer(2000).bounds() # 2km square patch

# 1. Fetch 1985 imagery (Landsat 5)
l5_1985 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
           .filterBounds(roi)
           .filterDate('1985-01-01', '1985-12-31')
           .sort('CLOUD_COVER')
           .first())

# 2. Fetch 2024 imagery (Landsat 9)
l9_2024 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
           .filterBounds(roi)
           .filterDate('2024-01-01', '2024-12-31')
           .sort('CLOUD_COVER')
           .first())

# Generate download URLs for the exact same patch
url_1985 = l5_1985.getDownloadURL({'region': roi, 'scale': 30, 'format': 'GEO_TIFF'})
url_2024 = l9_2024.getDownloadURL({'region': roi, 'scale': 30, 'format': 'GEO_TIFF'})

print(f"Download 1985 Patch: {url_1985}")
print(f"Download 2024 Patch: {url_2024}")
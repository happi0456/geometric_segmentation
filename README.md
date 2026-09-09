Note: I couldn't actually test this end-to-end because installing py3dtiles threw a C++/CMake build error on my machine. Hopefully the code works out-of-the-box! Also, please note that test_standalone.py doesn't exist in the repo—use test_direct.py for testing instead.
--- 
2D floor plan polygons (GeoJSON) and floor height ranges (JSON) into watertight 3D volumetric building models in OGC 3D Tiles format (.b3dm)

Install dependencies:
```
pip install -r requirements.txt
```
Add your data: Drop your floor plans into data/raw/floor_plans.geojson and your elevation config into data/raw/elevations.json.

Run the pipeline:
```
from pipeline_logic import run_pipeline_from_files
b3dm_path = run_pipeline_from_files(
    floor_plans_path="data/raw/floor_plans.geojson",
    elevations_path="data/raw/elevations.json",
    building_id="BLD001"
)
```
Run quick test:
```
python test_direct.py
```

Core Files
geometric_segmentation.py: Core geometry engine (Box3D, GeometricSegmentation, B3DMTileBuilder).
pipeline_logic.py: End-to-end orchestration (loads data, validates overlaps, exports tiles).
test_direct.py: Quick verification test using synthetic data.
data/raw/: Input folder for your GeoJSON and JSON files.
data/processed/: Output folder for generated .b3dm files
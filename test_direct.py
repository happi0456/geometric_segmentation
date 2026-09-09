"""Direct test without imports"""
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create test data
test_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"floor_number": 1, "unit_id": "U1"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
            }
        }
    ]
}

test_elevations = {
    "building_id": "TEST_BLD",
    "ground_level": 0.0,
    "floors": [
        {"floor_number": 1, "bottom": 0.0, "top": 3.0}
    ]
}

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Save test data
with open("data/raw/test_floor_plans.geojson", "w") as f:
    json.dump(test_geojson, f)

with open("data/raw/test_elevations.json", "w") as f:
    json.dump(test_elevations, f)

print("✓ Test data created")

# Now test the import chain step by step
try:
    print("\n1. Testing geometric_segmentation import...")
    from geometric_segmentation import GeometricSegmentation, FloorConfig, B3DMTileBuilder
    print("   ✓ geometric_segmentation imported")
    
    print("\n2. Testing pipeline_logic import...")
    from pipeline_logic import SegmentationPipeline
    print("   ✓ pipeline_logic imported")
    
    print("\n3. Running pipeline...")
    pipeline = SegmentationPipeline(building_id="TEST_BLD")
    b3dm_path = pipeline.run(
        "data/raw/test_floor_plans.geojson",
        "data/raw/test_elevations.json"
    )
    
    if b3dm_path:
        print(f"\n✓ SUCCESS: B3DM at {b3dm_path}")
        import os
        size = os.path.getsize(b3dm_path)
        print(f"✓ File size: {size} bytes")
    else:
        print("\n✗ Pipeline returned None")
        
except ImportError as e:
    print(f"\n✗ IMPORT ERROR: {e}")
    print("\nDebug: List files in current directory:")
    import os
    for f in os.listdir('.'):
        if f.endswith('.py'):
            print(f"  - {f}")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
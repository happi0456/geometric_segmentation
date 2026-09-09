"""
Geometric Segmentation: Convert 2D floor plans + height data into floor/flat 3D volumes
Output: .b3dm tiles for streaming
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
import pyproj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FloorConfig:
    """Configuration for a single floor level"""
    floor_number: int
    bottom_elevation: float
    top_elevation: float
    unit_polygons: List[Polygon]  # 2D polygons from floor plan


@dataclass
class FloorUnit:
    """3D volumetric representation of a single apartment/unit"""
    unit_id: str
    floor_number: int
    geometry: 'Box3D'  # 3D box geometry
    centroid: Tuple[float, float, float]
    area_2d: float
    volume: float


class Box3D:
    """Watertight 3D box representation of a unit"""
    
    def __init__(self, polygon_2d: Polygon, bottom_z: float, top_z: float):
        """
        Create a closed 3D box from 2D polygon and Z range
        
        Args:
            polygon_2d: Shapely Polygon (2D footprint)
            bottom_z: Ground elevation (m)
            top_z: Ceiling elevation (m)
        """
        self.polygon_2d = polygon_2d
        self.bottom_z = bottom_z
        self.top_z = top_z
        self.height = top_z - bottom_z
        
    def get_vertices(self) -> np.ndarray:
        """
        Return 3D vertices for the box: bottom ring + top ring
        Shape: (n_vertices, 3)
        """
        exterior_coords = np.array(self.polygon_2d.exterior.coords[:-1])  # Exclude closing point
        n = len(exterior_coords)
        
        # Bottom vertices (Z = bottom_z)
        bottom_verts = np.column_stack([exterior_coords, np.full(n, self.bottom_z)])
        
        # Top vertices (Z = top_z)
        top_verts = np.column_stack([exterior_coords, np.full(n, self.top_z)])
        
        return np.vstack([bottom_verts, top_verts])
    
    def get_faces(self) -> np.ndarray:
        """
        Return face indices as triangles (triangulated from vertices)
        Each face is [v0, v1, v2]
        """
        exterior_coords = np.array(self.polygon_2d.exterior.coords[:-1])
        n = len(exterior_coords)
        faces = []
        
        # Side faces (vertical walls)
        for i in range(n):
            next_i = (i + 1) % n
            # Two triangles per side quad
            faces.append([i, next_i, n + i])
            faces.append([next_i, n + next_i, n + i])
        
        # Bottom and top faces (using triangulation for robustness)
        bottom_triangles = self._triangulate_polygon(self.polygon_2d, z=self.bottom_z, reverse=True)
        top_triangles = self._triangulate_polygon(self.polygon_2d, z=self.top_z, reverse=False)
        
        if bottom_triangles is not None:
            faces.extend(bottom_triangles)
        if top_triangles is not None:
            faces.extend(top_triangles)
        
        return np.array(faces, dtype=np.uint32)
    
    def _triangulate_polygon(self, poly: Polygon, z: float, reverse: bool = False) -> Optional[List[List[int]]]:
        """Fan-triangulate a cap face using indices into get_vertices()."""
        try:
            exterior_coords = np.array(poly.exterior.coords[:-1])
            n = len(exterior_coords)
            n_base = len(np.array(self.polygon_2d.exterior.coords[:-1]))

            # Vertex buffer is [bottom ring | top ring]; indices must stay in-range.
            if n < 3 or n != n_base:
                logger.warning(
                    "Triangulation skipped: vertex count mismatch "
                    f"(cap={n}, base={n_base})"
                )
                return None

            # Bottom faces use 0..n-1; top faces use n..2n-1.
            offset = n_base if np.isclose(z, self.top_z) else 0

            triangles = []
            for i in range(1, n - 1):
                v0_idx = offset
                v1_idx = offset + i
                v2_idx = offset + i + 1

                if reverse:
                    triangles.append([v2_idx, v1_idx, v0_idx])
                else:
                    triangles.append([v0_idx, v1_idx, v2_idx])

            return triangles
        except Exception as e:
            logger.warning(f"Triangulation failed: {e}")
            return None
    
    def get_volume(self) -> float:
        """Calculate volume of the box"""
        return self.polygon_2d.area * self.height


class GeometricSegmentation:
    """
    Main pipeline: segment building into floor/flat units
    """
    
    def __init__(self, crs: str = "EPSG:32643"):  # UTM Zone 43N for India
        """
        Args:
            crs: Coordinate reference system (default: India UTM Zone 43N)
        """
        self.crs = crs
        self.transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self.units: List[FloorUnit] = []
    
    def segment_from_floor_configs(self, 
                                   floor_configs: List[FloorConfig],
                                   building_id: str) -> List[FloorUnit]:
        """
        Main entry point: convert floor configs into 3D units
        
        Args:
            floor_configs: List of FloorConfig objects (from Step 3: floor plan extraction)
            building_id: Unique building identifier
            
        Returns:
            List of FloorUnit objects with 3D geometry
        """
        self.units = []
        
        for floor_cfg in floor_configs:
            logger.info(f"Segmenting floor {floor_cfg.floor_number} "
                       f"(z: {floor_cfg.bottom_elevation} - {floor_cfg.top_elevation}m)")
            
            for unit_idx, unit_poly in enumerate(floor_cfg.unit_polygons):
                try:
                    # Validate polygon
                    if not unit_poly.is_valid:
                        unit_poly = unit_poly.buffer(0)  # Fix self-intersections
                    
                    if unit_poly.area < 5:  # Skip tiny polygons (< 5 m²)
                        logger.warning(f"Skipping floor {floor_cfg.floor_number} unit {unit_idx}: area={unit_poly.area:.2f}m²")
                        continue
                    
                    # Create 3D box
                    box_3d = Box3D(unit_poly, 
                                  floor_cfg.bottom_elevation,
                                  floor_cfg.top_elevation)
                    
                    # Create unit record
                    unit_id = f"{building_id}_F{floor_cfg.floor_number:02d}_U{unit_idx:03d}"
                    centroid_2d = unit_poly.centroid
                    centroid_3d = (centroid_2d.x, centroid_2d.y, 
                                  (floor_cfg.bottom_elevation + floor_cfg.top_elevation) / 2)
                    
                    unit = FloorUnit(
                        unit_id=unit_id,
                        floor_number=floor_cfg.floor_number,
                        geometry=box_3d,
                        centroid=centroid_3d,
                        area_2d=unit_poly.area,
                        volume=box_3d.get_volume()
                    )
                    
                    self.units.append(unit)
                    logger.debug(f"Created unit {unit_id}: volume={unit.volume:.2f}m³")
                
                except Exception as e:
                    logger.error(f"Failed to segment floor {floor_cfg.floor_number} unit {unit_idx}: {e}")
        
        logger.info(f"Segmentation complete: {len(self.units)} units created")
        return self.units
    
    def validate_no_overlaps(self) -> Tuple[bool, List[str]]:
        """
        Check for unit overlaps (3D intersection)
        Returns: (is_valid, error_messages)
        """
        errors = []
        
        for i, unit1 in enumerate(self.units):
            for unit2 in self.units[i+1:]:
                # Quick check: same floor = no vertical overlap possible
                if unit1.floor_number == unit2.floor_number:
                    # 2D intersection check
                    if unit1.geometry.polygon_2d.intersects(unit2.geometry.polygon_2d):
                        overlap_area = unit1.geometry.polygon_2d.intersection(
                            unit2.geometry.polygon_2d
                        ).area
                        if overlap_area > 0.01:  # Tolerance for numerical errors
                            errors.append(
                                f"{unit1.unit_id} overlaps {unit2.unit_id} "
                                f"(area={overlap_area:.2f}m²)"
                            )
        
        return len(errors) == 0, errors
    
    def export_units_to_dict(self) -> List[Dict]:
        """Convert units to serializable format for B3DM conversion"""
        return [
            {
                'unit_id': unit.unit_id,
                'floor_number': unit.floor_number,
                'vertices': unit.geometry.get_vertices().tolist(),
                'faces': unit.geometry.get_faces().tolist(),
                'centroid': unit.centroid,
                'volume': unit.volume,
                'area_2d': unit.area_2d,
            }
            for unit in self.units
        ]


class B3DMTileBuilder:
    """
    Convert segmented units into OGC 3D Tiles (.b3dm format)
    """
    
    def __init__(self):
        try:
            import trimesh
            self.trimesh = trimesh
        except ImportError:
            raise ImportError("trimesh required: pip install trimesh")
    
    def units_to_meshes(self, units: List[FloorUnit]) -> Dict[str, 'trimesh.Trimesh']:
        """Convert FloorUnit objects to trimesh Mesh objects"""
        meshes = {}
        
        for unit in units:
            try:
                vertices = unit.geometry.get_vertices()
                faces = unit.geometry.get_faces()
                
                mesh = self.trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
                meshes[unit.unit_id] = mesh
            except Exception as e:
                logger.error(f"Failed to create mesh for {unit.unit_id}: {e}")
        
        return meshes
    
    def export_gltf(self, unit: FloorUnit, output_path: str):
        """Export single unit as glTF (intermediate before B3DM)"""
        try:
            vertices = unit.geometry.get_vertices()
            faces = unit.geometry.get_faces()
            
            mesh = self.trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            mesh.export(output_path)
            logger.info(f"Exported {unit.unit_id} to {output_path}")
        except Exception as e:
            logger.error(f"glTF export failed for {unit.unit_id}: {e}")
    
    def create_b3dm_tile(self, units: List[FloorUnit], output_path: str):
        """
        Create single B3DM tile from multiple units using custom byte packing.
        Also writes a sidecar metadata JSON next to the tile.
        """
        import struct
        import json
        from pathlib import Path
        
        # Merge all unit meshes
        combined_vertices = []
        combined_faces = []
        vertex_offset = 0
        
        feature_table_json = {
            'BATCH_LENGTH': len(units),
            'BATCH_ID': []
        }
        
        for idx, unit in enumerate(units):
            vertices = unit.geometry.get_vertices()
            faces = unit.geometry.get_faces()
            
            combined_vertices.extend(vertices)
            combined_faces.extend(faces + vertex_offset)
            vertex_offset += len(vertices)
            
            feature_table_json['BATCH_ID'].append(unit.unit_id)
        
        combined_vertices = np.array(combined_vertices, dtype=np.float32).flatten()
        combined_faces = np.array(combined_faces, dtype=np.uint32).flatten()
        
        # Build feature table
        feature_table_bytes = json.dumps(feature_table_json).encode('utf-8')
        # Pad to 8-byte alignment
        padding = (8 - (len(feature_table_bytes) % 8)) % 8
        feature_table_bytes += b' ' * padding
        
        # B3DM header
        magic = b'b3dm'
        version = 1
        
        # Binary data (GLB format)
        glb_json = {
            'asset': {'version': '2.0'},
            'scene': 0,
            'scenes': [{'nodes': [0]}],
            'nodes': [{'mesh': 0}],
            'meshes': [{
                'primitives': [{
                    'attributes': {'POSITION': 0},
                    'indices': 1,
                    'mode': 4  # Triangles
                }]
            }],
            'bufferViews': [
                {'buffer': 0, 'byteOffset': 0, 'byteLength': len(combined_vertices) * 4},
                {'buffer': 0, 'byteOffset': len(combined_vertices) * 4, 'byteLength': len(combined_faces) * 4}
            ],
            'accessors': [
                {'bufferView': 0, 'componentType': 5126, 'count': len(combined_vertices) // 3, 'type': 'VEC3'},
                {'bufferView': 1, 'componentType': 5125, 'count': len(combined_faces), 'type': 'SCALAR'}
            ],
            'buffers': [{'byteLength': len(combined_vertices) * 4 + len(combined_faces) * 4}]
        }
        
        glb_json_bytes = json.dumps(glb_json).encode('utf-8')
        padding_json = (4 - (len(glb_json_bytes) % 4)) % 4
        glb_json_bytes += b' ' * padding_json
        
        # Write B3DM file
        with open(output_path, 'wb') as f:
            # Header
            f.write(magic)
            f.write(struct.pack('<I', version))
            
            # Byte lengths
            feature_table_length = len(feature_table_bytes)
            batch_table_length = 0
            binary_body_length = len(glb_json_bytes) + 4 + 4 + len(combined_vertices) * 4 + len(combined_faces) * 4
            file_length = 28 + feature_table_length + batch_table_length + binary_body_length
            
            f.write(struct.pack('<I', file_length))
            f.write(struct.pack('<I', feature_table_length))
            f.write(struct.pack('<I', batch_table_length))
            
            # Feature table
            f.write(feature_table_bytes)
            
            # Binary body (simplified GLB)
            f.write(struct.pack('<4s', b'glTF'))
            f.write(struct.pack('<I', 2))
            f.write(struct.pack('<I', len(glb_json_bytes) + 4 + 4))
            f.write(glb_json_bytes)
            f.write(combined_vertices.tobytes())
            f.write(combined_faces.tobytes())
        
        logger.info(f"B3DM tile created: {output_path} ({len(units)} units)")

        metadata_path = str(Path(output_path).with_suffix(".json"))
        if metadata_path == output_path:
            metadata_path = output_path + ".json"
        with open(metadata_path, "w", encoding="utf-8") as meta_f:
            json.dump(
                {
                    "output": output_path,
                    "unit_count": len(units),
                    "units": [
                        {
                            "unit_id": unit.unit_id,
                            "floor_number": unit.floor_number,
                            "centroid": list(unit.centroid),
                            "volume": unit.volume,
                            "area_2d": unit.area_2d,
                        }
                        for unit in units
                    ],
                },
                meta_f,
                indent=2,
            )
        logger.info(f"Metadata written: {metadata_path}")


def example_workflow():
    """Example: Load floor configs and generate B3DM"""
    
    # Simulate floor plan extraction from Step 3
    floor_configs = [
        FloorConfig(
            floor_number=4,
            bottom_elevation=9.0,
            top_elevation=12.0,
            unit_polygons=[
                Polygon([(0, 0), (5, 0), (5, 4), (0, 4)]),  # Unit 401
                Polygon([(5.5, 0), (10, 0), (10, 4), (5.5, 4)]),  # Unit 402
            ]
        ),
        FloorConfig(
            floor_number=5,
            bottom_elevation=12.0,
            top_elevation=15.0,
            unit_polygons=[
                Polygon([(0, 0), (5, 0), (5, 4), (0, 4)]),  # Unit 501
                Polygon([(5.5, 0), (10, 0), (10, 4), (5.5, 4)]),  # Unit 502
            ]
        )
    ]
    
    # Segment
    segmenter = GeometricSegmentation()
    units = segmenter.segment_from_floor_configs(floor_configs, building_id="BLD001")
    
    # Validate
    is_valid, errors = segmenter.validate_no_overlaps()
    if not is_valid:
        for err in errors:
            logger.error(err)
        return
    
    # Export to B3DM
    builder = B3DMTileBuilder()
    builder.create_b3dm_tile(units, "data/processed/building_001.b3dm")


if __name__ == "__main__":
    example_workflow()
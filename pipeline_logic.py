"""
End-to-end segmentation pipeline: floor plans + elevations -> B3DM tile.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from shapely.geometry import Polygon, shape

from geometric_segmentation import (
    B3DMTileBuilder,
    FloorConfig,
    GeometricSegmentation,
)

logger = logging.getLogger(__name__)


def _as_polygon(geom) -> Optional[Polygon]:
    if geom is None:
        return None
    if isinstance(geom, Polygon):
        return geom
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return None


def _floor_number(properties: Dict) -> Optional[int]:
    for key in ("floor_number", "floor_level", "floor"):
        if key in properties and properties[key] is not None:
            return int(properties[key])
    return None


def _elevation_lookup(elevations: Dict) -> Dict[int, Dict[str, float]]:
    lookup: Dict[int, Dict[str, float]] = {}

    floors = elevations.get("floors") or []
    for floor in floors:
        num = floor.get("floor_number", floor.get("floor_level"))
        if num is None:
            continue
        lookup[int(num)] = {
            "bottom": float(floor.get("bottom", floor.get("z_min_relative_m", 0.0))),
            "top": float(floor.get("top", floor.get("z_max_relative_m", 3.0))),
        }

    for layer in elevations.get("vertical_layers") or []:
        num = layer.get("floor_level", layer.get("floor_number"))
        if num is None:
            continue
        num = int(num)
        if num in lookup:
            continue
        lookup[num] = {
            "bottom": float(layer.get("z_min_relative_m", layer.get("bottom", 0.0))),
            "top": float(layer.get("z_max_relative_m", layer.get("top", 3.0))),
        }

    return lookup


class SegmentationPipeline:
    """Load inputs, segment units, and export a B3DM tile."""

    def __init__(self, building_id: str, output_dir: Union[str, Path] = "data/processed"):
        self.building_id = building_id
        self.output_dir = Path(output_dir)
        self.segmenter = GeometricSegmentation()
        self.builder = B3DMTileBuilder()

    def run(self, floor_plans_path: str, elevations_path: str) -> Optional[str]:
        floor_configs = self._load_floor_configs(floor_plans_path, elevations_path)
        if not floor_configs:
            logger.error("No floor configs could be built from input files")
            return None

        units = self.segmenter.segment_from_floor_configs(floor_configs, self.building_id)
        if not units:
            logger.error("Segmentation produced no units")
            return None

        is_valid, errors = self.segmenter.validate_no_overlaps()
        if not is_valid:
            for err in errors:
                logger.error(err)
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        b3dm_path = self.output_dir / f"{self.building_id}.b3dm"
        self.builder.create_b3dm_tile(units, str(b3dm_path))
        return str(b3dm_path)

    def _load_floor_configs(self, floor_plans_path: str, elevations_path: str) -> List[FloorConfig]:
        with open(floor_plans_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
        with open(elevations_path, "r", encoding="utf-8") as f:
            elevations = json.load(f)

        elev_by_floor = _elevation_lookup(elevations)
        polygons_by_floor: Dict[int, List[Polygon]] = defaultdict(list)

        for feature in geojson.get("features", []):
            props = feature.get("properties") or {}
            floor_num = _floor_number(props)
            if floor_num is None:
                logger.warning("Skipping feature with no floor number")
                continue

            geom = _as_polygon(shape(feature["geometry"]))
            if geom is None or geom.is_empty:
                logger.warning(f"Skipping feature on floor {floor_num}: invalid geometry")
                continue

            polygons_by_floor[floor_num].append(geom)

        configs: List[FloorConfig] = []
        for floor_num, polygons in sorted(polygons_by_floor.items()):
            elev = elev_by_floor.get(floor_num)
            if elev is None:
                logger.warning(f"No elevation data for floor {floor_num}; using 0-3m")
                elev = {"bottom": 0.0, "top": 3.0}
            configs.append(
                FloorConfig(
                    floor_number=floor_num,
                    bottom_elevation=elev["bottom"],
                    top_elevation=elev["top"],
                    unit_polygons=polygons,
                )
            )
        return configs


def run_pipeline_from_files(
    floor_plans_path: str,
    elevations_path: str,
    building_id: str = "BUILDING",
    output_dir: Union[str, Path] = "data/processed",
) -> Optional[str]:
    return SegmentationPipeline(building_id=building_id, output_dir=output_dir).run(
        floor_plans_path, elevations_path
    )

    """Segmentation module for B3DM tile generation."""
    from geometric_segmentation import (
        GeometricSegmentation,
        B3DMTileBuilder,
        Box3D,
        FloorUnit,
        FloorConfig
    )
    from pipeline_logic import (
        SegmentationPipeline,
        run_pipeline_from_files
    )

    __all__ = [
        'GeometricSegmentation',
        'B3DMTileBuilder',
        'Box3D',
        'FloorUnit',
        'FloorConfig',
        'SegmentationPipeline',
        'run_pipeline_from_files'
    ]

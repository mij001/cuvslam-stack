"""Frame-source registry and factory.

Maps the ``[input].type`` string to a :class:`FrameSource` implementation.
Sources are imported lazily so that a missing optional dependency (e.g.
``pyrealsense2`` for the live camera) only fails if that source is requested.
"""

from __future__ import annotations

from .base import FrameEvent, FrameSource, ImuEvent

_REGISTRY = {
    "image_folder": ("cuvslam_runner.sources.image_folder", "ImageFolderSource"),
    "euroc": ("cuvslam_runner.sources.euroc", "EurocSource"),
    "tum": ("cuvslam_runner.sources.tum", "TumSource"),
    "edex": ("cuvslam_runner.sources.edex", "EdexSource"),
    "video": ("cuvslam_runner.sources.video", "VideoSource"),
    "realsense": ("cuvslam_runner.sources.realsense", "RealsenseSource"),
}


def available_types() -> list:
    return sorted(_REGISTRY)


def build_source(input_table: dict) -> FrameSource:
    """Instantiate the FrameSource named by ``input_table['type']``."""
    kind = input_table["type"]
    if kind not in _REGISTRY:
        raise ValueError(
            f"Unknown input.type {kind!r}. Available: {available_types()}"
        )
    module_name, class_name = _REGISTRY[kind]
    import importlib

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(input_table)


__all__ = ["FrameSource", "FrameEvent", "ImuEvent", "build_source", "available_types"]

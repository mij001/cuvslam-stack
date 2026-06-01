"""Image loading helpers shared by the file-based frame sources.

cuVSLAM accepts:
  * mono8  : uint8, shape (H, W)
  * rgb8   : uint8, shape (H, W, 3)
  * depth  : uint16, shape (H, W)

The stock examples convert color images RGB->BGR before handing them to the
tracker, so we do the same by default (toggle with ``bgr=False``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

_BAYER_CODES = {"BG", "GB", "RG", "GR"}


def _demosaic(raw: np.ndarray, pattern: str) -> np.ndarray:
    """Debayer a single-channel raw image into BGR using OpenCV."""
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Bayer demosaicing requires opencv-python (`pip install opencv-python`)."
        ) from exc
    code = pattern.upper()[:2]
    if code not in _BAYER_CODES:
        raise ValueError(f"Unsupported bayer pattern {pattern!r}; expected one of {_BAYER_CODES}")
    # OpenCV's naming is shifted relative to the common sensor naming, hence the map.
    conversion = {
        "BG": cv2.COLOR_BayerBG2BGR,
        "GB": cv2.COLOR_BayerGB2BGR,
        "RG": cv2.COLOR_BayerRG2BGR,
        "GR": cv2.COLOR_BayerGR2BGR,
    }[code]
    return cv2.cvtColor(raw, conversion)


def load_image(
    path: str,
    *,
    bgr: bool = True,
    bayer: Optional[str] = None,
) -> np.ndarray:
    """Load a color/grayscale image as a contiguous uint8 ndarray.

    Args:
        path:  image file path.
        bgr:   if True (default) convert 3-channel images RGB->BGR to match the
               cuVSLAM example pipeline.
        bayer: if set (e.g. "GBRG"), treat the file as a raw Bayer image and
               demosaic it to a 3-channel image (used by datasets like Oxford
               RobotCar).
    """
    pil = Image.open(path)
    frame = np.array(pil)

    if bayer:
        if frame.ndim != 2:
            raise ValueError(f"Bayer source {path} must be single-channel, got shape {frame.shape}")
        frame = _demosaic(frame, bayer)  # already BGR
        return np.ascontiguousarray(frame.astype(np.uint8, copy=False))

    if pil.mode == "L":
        if frame.ndim != 2:
            raise ValueError(f"Expected mono8 image with 2 dims [H W], got {frame.shape}")
    elif pil.mode in ("RGB", "RGBA"):
        frame = frame[:, :, :3]
        if bgr:
            frame = frame[:, :, ::-1]
    elif pil.mode in ("I;16", "I;16B", "I"):
        # A 16-bit single channel file opened as a color image is unusual; coerce
        # down to 8-bit grayscale rather than silently misinterpreting it.
        frame = frame.astype(np.uint8)
    else:
        # Fall back to grayscale conversion for any other mode.
        frame = np.array(pil.convert("L"))

    return np.ascontiguousarray(frame.astype(np.uint8, copy=False))


def load_depth(path: str, *, scale_to_uint16: float = 1.0) -> np.ndarray:
    """Load a depth image as a contiguous uint16 ndarray of shape (H, W)."""
    pil = Image.open(path)
    frame = np.array(pil)
    if frame.ndim != 2:
        raise ValueError(f"Expected single-channel depth image, got shape {frame.shape} for {path}")
    if scale_to_uint16 != 1.0:
        frame = frame.astype(np.float64) * scale_to_uint16
    frame = np.clip(frame, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return np.ascontiguousarray(frame)


def empty_image() -> np.ndarray:
    """A zero-size placeholder; cuVSLAM skips cameras with empty image data."""
    return np.empty((0, 0), dtype=np.uint8)

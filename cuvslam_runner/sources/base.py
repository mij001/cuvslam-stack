"""Frame-source abstractions.

A :class:`FrameSource` is an iterable of *events*. Each event is either a
:class:`FrameEvent` (a synchronized set of camera images, optionally with depth
and masks) or an :class:`ImuEvent` (a single IMU sample). Events must be yielded
in strictly ascending timestamp order, exactly as cuVSLAM requires for
``track`` / ``register_imu_measurement`` calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

from ..specs import RigSpec


@dataclass
class ImuEvent:
    timestamp_ns: int
    linear_accelerations: List[float]   # [ax, ay, az]  m/s^2
    angular_velocities: List[float]     # [gx, gy, gz]  rad/s


@dataclass
class FrameEvent:
    timestamp_ns: int
    images: List[np.ndarray]                       # one per rig camera (may be empty)
    depths: Optional[List[np.ndarray]] = None      # aligned to images, or None
    masks: Optional[List[np.ndarray]] = None       # aligned to images, or None
    meta: dict = field(default_factory=dict)        # free-form (e.g. for viz)


class FrameSource:
    """Base class for all input sources."""

    #: number of cameras this source produces images for
    num_cameras: int = 0

    def build_rig_spec(self) -> Optional[RigSpec]:
        """Return a rig derived from the dataset, or None if it must be in the TOML."""
        return None

    def __iter__(self) -> Iterator:  # -> Iterator[FrameEvent | ImuEvent]
        raise NotImplementedError

    def __len__(self) -> int:
        raise TypeError(f"{type(self).__name__} does not expose a length")

    def close(self) -> None:
        """Release any held resources (cameras, files). Default: no-op."""

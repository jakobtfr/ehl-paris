"""Per-plan normalization.

Each plan's *shape* is normalized into a stable model-space box and centred at
the origin so the model sees scale-invariant geometry. This follows the MSD
image-style scale ``256 / max(delta_x, delta_y)`` while storing absolute area and
bounding-box dimensions as separate conditioning scalars. The inverse transform
maps generated coordinates back to metric space for the MSD renderer and
outline preservation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import affinity
from shapely.geometry.base import BaseGeometry

from ..config import MODEL_SPACE_SIZE


@dataclass(frozen=True)
class PlanTransform:
    """Affine transform from normalized space back to original metric space.

    Forward (applied to raw geometry): translate by (-cx, -cy), then scale by
    ``scale``. Inverse (applied to model output): scale by ``1 / scale``, then
    translate by (+cx, +cy).
    """

    centroid_x: float
    centroid_y: float
    scale: float  # model-space units per metre

    def normalize(self, geom: BaseGeometry) -> BaseGeometry:
        g = affinity.translate(geom, xoff=-self.centroid_x, yoff=-self.centroid_y)
        return affinity.scale(g, xfact=self.scale, yfact=self.scale, origin=(0, 0))

    def invert(self, geom: BaseGeometry) -> BaseGeometry:
        inv = 1.0 / self.scale if self.scale else 1.0
        g = affinity.scale(geom, xfact=inv, yfact=inv, origin=(0, 0))
        return affinity.translate(g, xoff=self.centroid_x, yoff=self.centroid_y)


@dataclass(frozen=True)
class ScaleFeatures:
    """Absolute-scale conditioning scalars, fed to the model separately from the
    normalized shape so the count/type heads keep a real scale signal."""

    area_m2: float
    bbox_w_m: float
    bbox_h_m: float
    aspect_ratio: float   # w / h
    perimeter_m: float
    compactness: float    # 4*pi*area / perimeter^2  (1.0 = circle)


def fit_transform(outline: BaseGeometry) -> PlanTransform:
    cx, cy = outline.centroid.x, outline.centroid.y
    minx, miny, maxx, maxy = outline.bounds
    max_delta = max(maxx - minx, maxy - miny)
    scale = (MODEL_SPACE_SIZE / max_delta) if max_delta > 0 else 1.0
    return PlanTransform(centroid_x=cx, centroid_y=cy, scale=scale)


def scale_features(outline: BaseGeometry) -> ScaleFeatures:
    minx, miny, maxx, maxy = outline.bounds
    w, h = maxx - minx, maxy - miny
    area = outline.area
    perim = outline.length
    compact = (4.0 * math.pi * area / (perim * perim)) if perim > 0 else 0.0
    return ScaleFeatures(
        area_m2=area,
        bbox_w_m=w,
        bbox_h_m=h,
        aspect_ratio=(w / h) if h > 0 else 0.0,
        perimeter_m=perim,
        compactness=compact,
    )

"""Per-plan normalization.

Each plan's *shape* is normalized to unit area and centred at the origin so the
model sees scale-invariant geometry. But unit-area normalization erases absolute
size, and room count / type mix depend on it (a 30 m^2 studio vs a 120 m^2
flat). So we store the absolute area and bounding-box dimensions as separate
conditioning scalars, plus the exact inverse transform to map generated
coordinates back to metric space for the MSD renderer and outline preservation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import affinity
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class PlanTransform:
    """Affine transform from normalized space back to original metric space.

    Forward (applied to raw geometry):  translate by (-cx, -cy), then scale by
    1/sqrt(area).  Inverse (applied to model output): scale by sqrt(area), then
    translate by (+cx, +cy).
    """

    centroid_x: float
    centroid_y: float
    scale: float  # = sqrt(outline_area); divide to normalize, multiply to invert

    def normalize(self, geom: BaseGeometry) -> BaseGeometry:
        g = affinity.translate(geom, xoff=-self.centroid_x, yoff=-self.centroid_y)
        return affinity.scale(g, xfact=1.0 / self.scale, yfact=1.0 / self.scale, origin=(0, 0))

    def invert(self, geom: BaseGeometry) -> BaseGeometry:
        g = affinity.scale(geom, xfact=self.scale, yfact=self.scale, origin=(0, 0))
        return affinity.translate(g, xoff=self.centroid_x, yoff=self.centroid_y)


@dataclass(frozen=True)
class ScaleFeatures:
    """Absolute-scale conditioning scalars, fed to the model separately from the
    unit-area-normalized shape so the count/type heads keep a real scale signal."""

    area_m2: float
    bbox_w_m: float
    bbox_h_m: float
    aspect_ratio: float   # w / h
    perimeter_m: float
    compactness: float    # 4*pi*area / perimeter^2  (1.0 = circle)


def fit_transform(outline: BaseGeometry) -> PlanTransform:
    cx, cy = outline.centroid.x, outline.centroid.y
    scale = math.sqrt(outline.area) if outline.area > 0 else 1.0
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

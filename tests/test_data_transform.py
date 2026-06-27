"""Normalization + inverse-transform invariants.

The model sees scale-invariant geometry in a 256-unit box; generated coordinates
must map back to metric space exactly for the MSD renderer. These tests lock the
``256 / max(delta_x, delta_y)`` scale and the normalize <-> invert round-trip.

Synthetic geometry only.
"""

from __future__ import annotations

import math

from shapely.geometry import Polygon

from floorgen.config import MODEL_SPACE_SIZE
from floorgen.data.normalize import fit_transform, scale_features

# A 10 m (wide) x 4 m (tall) rectangle: max delta = 10, so scale = 256 / 10.
WIDE = Polygon([(2, 3), (12, 3), (12, 7), (2, 7)])
# Same rectangle rotated to be taller than wide: max delta still 10.
TALL = Polygon([(0, 0), (4, 0), (4, 10), (0, 10)])
TOL = 1e-6


def _bbox_wh(geom):
    minx, miny, maxx, maxy = geom.bounds
    return maxx - minx, maxy - miny


def test_scale_is_model_space_over_max_delta_wide():
    tf = fit_transform(WIDE)
    assert tf.scale == MODEL_SPACE_SIZE / 10.0


def test_scale_uses_the_larger_dimension_when_tall():
    tf = fit_transform(TALL)
    assert tf.scale == MODEL_SPACE_SIZE / 10.0


def test_normalized_geometry_fits_model_space_box():
    tf = fit_transform(WIDE)
    w, h = _bbox_wh(tf.normalize(WIDE))
    assert abs(max(w, h) - MODEL_SPACE_SIZE) < 1e-4


def test_normalize_centres_geometry_on_origin():
    tf = fit_transform(WIDE)
    c = tf.normalize(WIDE).centroid
    assert abs(c.x) < 1e-6 and abs(c.y) < 1e-6


def test_invert_recovers_original_geometry():
    tf = fit_transform(WIDE)
    restored = tf.invert(tf.normalize(WIDE))
    assert restored.symmetric_difference(WIDE).area < 1e-9


def test_roundtrip_preserves_each_coordinate():
    tf = fit_transform(TALL)
    restored = tf.invert(tf.normalize(TALL))
    for (ox, oy), (rx, ry) in zip(TALL.exterior.coords, restored.exterior.coords):
        assert abs(ox - rx) < TOL and abs(oy - ry) < TOL


def test_degenerate_outline_scale_falls_back_to_one():
    # A zero-extent point has max_delta == 0; the guard avoids a divide-by-zero
    # and keeps scale finite (1.0). A line still has one nonzero delta, so only a
    # true point trips the guard.
    point = Polygon([(5, 5), (5, 5), (5, 5)])
    tf = fit_transform(point)
    assert tf.scale == 1.0


def test_scale_features_match_rectangle_geometry():
    f = scale_features(WIDE)
    assert math.isclose(f.area_m2, 40.0, rel_tol=1e-9)
    assert math.isclose(f.bbox_w_m, 10.0, rel_tol=1e-9)
    assert math.isclose(f.bbox_h_m, 4.0, rel_tol=1e-9)
    assert math.isclose(f.aspect_ratio, 2.5, rel_tol=1e-9)
    assert math.isclose(f.perimeter_m, 28.0, rel_tol=1e-9)
    assert math.isclose(f.compactness, 4.0 * math.pi * 40.0 / (28.0 ** 2), rel_tol=1e-9)

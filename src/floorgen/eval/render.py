"""MSD-parity rasterizer.

The organisers rasterise vector layouts with the official MSD repo's rendering
scripts (caspervanengelenburg/msd, ``plot.py``) before computing FID / density /
coverage. We mirror that style: rooms filled by room-type colour, black edges,
equal aspect, no axes, fixed canvas. The exact organiser wrapper (image size,
DPI, palette) is a known open question -- everything that could differ is a
single config switch here so we can match parity once confirmed.

This module is torch-free: it produces HxWx3 uint8 numpy arrays. The FID/PRDC
code consumes those arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from ..config import ROOM_NAMES

# MSD-style room-type colour map. Indexed by ROOM_NAMES order. These approximate
# the MSD CMAP_ROOMTYPE palette; swap to the exact organiser palette via config
# once confirmed, without touching call sites.
ROOM_COLORS: dict[str, tuple[int, int, int]] = {
    "Bedroom":    (135, 206, 235),
    "Livingroom": (255, 165,   0),
    "Kitchen":    (220,  20,  60),
    "Dining":     (255, 215,   0),
    "Corridor":   (190, 190, 190),
    "Stairs":     (139,  69,  19),
    "Storeroom":  (160, 160, 200),
    "Bathroom":   ( 60, 179, 113),
    "Balcony":    (152, 251, 152),
    "Structure":  ( 90,  90,  90),
}


@dataclass(frozen=True)
class RenderConfig:
    """Rasterisation settings. Defaults mirror MSD ``plot.py`` conventions; set
    to the organiser wrapper's exact values once known."""

    size: int = 256          # output is size x size
    pad_frac: float = 0.02   # padding around the layout, fraction of extent
    edge_width: int = 1      # black room edge thickness in px (0 to disable)
    bg: tuple[int, int, int] = (255, 255, 255)


def _iter_polys(geom: BaseGeometry):
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms


def render_layout(
    rooms: list[tuple[BaseGeometry, int]],
    outline: BaseGeometry,
    cfg: RenderConfig | None = None,
) -> np.ndarray:
    """Rasterise a layout to an (size, size, 3) uint8 array, MSD-style.

    The outline fixes the canvas extent so real and generated layouts share a
    coordinate frame -- essential for FID comparability. Uses matplotlib's Agg
    backend so it is headless and deterministic.
    """
    if cfg is None:
        cfg = RenderConfig()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    minx, miny, maxx, maxy = outline.bounds
    w, h = maxx - minx, maxy - miny
    pad = max(w, h) * cfg.pad_frac
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
    extent = max(maxx - minx, maxy - miny)
    # centre the layout in a square frame
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    x0, x1 = cx - extent / 2, cx + extent / 2
    y0, y1 = cy - extent / 2, cy + extent / 2

    dpi = 100
    figsize = cfg.size / dpi
    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(tuple(c / 255 for c in cfg.bg))

    for geom, label_idx in rooms:
        name = ROOM_NAMES[label_idx] if 0 <= label_idx < len(ROOM_NAMES) else "Structure"
        color = tuple(c / 255 for c in ROOM_COLORS.get(name, (90, 90, 90)))
        edge = "black" if cfg.edge_width > 0 else color
        for poly in _iter_polys(geom):
            if poly.is_empty:
                continue
            ax.add_patch(MplPolygon(
                list(poly.exterior.coords), closed=True,
                facecolor=color, edgecolor=edge, linewidth=cfg.edge_width,
            ))

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
    plt.close(fig)
    return img

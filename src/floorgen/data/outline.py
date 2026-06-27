"""Outline construction and label canonicalization.

The apartment outline is the *only* model input. It is built exactly as the
official MSD challenge script specifies: buffer each room out by 0.3 m, union,
then buffer back in by 0.3 m. This fuses the rooms (and the wall gaps between
them) into one solid exterior shell. Every entry must condition on this same
boundary, so this function is the single source of truth for it.
"""

from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..config import SUBTYPE_TO_ROOM, WALL_BRIDGE_DISTANCE_M


def build_outline(
    room_polygons: list[BaseGeometry],
    wall_bridge_distance: float = WALL_BRIDGE_DISTANCE_M,
) -> Polygon | MultiPolygon:
    """Fuse a unit's room polygons into the single conditioning outline.

    Buffer out -> union -> buffer in, matching the official data-construction
    script. Returns a Polygon for the common case; a MultiPolygon only if the
    rooms are genuinely disconnected (e.g. a detached balcony).
    """
    if not room_polygons:
        raise ValueError("Cannot build an outline from zero rooms.")
    buffered = [g.buffer(wall_bridge_distance) for g in room_polygons]
    return unary_union(buffered).buffer(-wall_bridge_distance)


def largest_shell(geom: Polygon | MultiPolygon) -> Polygon:
    """Reduce a possibly-MultiPolygon outline to its largest single shell.

    Useful where the generator API requires one connected outline. Holes inside
    the chosen shell are preserved.
    """
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    raise TypeError(f"Unexpected outline geometry type: {geom.geom_type}")


def canonical_room_name(entity_subtype: str | None, roomtype: str | None) -> str:
    """Map an MSD area row to a canonical ROOM_NAMES label.

    Prefers the explicit subtype mapping (finer-grained); falls back to the
    dataset's own `roomtype` column, then to 'Structure' for anything
    unrecognised so the partition stays complete and auditable.
    """
    if entity_subtype is not None:
        key = str(entity_subtype).strip().upper()
        if key in SUBTYPE_TO_ROOM:
            return SUBTYPE_TO_ROOM[key]
    if roomtype is not None:
        rt = str(roomtype).strip()
        # The CSV roomtype values already match ROOM_NAMES casing in most cases.
        for name in ("Bedroom", "Livingroom", "Kitchen", "Dining", "Corridor",
                     "Stairs", "Storeroom", "Bathroom", "Balcony"):
            if rt.lower() == name.lower():
                return name
        if rt.lower() in ("structure", "shaft", "elevator", "void"):
            return "Structure"
    return "Structure"

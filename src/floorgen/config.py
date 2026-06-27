"""Central configuration. One place for the seed, paths, taxonomy, and the
geometry constants the whole pipeline shares. No magic numbers scattered in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility. The final submission seed is documented through config; 42 is
# the local default for repeatable development runs.
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Geometry constants (MSD uses metres).
# ---------------------------------------------------------------------------
# Outline construction: buffer each room out by this distance, union, buffer
# back in. Bridges the wall gaps between rooms into one solid shell. This value
# is fixed by the official challenge data-construction script.
WALL_BRIDGE_DISTANCE_M = 0.3

# Tolerances used by the validity-repair layer and contract tests (metres / m^2).
SNAP_TOL_M = 0.05          # snap room edges to each other / to the outline
MIN_ROOM_AREA_M2 = 0.5     # rooms smaller than this are slivers -> merge away
PARTITION_AREA_TOL = 0.02  # fractional area mismatch allowed for "partitions outline"
MAX_REPAIR_GAP_FRAC = 0.12
MAX_REPAIR_OVERLAP_FRAC = 0.25
MODEL_SPACE_SIZE = 256.0

# ---------------------------------------------------------------------------
# Room-type taxonomy. Anchored on the MSD `roomtype` column values actually
# present in the CSV for entity_type == "area" rows. "Structure" covers
# shafts/elevator cores; it is a real area row in MSD but is not a habitable
# room. We keep it as a class so the outline and partition stay complete, and
# expose an explicit mapping back to MSD names.
#
# Order matters: the index into ROOM_NAMES is the integer room_type the MSD
# renderer (plot.py / CMAP_ROOMTYPE) expects.
# ---------------------------------------------------------------------------
ROOM_NAMES: tuple[str, ...] = (
    "Bedroom",
    "Livingroom",
    "Kitchen",
    "Dining",
    "Corridor",
    "Stairs",
    "Storeroom",
    "Bathroom",
    "Balcony",
    "Structure",
)
ROOM_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(ROOM_NAMES)}

# Map raw MSD entity_subtype -> canonical ROOM_NAMES entry. Derived from the
# distinct subtypes observed in the dataset; rare variants collapse onto their
# nearest canonical class with an explicit, auditable rule (never invented).
SUBTYPE_TO_ROOM: dict[str, str] = {
    "ROOM": "Bedroom",
    "BEDROOM": "Bedroom",
    "LIVING_ROOM": "Livingroom",
    "LIVING_DINING": "Livingroom",
    "KITCHEN": "Kitchen",
    "KITCHEN_DINING": "Kitchen",
    "DINING": "Dining",
    "CORRIDOR": "Corridor",
    "CORRIDORS_AND_HALLS": "Corridor",
    "STAIRCASE": "Stairs",
    "STOREROOM": "Storeroom",
    "BATHROOM": "Bathroom",
    "BALCONY": "Balcony",
    "TERRACE": "Balcony",
    "SHAFT": "Structure",
    "ELEVATOR": "Structure",
    "VOID": "Structure",
}

# Fixed number of room slots for the primary fixed-slot generative model. The
# value should be set from the processed room-count distribution before training;
# presence logits handle variable counts.
MAX_ROOMS_K = 24


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem paths. The MSD CSV is read from MSD_CSV_PATH so the
    machine that holds the (large, ungitted) Kaggle data can run preprocessing
    without code changes."""

    msd_csv: Path = field(
        default_factory=lambda: Path(os.environ.get("MSD_CSV_PATH", "mds_V2_5.372k.csv"))
    )
    processed_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("FLOORGEN_PROCESSED", "data/processed"))
    )
    reports_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("FLOORGEN_REPORTS", "reports"))
    )


PATHS = Paths()

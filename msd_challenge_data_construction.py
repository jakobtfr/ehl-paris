"""Build and visualize one MSD challenge input/target pair.

The hackathon task conditions on one apartment outline and predicts the room
polygons inside it. In MSD, one apartment/dwelling is keyed by ``unit_id``;
``plan_id`` is the broader floor-plan context.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_UNIT_ID = 64314
WALL_BRIDGE_DISTANCE_M = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MSD outline/rooms visualization for one unit_id.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=os.environ.get("MSD_CSV_PATH", "mds_V2_5.372k.csv"),
        help="Path to mds_V2_5.372k.csv. Can also be set with MSD_CSV_PATH.",
    )
    parser.add_argument(
        "--unit-id",
        type=int,
        default=DEFAULT_UNIT_ID,
        help="MSD unit_id to visualize. A unit_id is one apartment/dwelling.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional PNG output path. Parent directories are created.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the Matplotlib window after rendering.",
    )
    return parser.parse_args()


def load_msd_geometry(csv_path: Path) -> Any:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"MSD CSV not found: {csv_path}. Pass --csv or set MSD_CSV_PATH."
        )

    import geopandas as gpd
    import pandas as pd
    from shapely import wkt

    df = pd.read_csv(csv_path)
    required_columns = {"geom", "unit_id", "entity_type"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"MSD CSV is missing required columns: {', '.join(missing)}")

    df["geom"] = df["geom"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry="geom")


def select_unit_rooms(gdf: Any, unit_id: int) -> Any:
    unit_gdf = gdf[gdf["unit_id"] == unit_id]
    if unit_gdf.empty:
        raise ValueError(f"No rows found for unit_id={unit_id}.")

    rooms_gdf = unit_gdf[unit_gdf["entity_type"] == "area"].copy()
    if rooms_gdf.empty:
        raise ValueError(f"No entity_type='area' room rows found for unit_id={unit_id}.")
    return rooms_gdf


def union_geometries(geometries: Any) -> Any:
    """Support both newer GeoPandas union_all and older unary_union APIs."""
    union_all = getattr(geometries, "union_all", None)
    if callable(union_all):
        return union_all()
    return geometries.unary_union


def make_outline(
    rooms_gdf: Any,
    wall_bridge_distance: float = WALL_BRIDGE_DISTANCE_M,
) -> Any:
    buffered_rooms = rooms_gdf.geometry.buffer(wall_bridge_distance)
    return union_geometries(buffered_rooms).buffer(-wall_bridge_distance)


def plot_unit_pair(
    rooms_gdf: Any,
    outline_geom: Any,
    unit_id: int,
) -> Any:
    import geopandas as gpd
    import matplotlib.pyplot as plt

    outline_gdf = gpd.GeoDataFrame(geometry=[outline_geom], crs=rooms_gdf.crs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    outline_gdf.plot(ax=ax1, facecolor="#f4f4f4", edgecolor="black", linewidth=3)
    ax1.set_title("Input: Clean Apartment Outline", fontsize=14, fontweight="bold")
    ax1.axis("equal")
    ax1.axis("off")

    rooms_gdf.plot(ax=ax2, cmap="Set3", edgecolor="white", linewidth=1.5)
    outline_gdf.plot(
        ax=ax2,
        facecolor="none",
        edgecolor="black",
        linewidth=2,
        alpha=0.4,
    )
    ax2.set_title("Target: Generated Rooms", fontsize=14, fontweight="bold")
    ax2.axis("equal")
    ax2.axis("off")

    fig.suptitle(
        f"Generative Task Data Pairing (Unit ID: {unit_id})",
        fontsize=16,
    )
    fig.tight_layout()
    return fig


def main() -> int:
    args = parse_args()

    try:
        gdf = load_msd_geometry(args.csv)
        rooms_gdf = select_unit_rooms(gdf, args.unit_id)
        outline_geom = make_outline(rooms_gdf)
        fig = plot_unit_pair(rooms_gdf, outline_geom, args.unit_id)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180, bbox_inches="tight")
        print(f"Wrote {args.output}")

    if args.show or not args.output:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

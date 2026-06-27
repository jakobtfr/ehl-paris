"""Live demo: outline in -> diverse room layouts out.

Calls floorgen.generate.sample_layouts, so it shows whatever backend is wired
into GENERATOR -- the baseline today, the trained flow model once registered.
No demo code changes when the model lands.

Run locally:   python -m floorgen.demo.app
Deploy:        copy this repo to a HuggingFace Space (SDK: gradio), entry app.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from shapely import wkt  # noqa: E402
from shapely.geometry.base import BaseGeometry  # noqa: E402

from ..config import ROOM_NAMES, SEED  # noqa: E402
from ..eval.render import ROOM_COLORS  # noqa: E402
from ..generate import sample_layouts  # noqa: E402

PRESETS = json.loads((Path(__file__).parent / "presets.json").read_text())


def _draw(ax, layout, outline: BaseGeometry, title: str) -> None:
    for r in layout:
        poly = r["polygon"]
        color = tuple(c / 255 for c in ROOM_COLORS.get(r["label"], (90, 90, 90)))
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color=color, ec="black", lw=0.8)
        c = poly.centroid
        ax.text(c.x, c.y, r["label"][:4], ha="center", va="center", fontsize=6)
    ox, oy = outline.exterior.xy
    ax.plot(ox, oy, color="black", lw=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def generate_layouts(preset_name: str, custom_wkt: str, n_samples: int, seed: int):
    src = custom_wkt.strip() if custom_wkt.strip() else PRESETS[preset_name]
    try:
        outline = wkt.loads(src)
    except Exception as e:
        raise gr.Error(f"Could not parse outline WKT: {e}") from e

    samples = sample_layouts(outline, seed=int(seed), n_samples=int(n_samples))
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, layout in zip(axes, samples):
        _draw(ax, layout, outline, f"{len(layout)} rooms")
    fig.suptitle("Sampled layouts (same outline, seed-stable, diverse)", fontsize=12)
    fig.tight_layout()

    # also return the first sample's polygons as GeoJSON text for inspection
    gj = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "properties": {"label": r["label"]},
             "geometry": r["geojson"]}
            for r in samples[0]
        ],
    }
    return fig, json.dumps(gj, indent=2)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="floorgen — floor-plan generator") as demo:
        gr.Markdown(
            "# floorgen\n"
            "Generate interior room layouts from an apartment **outline**. "
            "Pick a real Swiss apartment outline or paste your own polygon WKT, "
            "then sample several diverse arrangements."
        )
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    choices=list(PRESETS), value=list(PRESETS)[0],
                    label="Preset outline (real MSD apartment)")
                custom = gr.Textbox(
                    label="…or custom outline WKT (overrides preset)",
                    placeholder="POLYGON ((0 0, 10 0, 10 8, 0 8, 0 0))", lines=2)
                n_samples = gr.Slider(1, 6, value=3, step=1, label="Samples")
                seed = gr.Number(value=SEED, label="Seed", precision=0)
                go = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Layouts")
                geojson = gr.Code(label="Sample 1 — room polygons (GeoJSON)", language="json")
        gr.Markdown(
            "Room types: " + ", ".join(ROOM_NAMES) + ".  \n"
            "*Backend swaps to the trained flow-matching model automatically once "
            "registered — this UI does not change.*"
        )
        go.click(generate_layouts, [preset, custom, n_samples, seed], [plot, geojson])
        demo.load(generate_layouts, [preset, custom, n_samples, seed], [plot, geojson])
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()

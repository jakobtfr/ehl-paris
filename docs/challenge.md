# Mirror Mirror on the Wall: Who has the best room of them all?

By Davis AI. Prize: 500 EUR.

## Challenge

Given only the outline of an apartment, generate a complete and plausible set of
interior rooms as vector polygons, using a diffusion or flow-matching model.
The model decides where the kitchen, bedrooms, bathroom, hallway, and living
spaces go, then draws each one as a clean polygon that fills the available floor
area.

Entries are scored on the realism and diversity of the generated layouts,
measured by FID, density, and coverage against real Swiss residential floor
plans. Submissions are not judged against one correct answer. They are judged
against the distribution of real Swiss residential floor plans. A strong
submission produces layouts that are both:

- **Realistic:** they look like something an architect would actually draw.
- **Diverse:** given the same outline, the model can propose several genuinely
  different valid arrangements, rather than one memorised template.

## Inputs and Outputs

- **Input:** boundary polygon of an apartment, used as the condition.
- **Output:** set of interior room polygons that partition the apartment, each
  labelled by room type, in the same geometry format as the MSD `geom` column
  so the same rendering script can be used.
- **Method constraint:** the generator must be a diffusion or flow-matching
  model, not a deterministic solver or a purely rule-based partitioner.

The model receives one input, the apartment outline, and must produce the rooms
that fill it:

- where each room sits
- each room's shape
- each room's type

Rooms must be returned as vector polygons, never as a pixel grid. Teams may
choose how to encode the outline, which diffusion or flow-matching model to
build, and how to parameterise rooms.

From a bare outline, the model fills in typed rooms. The outline carries no
rooms, walls, or graph: it is the only condition.

## Data

The dataset is Modified Swiss Dwellings (MSD). The room polygons are in the
geometry dataframe, not in the image or graph folders.

| Location | Description |
| --- | --- |
| `mds_V2_5.372k.csv` | Room geometry. Polygons are in the `geom` column as WKT. This is the vector modality, separate from the `struct_in`, `graph_in`, `graph_out`, and `full_out` folders, which this task does not use. |
| `unit_id` | Selects one apartment/dwelling. Keep `plan_id` and `floor_id` as broader context metadata. |
| `entity_type == area` | Selects the room and space polygons for a given `unit_id`. |
| `outline` | Built from rooms by the provided script: buffer each room out by 0.3 m, union, then buffer back in, fusing them into one exterior shell. |

## Fixed Requirements

- **Input:** apartment outline as one polygon.
- **Output:** set of typed room polygons.
- **Outline construction:** use the standard outline-construction script so
  every entry conditions on the same boundary.

## Open Design Choices

Teams are free to choose:

- how to encode the outline
- the diffusion or flow-matching architecture
- the room representation, such as rectangles or corner sequences

The representation may be anything except a pixel grid.

## Judging Criteria

Solutions are scored against FID and Diversity & Coverage. In metric terms,
this means FID, density, and coverage against a held-out set of real Swiss floor
plans.

To win, the generated layouts need to be both realistic and broad: low-FID
plans that resemble real Swiss apartments, high-density samples that stay on the
real data manifold, and high-coverage samples that avoid collapsing to one safe
layout pattern.

Evaluator clarification from organisers:

- **Density and coverage:** computed with the formulas/implementation from
  `clovaai/generative-evaluation-prdc`.
- **FID:** computed with TorchMetrics'
  `torchmetrics.image.fid.FrechetInceptionDistance`.
- **Rendering:** vector layouts are rasterised with the rendering scripts from
  the official MSD repository, especially `plot.py`. The MSD script is the
  source of truth for rendering settings such as `512x512` image size, colors,
  line widths, padding, antialiasing, and whether graph details are drawn.
- **Room types:** room labels are used in rendering, so semantic labels affect
  rendered colors and can directly affect FID.
- **Test-time compute:** seed and sample count are not predetermined by the
  organisers. Teams may generate as many candidates as they want, and may use
  post-processing or ranking, as long as the method is documented properly.

### FID: Frechet Inception Distance

FID measures how close the distribution of generated layouts is to the
distribution of real floor plans. Rather than comparing one generated plan to
one real plan, it compares the full population of outputs to the full
population of real designs in a learned feature space. Lower FID means generated
layouts are statistically harder to distinguish from real ones. It rewards
plausibility and penalises visual artefacts, implausible room shapes, and mode
collapse.

### Density and Coverage

Density and coverage separate the two things FID blends together:

- **Density:** do generated layouts land in regions where real layouts are
  common? This is fidelity: outputs should be high-quality and on-distribution.
- **Coverage:** do generated layouts span the full variety of real layouts? This
  is diversity: outputs should explore the design space rather than repeating
  one safe answer.

A model that memorises a few safe layouts can score well on density but should
be punished on coverage. A model that produces wild variety but unrealistic
plans should score high coverage but low density. The goal is to score well on
both.

Scoring is run by the organisers on a held-out set of plans with a fixed
reference set and rasterisation path. Teams should document their generation
seed, number of candidates per outline, post-processing, and ranking strategy so
scores are interpretable and reproducible.

## Rules and Integrity

- **Time:** Saturday to Sunday.
- **Model:** diffusion- or flow-matching-based, trained from scratch.
- **Reproducibility:** document random seeds, sample counts, test-time compute,
  post-processing, and ranking. Seed `42` may be useful for local reproducible
  experiments, but the organiser clarified that the final seed is not
  predetermined.
- **Honest reporting:** results must be honest and reproducible.
- **No leakage:** do not train on the test/evaluation set or otherwise leak
  held-out data into training.
- **Metrics:** report correct metrics.

## Submission

Submission requirements:

- Pitch deck
- GitHub repository
- Live demo

| Field | What it is | Format | Required |
| --- | --- | --- | --- |
| Pitch deck | Short deck explaining the approach and results | `.pdf` or `.pptx` | Yes |
| GitHub repository | Code, public or private | Repo link | Yes |
| Live demo | Working demo of the model | URL | Yes |

The repository should include training and generation code, a
`generate(outline)` entry point that returns room polygons, and a short
methodology writeup. Include model weights when practical, or at minimum make
the weight/generation provenance inspectable from code, configs, and logs.
Generated test-split outputs should match the MSD `geom` column format so the
same rendering script can be used.

Organiser clarification: generally the generated outputs for the test split are
enough for scoring, but the presentation must explain the model and
parameterisation, and the code should make it clear how the submitted
weights/generations were obtained.

Teams must include their full working-session record: the
`entire/checkpoints/v1` branch, with at least one prompt. This is captured for a
process-quality review and is advisory only. It does not count toward placement.

Submissions with a GitHub repository receive an automated, LLM-assisted code
review. Review weighting:

| Criterion | Weight |
| --- | ---: |
| Code quality | 30% |
| Architecture | 25% |
| Challenge alignment | 25% |
| Innovation | 20% |

## At a Glance

- **By:** Davis AI
- **Prize:** 500 EUR
- **Type:** scored challenge, counts for league points
- **Task:** generative floor-plan completion from an apartment outline
- **Model class:** diffusion or flow matching
- **Metrics:** FID, density, coverage
- **Dataset:** real Swiss residential floor plans
- **Winning guidance:** be original, keep room labels meaningful, and document
  any test-time compute or post-processing.

## Resources

- [Modified Swiss Dwellings dataset on Kaggle](https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings/data)
- [Hackathon slide deck](https://hackathon.commercialdeckdavis.com/) - slide 7 contains the challenge description.

## Appendix: Provided Data-Construction Script

The reference script loads the geometry dataframe, selects one dwelling's room
polygons where `entity_type == area`, fuses them into the single outline used as
the model input, and plots the input and target side by side.

```python
import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt

# 1. Load data
csv_path = "mds_V2_5.372k.csv"
df = pd.read_csv(csv_path)
df["geom"] = df["geom"].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry="geom")

# 2. Isolate one apartment/dwelling
sample_unit_id = 64314
apartment_gdf = gdf[gdf["unit_id"] == sample_unit_id]
rooms_gdf = apartment_gdf[apartment_gdf["entity_type"] == "area"]

# Merge the rooms to create the outline:
# 1. Buffer outward by 30 centimeters (MSD uses meters as units)
# 2. Cleanly unify them into one solid geometry
# 3. Buffer back inward by 30 centimeters to restore the original scale
wall_bridge_distance = 0.3
solid_outline_geom = (
    rooms_gdf.geometry.buffer(wall_bridge_distance)
    .union_all()
    .buffer(-wall_bridge_distance)
)

# Convert the resulting Shapely geometry back into a GeoDataFrame for plotting
outline_gdf = gpd.GeoDataFrame(geometry=[solid_outline_geom], crs=rooms_gdf.crs)

# 3. Plot side-by-side again
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left plot: condition
outline_gdf.plot(ax=ax1, facecolor="#f4f4f4", edgecolor="black", linewidth=3)
ax1.set_title("Input: Clean Apartment Outline", fontsize=14, fontweight="bold")
ax1.axis("equal")
ax1.axis("off")

# Right plot: target
rooms_gdf.plot(ax=ax2, cmap="Set3", edgecolor="white", linewidth=1.5)
outline_gdf.plot(ax=ax2, facecolor="none", edgecolor="black", linewidth=2, alpha=0.4)
ax2.set_title("Target: Generated Rooms", fontsize=14, fontweight="bold")
ax2.axis("equal")
ax2.axis("off")

plt.suptitle(f"Generative Task Data Pairing (Unit ID: {sample_unit_id})", fontsize=16)
plt.tight_layout()
plt.show()
```

Dataset: Modified Swiss Dwellings on Kaggle. Challenge by Davis, in partnership
with TUM.ai and Iterate.

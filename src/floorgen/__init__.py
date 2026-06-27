"""floorgen: generative floor-plan completion from an apartment outline.

MSD / Davis AI hackathon. The model emits room geometry directly (vector
polygons); a deterministic layer only repairs validity. Scored on FID, density,
and coverage against real Swiss residential floor plans.
"""

__version__ = "0.1.0"

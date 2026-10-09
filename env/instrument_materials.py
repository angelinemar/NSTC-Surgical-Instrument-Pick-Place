"""Recorder-owned, visual-only materials for every surgical instrument.

The source USDs are inconsistent: some contain authored PBR materials and the
scissor contains none. Recording must nevertheless present every target and
table distractor with a predictable, readable stainless-steel response under
the per-episode randomized lights. These overrides affect appearance only;
collision geometry, mass, scale, and semantic labels are unchanged.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualMaterialSpec:
    diffuse_color: tuple[float, float, float]
    metallic: float
    roughness: float


# Slightly different neutral finishes keep highlights from collapsing all thin
# tools into the same tone. Differences stay subtle: class identity must come
# from geometry, not a synthetic class-colour shortcut. Roughness stays above
# 0.3 to avoid mirror-like white/black clipping.
INSTRUMENT_MATERIAL_OVERRIDES = {
    "scalpel": VisualMaterialSpec(
        diffuse_color=(0.52, 0.54, 0.56),
        metallic=0.82,
        roughness=0.34,
    ),
    "scissor": VisualMaterialSpec(
        diffuse_color=(0.48, 0.50, 0.52),
        metallic=0.78,
        roughness=0.36,
    ),
    "love_retractor": VisualMaterialSpec(
        diffuse_color=(0.46, 0.49, 0.51),
        metallic=0.76,
        roughness=0.40,
    ),
    "kelly": VisualMaterialSpec(
        diffuse_color=(0.50, 0.52, 0.54),
        metallic=0.80,
        roughness=0.37,
    ),
    "scalpel_type2": VisualMaterialSpec(
        diffuse_color=(0.55, 0.54, 0.51),
        metallic=0.80,
        roughness=0.35,
    ),
}

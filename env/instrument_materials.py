"""Visual-only fallbacks for instrument source assets with no material."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualMaterialSpec:
    diffuse_color: tuple[float, float, float]
    metallic: float
    roughness: float


# Source-asset audit: scalpel, love_retractor, kelly, and scalpel_type2 already
# carry authored USD/PBR bindings. The scissor mesh carries none, so give every
# spawned scissor (target, base object, and duplicate table distractor) a
# readable brushed-steel fallback. Do not flatten the valid multi-material
# assets into one generic shader merely to increase contrast.
INSTRUMENT_MATERIAL_OVERRIDES = {
    "scissor": VisualMaterialSpec(
        diffuse_color=(0.48, 0.50, 0.52),
        metallic=0.78,
        roughness=0.36,
    ),
}

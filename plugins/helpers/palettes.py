"""Color-theory palette catalog.

Each palette has five named slots: primary, secondary, tertiary, accent, background.
Colors are derived from a base hue using simple HSV math, so the catalog is just a
list of (id, name, kind, base_hue) tuples — actual hex values are computed at import.

Skills consume a Palette via the Canvas API and should reference slots by name
rather than hard-coding hex values, so the user's selected palette is honored.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    id: str
    name: str
    kind: str
    base_hue: int  # 0..359
    primary: str
    secondary: str
    tertiary: str
    accent: str
    background: str

    @property
    def colors(self) -> dict:
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "tertiary": self.tertiary,
            "accent": self.accent,
            "background": self.background,
        }

    @property
    def slots(self) -> list[str]:
        return [self.primary, self.secondary, self.tertiary, self.accent, self.background]

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "base_hue": self.base_hue, "colors": self.colors}


def _hex(h: float, s: float, v: float) -> str:
    h = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


def _monochromatic(id_: str, name: str, base: int) -> Palette:
    return Palette(
        id=id_, name=name, kind="monochromatic", base_hue=base,
        primary=_hex(base, 0.78, 0.88),
        secondary=_hex(base, 0.55, 0.70),
        tertiary=_hex(base, 0.35, 0.50),
        accent=_hex(base, 0.10, 0.98),
        background=_hex(base, 0.30, 0.10),
    )


def _neutral(id_: str, name: str) -> Palette:
    return Palette(id=id_, name=name, kind="monochromatic", base_hue=220, primary="#C9D3E0", secondary="#8D99A8", tertiary="#586271", accent="#FFFFFF", background="#111318")


def _complementary(id_: str, name: str, base: int) -> Palette:
    comp = (base + 180) % 360
    return Palette(
        id=id_, name=name, kind="complementary", base_hue=base,
        primary=_hex(base, 0.78, 0.88),
        secondary=_hex(comp, 0.78, 0.88),
        tertiary=_hex(base, 0.40, 0.55),
        accent=_hex(comp, 0.30, 0.98),
        background=_hex(base, 0.25, 0.08),
    )


def _analogous(id_: str, name: str, base: int) -> Palette:
    return Palette(
        id=id_, name=name, kind="analogous", base_hue=base,
        primary=_hex(base, 0.75, 0.88),
        secondary=_hex((base + 30) % 360, 0.70, 0.85),
        tertiary=_hex((base - 30) % 360, 0.70, 0.80),
        accent=_hex(base, 0.18, 0.98),
        background=_hex((base - 60) % 360, 0.40, 0.10),
    )


def _triadic(id_: str, name: str, base: int) -> Palette:
    return Palette(
        id=id_, name=name, kind="triadic", base_hue=base,
        primary=_hex(base, 0.78, 0.88),
        secondary=_hex((base + 120) % 360, 0.78, 0.85),
        tertiary=_hex((base + 240) % 360, 0.78, 0.85),
        accent=_hex(base, 0.10, 0.98),
        background=_hex((base + 120) % 360, 0.40, 0.10),
    )


def _tetradic(id_: str, name: str, base: int) -> Palette:
    return Palette(
        id=id_, name=name, kind="tetradic", base_hue=base,
        primary=_hex(base, 0.78, 0.88),
        secondary=_hex((base + 90) % 360, 0.72, 0.85),
        tertiary=_hex((base + 180) % 360, 0.78, 0.82),
        accent=_hex((base + 270) % 360, 0.72, 0.90),
        background=_hex(base, 0.30, 0.08),
    )


def _split_comp(id_: str, name: str, base: int) -> Palette:
    return Palette(
        id=id_, name=name, kind="split_complementary", base_hue=base,
        primary=_hex(base, 0.78, 0.88),
        secondary=_hex((base + 150) % 360, 0.75, 0.85),
        tertiary=_hex((base + 210) % 360, 0.75, 0.85),
        accent=_hex(base, 0.15, 0.98),
        background=_hex(base, 0.35, 0.09),
    )


_CATALOG: list[Palette] = [
    _neutral("monochromatic_neutral", "Neutral Mono"),
    _monochromatic("monochromatic_indigo", "Indigo Mono", 245),
    _monochromatic("monochromatic_rose", "Rose Mono", 345),
    _complementary("complementary_teal_coral", "Teal & Coral", 185),
    _complementary("complementary_violet_lime", "Violet & Lime", 280),
    _analogous("analogous_sunset", "Sunset Analogous", 20),
    _analogous("analogous_forest", "Forest Analogous", 130),
    _triadic("triadic_warm", "Warm Triad", 20),
    _triadic("triadic_jewel", "Jewel Triad", 200),
    _tetradic("tetradic_neon", "Neon Tetrad", 300),
    _split_comp("split_complementary_ember", "Ember Split", 15),
]

_BY_ID: dict[str, Palette] = {p.id: p for p in _CATALOG}

DEFAULT_PALETTE_ID = "monochromatic_neutral"


def list_palettes() -> list[Palette]:
    return list(_CATALOG)


def get_palette(palette_id: str | None) -> Palette:
    if palette_id and palette_id in _BY_ID:
        return _BY_ID[palette_id]
    return _BY_ID[DEFAULT_PALETTE_ID]


def palette_exists(palette_id: str) -> bool:
    return palette_id in _BY_ID

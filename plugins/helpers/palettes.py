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


def _curated(id_: str, name: str, kind: str, base: int, *, primary: str, secondary: str, tertiary: str, accent: str, background: str) -> Palette:
    return Palette(id=id_, name=name, kind=kind, base_hue=base,
                   primary=primary, secondary=secondary, tertiary=tertiary,
                   accent=accent, background=background)


_CATALOG: list[Palette] = [
    # ── Muted / modern / minimal (default-leaning) ───────────────────────────
    _curated("neutral_mono", "Neutral Mono", "monochromatic", 220,
             primary="#C9D3E0", secondary="#8D99A8", tertiary="#586271",
             accent="#FFFFFF", background="#111318"),
    _curated("sandstone", "Sandstone", "earthy", 30,
             primary="#D4A574", secondary="#A67B5B", tertiary="#7A6553",
             accent="#F5EBE0", background="#2C2419"),
    _curated("coastal_mist", "Coastal Mist", "cool_muted", 210,
             primary="#94A8B5", secondary="#B4BFC4", tertiary="#6A7884",
             accent="#E8E2D5", background="#1F2832"),
    _curated("japandi", "Japandi", "warm_minimal", 35,
             primary="#BFA888", secondary="#7A7568", tertiary="#3D3A36",
             accent="#E8E0D0", background="#2A2724"),
    _curated("linen", "Linen", "soft_neutral", 25,
             primary="#C9B8A8", secondary="#B89A8E", tertiary="#8B9A85",
             accent="#EFE7DB", background="#3A3530"),
    _curated("studio_mono", "Studio Mono", "monochromatic", 210,
             primary="#A8B0B8", secondary="#7A828A", tertiary="#4E555C",
             accent="#D8D5CF", background="#1A1D20"),
    _curated("botanical", "Botanical", "earthy", 90,
             primary="#8FA68E", secondary="#B58572", tertiary="#5C6E5C",
             accent="#EBE2D4", background="#1F2A1D"),
    _curated("concrete", "Concrete", "industrial", 30,
             primary="#B8B0A6", secondary="#8C807A", tertiary="#564F49",
             accent="#C9967A", background="#232120"),
    _curated("dusk", "Dusk", "moody", 320,
             primary="#B59DAE", secondary="#D4A89A", tertiary="#7A6A78",
             accent="#EDDCD2", background="#2A1F2A"),
    _curated("cafe", "Café", "warm_earthy", 30,
             primary="#C9A876", secondary="#8B5E3C", tertiary="#5C4030",
             accent="#EBDBC7", background="#1F1410"),
    _curated("ochre", "Ochre", "midcentury", 40,
             primary="#C99A48", secondary="#8B6F4E", tertiary="#5A4030",
             accent="#E8DCC8", background="#1F1812"),
    _curated("frost", "Frost", "cool_minimal", 200,
             primary="#D8DFE3", secondary="#A5B0B8", tertiary="#6A7884",
             accent="#EFEDE9", background="#1B2026"),
    _curated("clay", "Clay", "earthy", 15,
             primary="#C4856C", secondary="#9C6B57", tertiary="#5E3F33",
             accent="#E8D4C0", background="#241813"),
    _curated("nordic", "Nordic", "cool_minimal", 215,
             primary="#A8B5C2", secondary="#7B8794", tertiary="#4A5560",
             accent="#E5E1D6", background="#1C2128"),
    _curated("moss", "Moss", "earthy", 110,
             primary="#7A8C6B", secondary="#A8A082", tertiary="#4D5C44",
             accent="#E1DCC8", background="#1E2419"),
    _curated("ink_paper", "Ink & Paper", "monochromatic", 35,
             primary="#3A3530", secondary="#7A7268", tertiary="#A89C8C",
             accent="#C99A48", background="#E8E0D0"),
    # ── Vivid / playful (used sparingly) ─────────────────────────────────────
    _analogous("analogous_sunset", "Sunset Analogous", 20),
    _triadic("triadic_jewel", "Jewel Triad", 200),
    _tetradic("tetradic_neon", "Neon Tetrad", 300),
    _complementary("teal_coral", "Teal & Coral", 185),
]

_BY_ID: dict[str, Palette] = {p.id: p for p in _CATALOG}

DEFAULT_PALETTE_ID = "japandi"


def list_palettes() -> list[Palette]:
    return list(_CATALOG)


def get_palette(palette_id: str | None) -> Palette:
    if palette_id and palette_id in _BY_ID:
        return _BY_ID[palette_id]
    return _BY_ID[DEFAULT_PALETTE_ID]


def palette_exists(palette_id: str) -> bool:
    return palette_id in _BY_ID

"""Color palette catalog for generative art.

Each palette has five named slots: primary, secondary, tertiary, accent, background.

Techniques consume a Palette via the Canvas API and should reference slots by name
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

    def fingerprint(self) -> str:
        """Short content hash of this palette's slots.

        Cache keys use this rather than ``id`` so a palette whose colors
        were edited produces a fresh key (and old cached renders become
        unreachable rather than stale)."""
        import hashlib
        return hashlib.sha256("|".join(self.slots).encode("utf-8")).hexdigest()[:12]


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
    _curated("japandi", "Japandi", "warm_minimal", 35, primary="#CBB48B", secondary="#8B8574", tertiary="#4B4740", accent="#EFE6D2", background="#26231F"),
    _curated("neutral_mono", "Neutral Mono", "monochromatic", 220, primary="#D7DEE8", secondary="#9DA8B7", tertiary="#5B6472", accent="#FFFFFF", background="#101217"),
    _curated("ink_paper", "Ink & Paper", "monochromatic", 35, primary="#28231F", secondary="#6F665B", tertiary="#B8AA96", accent="#D99A3D", background="#EFE5D0"),
    _curated("obsidian", "Obsidian", "dark_prismatic", 265, primary="#7C5CFF", secondary="#1BD8D2", tertiary="#F05A9D", accent="#F7F2FF", background="#07070D"),
    _curated("frost", "Frost", "cool_minimal", 200, primary="#DCEBEE", secondary="#91B8C4", tertiary="#4D6D82", accent="#FFF7D6", background="#111A22"),
    _curated("aurora", "Aurora", "luminous_cool", 155, primary="#30F2A2", secondary="#3EA7FF", tertiary="#9A6BFF", accent="#E7FF7A", background="#06111C"),
    _curated("magma", "Magma", "volcanic", 15, primary="#FF5A3D", secondary="#F6A32D", tertiary="#7B2346", accent="#FFE66D", background="#140609"),
    _curated("deep_sea", "Deep Sea", "aquatic", 195, primary="#2DD6C7", secondary="#2567D8", tertiary="#17345F", accent="#B6FFF1", background="#03121F"),
    _curated("botanical", "Botanical", "earthy", 105, primary="#9CCB70", secondary="#4F8F5B", tertiary="#D69A52", accent="#F0E6A8", background="#102116"),
    _curated("desert_bloom", "Desert Bloom", "warm_earthy", 25, primary="#E09A5F", secondary="#C95C52", tertiary="#7E6A49", accent="#FFDFA3", background="#24150E"),
    _curated("rose_circuit", "Rose Circuit", "electric_warm", 330, primary="#FF4FA3", secondary="#7A5CFF", tertiary="#26D4FF", accent="#FFE870", background="#130817"),
    _curated("acid_lime", "Acid Lime", "high_energy", 85, primary="#B7FF2A", secondary="#00D9A3", tertiary="#2852FF", accent="#F8FF90", background="#071006"),
    _curated("violet_storm", "Violet Storm", "moody", 275, primary="#A66BFF", secondary="#4D78FF", tertiary="#B23A62", accent="#D8C8FF", background="#100A1D"),
    _curated("candy_glass", "Candy Glass", "pastel_vivid", 330, primary="#FF8BCB", secondary="#82E6FF", tertiary="#B69CFF", accent="#FFF2A8", background="#201326"),
    _curated("bauhaus", "Bauhaus", "graphic", 45, primary="#E23D32", secondary="#F2C230", tertiary="#1D66B2", accent="#F7F1DF", background="#101010"),
    _curated("synthwave", "Synthwave", "retro_neon", 300, primary="#FF3BD4", secondary="#23D7FF", tertiary="#6C4DFF", accent="#FFD166", background="#0B0620"),
    _curated("lichen", "Lichen", "muted_organic", 95, primary="#B8C97A", secondary="#7D9A6B", tertiary="#D1B16A", accent="#F0EBD0", background="#202719"),
    _curated("noir_gold", "Noir Gold", "luxury_dark", 42, primary="#D4AF37", secondary="#8A6A2A", tertiary="#C45C35", accent="#FFF0B8", background="#090807"),
    _curated("coral_reef", "Coral Reef", "tropical", 175, primary="#2DD4BF", secondary="#FF6B6B", tertiary="#5B7CFA", accent="#FFE66D", background="#06242D"),
    _curated("radioactive", "Radioactive", "toxic_contrast", 115, primary="#75FF33", secondary="#00B8A9", tertiary="#8A2BE2", accent="#F5FF00", background="#050A05"),
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

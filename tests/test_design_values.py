"""DESIGN.md is executable: its ontology and this test suite may not drift apart.

DESIGN.md names guardian tests for each core value (part). This manifest test
pins the binding in both directions it can decide:

- every guardian test file named in DESIGN.md must exist, and
- every part chapter must name at least one guardian.

Renaming or deleting a guardian test therefore requires amending DESIGN.md in
the same change — the same discipline test_kernel_boundary.py imposes on the
kernel boundary.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESIGN = ROOT / "DESIGN.md"

CORE_VALUES = [
    "Correctness",
    "Modularity",
    "Safety",
    "Simplicity",
    "Efficiency",
    "Elegance",
    "Practicality",
    "Readability",
]


def _text():
    return DESIGN.read_text(encoding="utf-8")


def test_design_md_exists():
    assert DESIGN.is_file(), "DESIGN.md is the design ontology; do not remove it"


def test_every_named_guardian_test_exists():
    named = set(re.findall(r"`(tests/test_\w+\.py)`", _text()))
    assert named, "DESIGN.md names no guardian tests — the ontology lost its teeth"
    missing = sorted(p for p in named if not (ROOT / p).is_file())
    assert not missing, f"DESIGN.md names guardian tests that do not exist: {missing}"


def test_every_core_value_has_a_chapter_with_guardians():
    text = _text()
    for value in CORE_VALUES:
        m = re.search(rf"^## Part [IV]+: {value}$", text, flags=re.MULTILINE)
        assert m, f"DESIGN.md has no chapter for core value {value!r}"
        chapter = text[m.end():]
        end = chapter.find("\n## ")
        if end != -1:
            chapter = chapter[:end]
        assert "**Guardian tests:**" in chapter, (
            f"chapter {value!r} names no guardian tests"
        )

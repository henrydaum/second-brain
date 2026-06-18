from plugins.services.helpers import parser_registry


def test_native_modality_defaults_treat_gif_as_image():
    parser_registry.clear()

    assert parser_registry.get_modality(".gif") == "image"
    assert parser_registry.get_modality(".png") == "image"


def test_store_gif_parsers_register_image_and_video():
    import importlib.util
    from pathlib import Path

    import pytest

    store = Path(__file__).resolve().parents[2] / "sb-store" / "services" / "helpers"
    if not store.exists():
        pytest.skip("sb-store worktree not present")

    parser_registry.clear()
    for name in ("parse_image", "parse_video"):
        path = store / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"store_{name}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ImportError as e:
            pytest.skip(str(e))

    assert parser_registry.get_modality(".gif") == "image"
    assert set(parser_registry.get_modalities_for(".gif")) == {"image", "video"}

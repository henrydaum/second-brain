"""Agent-facing authoring instructions agree with the executable tree table."""

from pathlib import Path

import trees


ROOT = Path(__file__).resolve().parents[1]
AUTHORING_SOURCES = (
    ROOT / "README.md",
    ROOT / "docs" / "SDK.md",
    ROOT / "docs" / "MIGRATING_PLUGINS.md",
    ROOT / "templates" / "tool_template.py",
    ROOT / "templates" / "task_template.py",
    ROOT / "templates" / "service_template.py",
    ROOT / "templates" / "command_template.py",
    ROOT / "templates" / "frontend_template.py",
    ROOT / "templates" / "script_template.py",
    ROOT / "templates" / "llm_backend_template.py",
    ROOT / "templates" / "parser_template.py",
)


def test_the_authoring_roots_come_from_the_kernel_table():
    """Every root the kernel routes is taught where an author is sent to read.

    This used to check the system prompt, which carried the eight roots as an
    ASCII table. That table left with the rest of the authoring tutorial: the
    prompt is paid on every turn including the ones about nothing in this
    codebase, and the roots are only needed at the moment somebody writes a
    file. What the prompt keeps is the pointer.

    So the invariant moved rather than went away, and it is the same one — a
    root the kernel routes but no authoring source names is a folder an agent
    can only find by guessing. It is checked across the sources collectively
    because that is how they divide the work: each template names its own
    family folder, and SDK.md covers the roots that belong to no family.
    """
    documented = "\n".join(path.read_text(encoding="utf-8")
                            for path in AUTHORING_SOURCES)
    for root in trees.ROOTS:
        assert f"{root.name}/" in documented, root.name


def test_system_prompt_teaches_the_sdk_template_and_permission_workflow():
    documented = (ROOT / "agent" / "system_prompt_static.md").read_text(
        encoding="utf-8")
    # The order is the invariant, not the sentence: SDK.md defines the
    # vocabulary the template is written in, so an agent sent to the template
    # first reads a worked example in a language it has not been taught.
    # Matched on the two references rather than on a phrase, which is what let
    # a rewording of the same instruction fail this.
    sdk_position = documented.index("docs/SDK.md")
    template_position = documented.index("templates/")
    assert sdk_position < template_position
    assert "fs_writable_dirs" in documented
    assert "belong to the user" in documented
    assert "docs/PERMISSIONS_MAP.md" in documented
    assert "Telegram" not in documented


def test_every_template_routes_back_to_sdk_and_implementation():
    for path in (ROOT / "templates").glob("*_template.py"):
        text = path.read_text(encoding="utf-8")
        assert "docs/SDK.md" in text, path.name
        assert "sandbox/" in text, path.name


def test_authoring_sources_do_not_restore_retired_layouts():
    retired = ("plugins/tools/", "plugins/frontends/", "helpers/llm_",
               "helpers/parse_image.py")
    offenders = []
    for path in AUTHORING_SOURCES:
        text = path.read_text(encoding="utf-8")
        for phrase in retired:
            if phrase in text:
                offenders.append(f"{path.relative_to(ROOT)}: {phrase}")
    assert not offenders, "retired authoring paths:\n  " + "\n  ".join(offenders)


def test_each_plugin_template_names_its_workspace_family():
    for family in trees.FAMILIES:
        template = ROOT / "templates" / f"{family.name[:-1]}_template.py"
        text = template.read_text(encoding="utf-8")
        assert f"workspace/{family.name}/" in text

    llm = (ROOT / "templates" / "llm_backend_template.py").read_text(
        encoding="utf-8")
    assert "llm/llm_<provider>.py" in llm

    parser = (ROOT / "templates" / "parser_template.py").read_text(
        encoding="utf-8")
    assert "workspace/parsers/" in parser

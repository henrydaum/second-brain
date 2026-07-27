"""Guest form values stay pure, serializable, and bridge-compatible."""

from pathlib import Path

from sandbox import Sandbox
from sandbox.bridge import adapt
from sandbox.validator import validate_file
from guest.forms import FormStep


def test_guest_form_step_is_plain_mapping_data():
    step = FormStep(
        "action",
        "Choose.",
        enum=["load", "unload"],
        enum_labels=["Load it", "Unload it"],
        columns=2,
    )

    assert isinstance(step, dict)
    assert step.to_dict() == {
        "name": "action",
        "prompt": "Choose.",
        "required": True,
        "type": "string",
        "enum": ["load", "unload"],
        "enum_labels": ["Load it", "Unload it"],
        "default": None,
        "prompt_when_missing": False,
        "columns": 2,
    }
    assert "validator" not in step


def test_guest_form_step_crosses_subprocess_and_rehydrates(tmp_path):
    plugin = tmp_path / "command_form.py"
    plugin.write_text(
        "from guest.bases import BaseCommand\n"
        "from guest.forms import FormStep\n\n"
        "class FormCommand(BaseCommand):\n"
        "    name = 'form'\n"
        "    isolation = 'subprocess'\n"
        "    def form(self, sdk, args):\n"
        "        return [FormStep('value', 'Enter it.', False, "
        "type='integer', default=3)]\n"
        "    def run(self, sdk, args):\n"
        "        return args.get('value')\n",
        encoding="utf-8",
    )

    report = validate_file(plugin)
    assert report.ok, report.render()

    sandbox = Sandbox()
    try:
        result = sandbox.run(
            plugin, "FormCommand", kwargs={"args": {}}, method="form")
    finally:
        sandbox.shutdown()
    assert result.ok, result.error
    assert result.data == [FormStep(
        "value", "Enter it.", False, type="integer", default=3)]

    module = adapt(plugin)
    command_cls = next(
        value for value in vars(module).values()
        if isinstance(value, type) and getattr(value, "_sandboxed", False)
    )
    [native] = command_cls().form({}, None)
    assert native.to_dict() == FormStep(
        "value", "Enter it.", False, type="integer", default=3)

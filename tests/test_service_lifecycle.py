from types import SimpleNamespace

from plugins.BaseService import EXTENSION, BaseService, is_user_managed_service, should_autoload_service
from plugins.commands.command_services import ServicesCommand
from plugins.frontends.helpers.formatters import format_services


class ManagedService(BaseService):
    """Managed service."""
    model_name = "Managed"


class ExtensionService(BaseService):
    """Extension service."""
    model_name = "Extension"
    lifecycle = EXTENSION


def test_base_service_has_default_noop_lifecycle():
    service = BaseService()

    assert service.load() is True
    assert service.loaded is True
    service.unload()
    assert service.loaded is False


def test_extension_services_autoload_without_config_entry():
    managed = ManagedService()
    extension = ExtensionService()

    assert not should_autoload_service("managed", managed, {"autoload_services": []})
    assert should_autoload_service("managed", managed, {"autoload_services": ["managed"]})
    assert should_autoload_service("extension", extension, {"autoload_services": []})
    assert is_user_managed_service(managed)
    assert not is_user_managed_service(extension)


def test_services_command_does_not_offer_load_unload_for_extensions():
    extension = ExtensionService()
    extension.load()
    context = SimpleNamespace(services={"extension": extension})

    steps = ServicesCommand().form({"service_name": "extension"}, context)
    result = ServicesCommand().run({"service_name": "extension"}, context)
    blocked = ServicesCommand().run({"service_name": "extension", "action": "unload"}, context)

    assert [step.name for step in steps] == ["service_name"]
    assert "Status: Extension" in result
    assert blocked == "extension is an installed extension and is loaded automatically."


def test_format_services_groups_extensions():
    text = format_services([
        {"name": "extension", "loaded": True, "model_name": "Extension", "lifecycle": "extension"},
        {"name": "managed", "loaded": True, "model_name": "Managed", "lifecycle": "managed"},
        {"name": "cold", "loaded": False, "model_name": "Cold", "lifecycle": "managed"},
    ])

    assert "Extensions:" in text
    assert "Loaded:" in text
    assert "Unloaded:" in text
    assert text.index("Extensions:") < text.index("Loaded:") < text.index("Unloaded:")

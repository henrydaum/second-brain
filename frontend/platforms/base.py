from __future__ import annotations

from abc import ABC, abstractmethod

from frontend.types import FrontendAction, FrontendSession, PlatformCapabilities


class BasePlatformAdapter(ABC):
    name = ""
    capabilities = PlatformCapabilities()

    def __init__(self, ctrl, shutdown_fn, shutdown_event, tool_registry, services, config, root_dir):
        self.ctrl = ctrl
        self.shutdown_fn = shutdown_fn
        self.shutdown_event = shutdown_event
        self.tool_registry = tool_registry
        self.services = services
        self.config = config
        self.root_dir = root_dir
        self.runtime = None

    def bind_runtime(self, runtime):
        self.runtime = runtime

    @abstractmethod
    def start(self):
        raise NotImplementedError

    def stop(self):
        return None

    @abstractmethod
    def send_action(self, session: FrontendSession, action: FrontendAction):
        raise NotImplementedError

    def default_session(self) -> FrontendSession | None:
        return None

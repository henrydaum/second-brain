"""
Controller.

The user-facing command layer between frontends and the core runtime.
It exposes control actions as plain methods so the REPL, Telegram, and
other frontends can all drive the same behavior.

The controller never prints — it returns structured data or status strings.
The caller decides how to display them.
"""

import logging
import threading
from pathlib import Path

from frontend.token_stripper import strip_model_tokens

logger = logging.getLogger("Controller")


class Controller:
    def __init__(self, orchestrator, db, services: dict, config: dict, tool_registry=None):
        self.orchestrator = orchestrator
        self.db = db
        self.services = services
        self.config = config
        self.tool_registry = tool_registry
        self._title_generation_lock = threading.Lock()
        self._pending_title_generations: set[int] = set()

    # =================================================================
    # SERVICES
    # =================================================================

    def list_services(self) -> list[dict]:
        """List all services and their status."""
        return [
            {"name": name, "loaded": getattr(svc, 'loaded', False),
             "model_name": getattr(svc, 'model_name', '')}
            for name, svc in self.services.items()
        ]

    # =================================================================
    # CONVERSATION TITLES
    # =================================================================

    _TITLE_MAX_LEN = 80
    _TITLE_SYSTEM_PROMPT = "You label conversations with short, concrete titles. You output only the title — never a sentence, greeting, or explanation."
    _TITLE_USER_TEMPLATE = (
        "<conversation>\n"
        "{transcript}\n"
        "</conversation>\n\n"
        "Write a 2-6 word title summarizing what the conversation is about.\n"
        "Rules:\n"
        "- Output only the title, no preamble, no quotes, no markdown\n"
        "- Be concrete and specific, not generic\n"
        "- Use title case\n\n"
        "Examples:\n"
        "Conversation about Rolls-Royce Cullinan pricing -> Cullinan Price\n"
        "Conversation planning a Virginia holiday -> Virginia Holiday Getaway\n"
        "Conversation debugging a SQLite migration -> SQLite Migration Bug\n\n"
        "Title:"
    )

    def maybe_generate_conversation_title_async(self, conversation_id: int):
        """Best-effort async title generation for conversations.

        Runs only when the conversation still has an auto-generated fallback
        title, so manual or already-upgraded titles are left untouched.
        """
        if not conversation_id:
            return

        with self._title_generation_lock:
            if conversation_id in self._pending_title_generations:
                return
            self._pending_title_generations.add(conversation_id)

        thread = threading.Thread(
            target=self._generate_conversation_title_worker,
            args=(conversation_id,),
            daemon=True,
            name=f"TitleGen-{conversation_id}",
        )
        thread.start()

    def _generate_conversation_title_worker(self, conversation_id: int):
        try:
            self._generate_conversation_title(conversation_id)
        except Exception as e:
            logger.debug(f"Conversation title generation failed for {conversation_id}: {e}")
        finally:
            with self._title_generation_lock:
                self._pending_title_generations.discard(conversation_id)

    def _generate_conversation_title(self, conversation_id: int):
        conversation = self.db.get_conversation(conversation_id)
        if not conversation:
            return

        messages = self.db.get_conversation_messages(conversation_id)
        if len(messages) < 2:
            return

        current_title = self._normalize_title(conversation.get("title"))
        fallback_title = self._fallback_conversation_title(messages)
        if not self._should_replace_conversation_title(current_title, fallback_title):
            return

        llm = self.services.get("llm")
        if llm is None or not getattr(llm, "loaded", False):
            return
        if getattr(llm, "active", None) is None and hasattr(llm, "active"):
            return

        transcript = self._title_generation_transcript(messages)
        if not transcript:
            return

        response = llm.invoke([
            {"role": "system", "content": self._TITLE_SYSTEM_PROMPT},
            {"role": "user", "content": self._TITLE_USER_TEMPLATE.format(transcript=transcript)},
        ])
        if getattr(response, "error", None):
            return

        title = self._sanitize_generated_title(getattr(response, "content", ""))
        if not title:
            return

        latest = self.db.get_conversation(conversation_id)
        if not latest:
            return
        latest_title = self._normalize_title(latest.get("title"))
        if not self._should_replace_conversation_title(latest_title, fallback_title):
            return

        self.db.update_conversation_title(conversation_id, title)
        logger.info(f"Updated conversation {conversation_id} title to '{title}'")

    def _title_generation_transcript(self, messages: list[dict]) -> str:
        lines = []
        for msg in messages[:6]:
            role = (msg.get("role") or "").upper()
            if role == "TOOL":
                continue

            content = msg.get("content") or ""
            if role == "ASSISTANT":
                try:
                    import json
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        content = parsed.get("content") or ""
                except Exception:
                    pass

            content = " ".join(content.split()).strip()
            if not content:
                continue
            if len(content) > 300:
                content = content[:300].rstrip() + "..."
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _fallback_conversation_title(self, messages: list[dict]) -> str:
        for msg in messages:
            if (msg.get("role") or "") == "user":
                return self._truncate_title(msg.get("content") or "")
        return "New conversation"

    def _should_replace_conversation_title(self, current_title: str, fallback_title: str) -> bool:
        if not current_title:
            return True
        lowered = current_title.casefold()
        if lowered in {"new conversation", "conversation", "new chat", "chat"}:
            return True
        if current_title == fallback_title:
            return True
        return False

    def _truncate_title(self, text: str) -> str:
        text = " ".join((text or "").replace("\n", " ").split()).strip()
        if not text:
            return "New conversation"
        return text[:self._TITLE_MAX_LEN]

    def _normalize_title(self, text: str | None) -> str:
        return " ".join((text or "").replace("\n", " ").split()).strip()

    def _sanitize_generated_title(self, text: str) -> str:
        title, _ = strip_model_tokens(text or "")
        title = title.strip()
        if not title:
            return ""

        title = title.splitlines()[0].strip()
        title = title.strip().strip("\"'`*#-: ")
        title = " ".join(title.split())
        title = title[:self._TITLE_MAX_LEN].strip()

        generic = {"new conversation", "conversation", "chat", "untitled", "title"}
        if not title or title.casefold() in generic:
            return ""

        return title

    def load_service(self, name: str) -> str:
        """Load a service and re-check blocked tasks."""
        svc = self.services.get(name)
        if svc is None:
            return f"Unknown service: '{name}'. Run /services to see all services."

        if getattr(svc, 'loaded', False):
            return f"Service '{name}' is already loaded."

        logger.info(f"Loading service '{name}'...")
        try:
            success = svc.load()
        except Exception as e:
            return f"Failed to load '{name}': {e}. Run /config to review service-related settings."

        if not success:
            return f"Service '{name}' failed to load. Run /config to review service-related settings."

        from events.event_bus import bus
        from events.event_channels import SERVICE_LOADED
        bus.emit(SERVICE_LOADED, {"name": name, "loaded": True})

        return f"Service '{name}' loaded."

    def unload_service(self, name: str) -> str:
        """Unload a service to free resources."""
        svc = self.services.get(name)
        if svc is None:
            return f"Unknown service: '{name}'. Run /services to see all services."

        if not getattr(svc, 'loaded', False):
            return f"Service '{name}' is already unloaded."

        try:
            svc.unload()
        except Exception as e:
            return f"Error unloading '{name}': {e}"

        from events.event_bus import bus
        from events.event_channels import SERVICE_LOADED
        bus.emit(SERVICE_LOADED, {"name": name, "loaded": False})

        return f"Service '{name}' unloaded."

    def reload_services_for_settings(self, changed_keys: set, root_dir: Path) -> list[str]:
        """Rebuild only the services whose config_settings include a changed key.

        Groups affected services by source module so that build_services()
        is called once per module (it may return multiple services).

        Returns a list of human-readable feedback strings.
        """
        from plugins.plugin_discovery import get_setting_service_map, discover_services

        svc_map = get_setting_service_map()
        affected: set[str] = set()
        for key in changed_keys:
            if key in svc_map:
                affected.update(svc_map[key])

        if not affected:
            return []

        feedback = []

        # Group affected services by source module path.
        # Baked-in services share a module path derived from their class.
        module_groups: dict[str, list[str]] = {}
        for svc_name in affected:
            svc = self.services.get(svc_name)
            if svc is None:
                continue
            src = getattr(svc, '_source_path', None) or svc.__class__.__module__
            module_groups.setdefault(src, []).append(svc_name)

        # Pause orchestrator to prevent dispatches during the swap
        saved_pauses = set(self.orchestrator.paused)
        self.orchestrator.paused.update(self.orchestrator.tasks.keys())

        try:
            # Unload all affected services first
            previously_loaded: list[str] = []
            for svc_names in module_groups.values():
                for n in svc_names:
                    if getattr(self.services.get(n), 'loaded', False):
                        previously_loaded.append(n)
                        try:
                            self.services[n].unload()
                        except Exception as ex:
                            logger.warning(f"Failed to unload '{n}': {ex}")

            # Single rediscovery — cherry-pick only the affected services
            new_services = discover_services(root_dir, self.config)
            for n in affected:
                if n in new_services:
                    self.services[n] = new_services[n]

            # Reload services that were previously loaded
            for n in previously_loaded:
                svc = self.services.get(n)
                if svc:
                    try:
                        svc.load()
                        feedback.append(f"'{n}' reloaded.")
                    except Exception as ex:
                        feedback.append(f"'{n}' failed to reload: {ex}")
        finally:
            # Restore orchestrator pauses + notify listeners
            self.orchestrator.paused.clear()
            self.orchestrator.paused.update(saved_pauses)
            from events.event_bus import bus
            from events.event_channels import SERVICE_LOADED
            bus.emit(SERVICE_LOADED, {"name": None, "loaded": True})

        return feedback

    def apply_runtime_config_changes(self, changed_keys: set[str]) -> list[str]:
        """Apply config changes that can take effect without a restart."""
        feedback = []

        if "poll_interval" in changed_keys:
            self.orchestrator.poll_interval = self.config["poll_interval"]
            feedback.append(f"Poll interval -> {self.config['poll_interval']}s")

        if "max_workers" in changed_keys:
            import threading
            from concurrent.futures import ThreadPoolExecutor

            old_executor = self.orchestrator.executor
            self.orchestrator.max_workers = self.config["max_workers"]
            self.orchestrator.executor = ThreadPoolExecutor(
                max_workers=self.config["max_workers"],
                thread_name_prefix="Worker",
            )

            # Tasks with max_workers=0 inherit the global worker limit.
            for name, task in self.orchestrator.tasks.items():
                if task.max_workers <= 0:
                    self.orchestrator.task_semaphores[name] = threading.Semaphore(
                        self.config["max_workers"]
                    )

            old_executor.shutdown(wait=False)
            feedback.append(f"Worker pool -> {self.config['max_workers']} threads")

        return feedback

    # =================================================================
    # TASKS
    # =================================================================

    def list_tasks(self, trigger: str | None = None) -> list[dict]:
        """List tasks with counts, paused state, and trigger metadata."""
        stats = self.db.get_system_stats()
        path_task_stats = stats.get("tasks", {})
        event_task_stats = self.db.get_run_stats()
        empty_counts = {"PENDING": 0, "PROCESSING": 0, "DONE": 0, "FAILED": 0}

        tasks = []
        for name, task in self.orchestrator.tasks.items():
            task_trigger = getattr(task, "trigger", "path")
            if trigger and task_trigger != trigger:
                continue

            count_source = event_task_stats if task_trigger == "event" else path_task_stats
            counts = {**empty_counts, **count_source.get(name, {})}
            tasks.append({
                "name": name,
                "trigger": task_trigger,
                "trigger_channels": list(getattr(task, "trigger_channels", []) or []),
                "counts": counts,
                "paused": name in self.orchestrator.paused,
                "requires_services": list(getattr(task, "requires_services", []) or []),
            })

        return sorted(tasks, key=lambda item: item["name"])

    def list_task_names(self, trigger: str | None = None) -> list[str]:
        """Return task names, optionally filtered by trigger kind."""
        return [task["name"] for task in self.list_tasks(trigger=trigger)]

    def pause_task(self, name: str) -> str:
        """Pause a task. It stays PENDING but won't be dispatched."""
        if name not in self.orchestrator.tasks:
            return f"Unknown task: '{name}'. Run /tasks to see all tasks."

        if name in self.orchestrator.paused:
            return f"Task '{name}' is already paused."

        self.orchestrator.paused.add(name)
        return f"Task '{name}' paused."

    def unpause_task(self, name: str) -> str:
        """Unpause a task. Pending work will resume on next dispatch cycle."""
        if name not in self.orchestrator.tasks:
            return f"Unknown task: '{name}'. Run /tasks to see all tasks."

        if name not in self.orchestrator.paused:
            return f"Task '{name}' is not paused."

        self.orchestrator.paused.discard(name)
        return f"Task '{name}' unpaused."

    def reset_task(self, name: str) -> str:
        """Reset ALL entries for a task back to PENDING, including downstream tasks."""
        if name not in self.orchestrator.tasks:
            return f"Unknown task: '{name}'. Run /tasks to see all tasks."
        if getattr(self.orchestrator.tasks[name], "trigger", "path") != "path":
            return (f"Task '{name}' is event-triggered — only path-driven tasks can be reset. "
                    f"Run /trigger {name} to fire it instead.")

        self.db.reset_task(name)
        downstream = self.orchestrator.get_all_downstream(name)
        if downstream:
            self.db.invalidate_tasks_bulk(downstream)
        return f"Task '{name}' reset to Pending (+ {len(downstream)} downstream)."

    def retry_task(self, name: str) -> str:
        """Retry only FAILED entries for a task, invalidating their downstream tasks."""
        if name not in self.orchestrator.tasks:
            return f"Unknown task: '{name}'. Run /tasks to see all tasks."
        if getattr(self.orchestrator.tasks[name], "trigger", "path") != "path":
            return (f"Task '{name}' is event-triggered — only path-driven tasks can be retried. "
                    f"Run /trigger {name} to fire a new run instead.")

        failed_paths = self.db.get_paths_for_task_status(name, "FAILED")
        self.db.reset_failed_tasks(name)
        downstream = self.orchestrator.get_all_downstream(name)
        if downstream and failed_paths:
            self.db.invalidate_tasks_for_paths(downstream, failed_paths)
        return f"Task '{name}' — failed entries reset to Pending."

    def trigger_event_task(self, name: str, payload: dict | None = None) -> str:
        """Manually fire an event-triggered task by emitting on its first
        declared channel. Convenience wrapper for tools/REPL — direct
        bus.emit works identically."""
        task = self.orchestrator.tasks.get(name)
        if task is None:
            return f"Unknown task: '{name}'. Run /tasks to see all tasks."
        if getattr(task, "trigger", "path") != "event":
            return f"Task '{name}' is not event-triggered. Run /tasks to see event-driven tasks."
        channels = getattr(task, "trigger_channels", []) or []
        if not channels:
            return f"Task '{name}' has no trigger channels declared."
        payload = payload or {}
        if not isinstance(payload, dict):
            return "Trigger payload must be a JSON object."
        schema = getattr(task, "event_payload_schema", {}) or {}
        required = schema.get("required", [])
        missing = []
        for field in required:
            value = payload.get(field)
            if value is None:
                missing.append(field)
            elif isinstance(value, str) and not value.strip():
                missing.append(field)
        if missing:
            missing_str = ", ".join(missing)
            return f"Task '{name}' requires payload fields: {missing_str}."
        from events.event_bus import bus
        bus.emit(channels[0], payload)
        return f"Emitted '{channels[0]}' for task '{name}'."

    def retry_all(self) -> str:
        """Retry all FAILED path-task entries, invalidating downstream."""
        for name, task in self.orchestrator.tasks.items():
            if getattr(task, "trigger", "path") != "path":
                continue
            failed_paths = self.db.get_paths_for_task_status(name, "FAILED")
            if failed_paths:
                downstream = self.orchestrator.get_all_downstream(name)
                if downstream:
                    self.db.invalidate_tasks_for_paths(downstream, failed_paths)
        self.db.reset_failed_tasks()
        return "All failed path-driven tasks reset to Pending."

    # =================================================================
    # TOOLS
    # =================================================================

    def list_tools(self) -> list[dict]:
        """List all registered tools with descriptions and required services."""
        if self.tool_registry is None:
            return []
        return [
            {"name": name,
                "description": (tool.description or "").split("\n")[0],
                "max_calls": tool.max_calls,
                "requires_services": getattr(tool, 'requires_services', []),
                "parameters": getattr(tool, 'parameters', {}),
                "_source_path": getattr(tool, '_source_path', None),
                "_mutable": getattr(tool, '_mutable', False),
            }
            for name, tool in self.tool_registry.tools.items()
        ]

    def list_locations(self, filter_type: str | None = None) -> dict:
        """Walk ROOT_DIR and DATA_DIR and return their file trees.

        Args:
            filter_type: Optional — one of 'tools', 'tasks', or 'services'.
                         When set, only the directories relevant to that
                         plugin type are listed.

        Returns a dict with:
            root_path  — absolute path to ROOT_DIR
            data_path  — absolute path to DATA_DIR
            root_tree  — list of relative-path strings under ROOT_DIR
            data_tree  — list of absolute-path strings under DATA_DIR
        """
        from paths import ROOT_DIR, DATA_DIR, SANDBOX_TOOLS, SANDBOX_TASKS, SANDBOX_SERVICES
        import plugins.plugin_discovery as _pd

        # Which subdirectories matter for each plugin type.
        _type_dirs = {
            "tools":    {
                "root": [_pd._TOOL_CONFIG["baked_in_dir"]],
                "data": [SANDBOX_TOOLS],
            },
            "tasks":    {
                "root": [_pd._TASK_CONFIG["baked_in_dir"]],
                "data": [SANDBOX_TASKS],
            },
            "services": {
                "root": [_pd._SERVICE_CONFIG["baked_in_dir"]],
                "data": [SANDBOX_SERVICES],
            },
        }

        def _walk_tree(base: Path, root_dirs: list[Path] | None = None) -> list[Path]:
            """Return sorted file paths under *base*.

            If *root_dirs* is given, only files that live under one of
            those directories are included.
            """
            if not base.exists():
                return []
            paths: list[Path] = []
            for p in sorted(base.rglob("*")):
                if p.is_dir():
                    continue
                # Skip hidden / cache directories
                parts = p.relative_to(base).parts
                if any(part.startswith(".") or part == "__pycache__" for part in parts):
                    continue
                if root_dirs:
                    if not any(p.is_relative_to(rd) for rd in root_dirs):
                        continue
                paths.append(p)
            return paths

        if filter_type:
            dirs = _type_dirs.get(filter_type)
            if dirs is None:
                return {"root_path": str(ROOT_DIR), "data_path": str(DATA_DIR),
                        "root_tree": [], "data_tree": []}
            root_files = _walk_tree(ROOT_DIR, dirs["root"])
            data_files = _walk_tree(DATA_DIR, dirs["data"])
        else:
            root_files = _walk_tree(ROOT_DIR)
            data_files = _walk_tree(DATA_DIR)

        root_tree = [str(p.relative_to(ROOT_DIR)) for p in root_files]
        data_tree = [str(p) for p in data_files]

        return {
            "root_path": str(ROOT_DIR),
            "data_path": str(DATA_DIR),
            "root_tree": root_tree,
            "data_tree": data_tree,
        }

    def call_tool(self, name: str, kwargs: dict):
        """Call a tool by name and return the ToolResult.
        Approval prompts for destructive tools flow through the event bus
        (APPROVAL_REQUESTED), not a callback threaded through here."""
        if self.tool_registry is None:
            from plugins.BaseTool import ToolResult
            return ToolResult.failed("No tool registry available.")
        return self.tool_registry.call(name, **kwargs)

    # =================================================================
    # PLUGINS
    # =================================================================

    def reload_plugins(self, root_dir: Path) -> str:
        """Re-discover tasks and tools from all plugin directories."""
        from plugins.plugin_discovery import discover_tasks, discover_tools

        saved_pauses = set(self.orchestrator.paused)
        self.orchestrator.paused.update(self.orchestrator.tasks.keys())

        try:
            mutable_task_names = [
                name for name, task in list(self.orchestrator.tasks.items())
                if getattr(task, "_mutable", False)
            ]
            for name in mutable_task_names:
                self.orchestrator.unregister_task(name)

            mutable_tool_names = [
                name for name, tool in list(self.tool_registry.tools.items())
                if getattr(tool, "_mutable", False)
            ]
            for name in mutable_tool_names:
                self.tool_registry.unregister(name)

            discover_tasks(root_dir, self.orchestrator, self.config, reload=True)
            discover_tools(root_dir, self.tool_registry, self.config, reload=True)
            self.orchestrator.refresh_event_subscriptions()
        finally:
            self.orchestrator.paused.clear()
            self.orchestrator.paused.update(saved_pauses)

        return "Plugins reloaded."

    # =================================================================
    # HELP
    # =================================================================

    def help(self) -> list[dict]:
        """Command list for the REPL."""
        return [
            {"command": "call <tool> <json>", "description": "Call a tool directly"},
            {"command": "disable <n>", "description": "Disable a tool from agent use"},
            {"command": "enable <n>", "description": "Enable a tool for agent use"},
            {"command": "load <n>", "description": "Load a service"},
            {"command": "pause <n>", "description": "Pause a task"},
            {"command": "pipeline", "description": "Show the path-driven task dependency graph"},
            {"command": "quit / exit", "description": "Shutdown"},
            {"command": "reload", "description": "Hot-reload tasks and tools"},
            {"command": "reset <n>", "description": "Reset all entries for a path-driven task"},
            {"command": "retry <n>", "description": "Retry failed entries for a path-driven task"},
            {"command": "retry all", "description": "Retry all failed across all path-driven tasks"},
            {"command": "services", "description": "List services and status"},
            {"command": "tasks", "description": "List path-driven and event-driven tasks"},
            {"command": "tools", "description": "List registered tools"},
            {"command": "trigger <n> [json]", "description": "Manually fire an event-triggered task"},
            {"command": "unload <n>", "description": "Unload a service"},
            {"command": "unpause <n>", "description": "Unpause a task"},
        ]

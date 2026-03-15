"""
Auto-discover tasks.

Scans Stage_2/tasks/task_*.py for BaseTask subclasses and registers them
with the orchestrator.

To add a new task, drop a task_<name>.py file into Stage_2/tasks/.
"""

import importlib
import inspect
import logging
import sys
from pathlib import Path

from Stage_2.BaseTask import BaseTask

logger = logging.getLogger("Discovery")


def discover(root_dir: Path, orchestrator, config: dict, reload: bool = False):
    tasks_dir = root_dir / "Stage_2" / "tasks"
    for py_file in sorted(tasks_dir.glob("task_*.py")):
        module_name = f"Stage_2.tasks.{py_file.stem}"
        try:
            if reload and module_name in sys.modules:
                module = importlib.reload(sys.modules[module_name])
            else:
                module = importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Could not import {module_name}: {e}")
            continue
        except Exception as e:
            logger.error(f"Failed to load {module_name}: {e}", exc_info=True)
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, BaseTask) and cls is not BaseTask and cls.__module__ == module_name:
                try:
                    orchestrator.register_task(cls())
                except Exception as e:
                    logger.error(f"Could not register task {cls.__name__}: {e}", exc_info=True)

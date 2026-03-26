"""
Auto-discover tasks.

Scans Stage_2/tasks/task_*.py for BaseTask subclasses (baked-in),
then scans SANDBOX_TASKS for agent-created tasks. Registers them
with the orchestrator.

Name collisions between sandbox and baked-in tasks are rejected.
"""

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path

from Stage_2.BaseTask import BaseTask
from paths import SANDBOX_TASKS

logger = logging.getLogger("Discovery")


def discover(root_dir: Path, orchestrator, config: dict, reload: bool = False):
    import time
    t0 = time.time()
    count = 0
    baked_in_names = set()

    # --- Baked-in tasks ---
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
                    instance = cls()
                    instance._mutable = False
                    orchestrator.register_task(instance)
                    baked_in_names.add(instance.name)
                    count += 1
                except Exception as e:
                    logger.error(f"Could not register task {cls.__name__}: {e}", exc_info=True)

    # --- Sandbox tasks ---
    if SANDBOX_TASKS.exists():
        for py_file in sorted(SANDBOX_TASKS.glob("task_*.py")):
            module_name = f"sandbox_tasks.{py_file.stem}"
            try:
                if reload and module_name in sys.modules:
                    module = importlib.reload(sys.modules[module_name])
                else:
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
            except Exception as e:
                logger.error(f"Failed to load sandbox task {py_file.name}: {e}")
                continue
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseTask) and cls is not BaseTask and cls.__module__ == module_name:
                    try:
                        instance = cls()
                        if instance.name in baked_in_names:
                            logger.warning(f"Sandbox task '{instance.name}' collides with baked-in — skipped")
                            continue
                        instance._mutable = True
                        instance._source_path = str(py_file)
                        orchestrator.register_task(instance)
                        count += 1
                    except Exception as e:
                        logger.error(f"Could not register sandbox task {cls.__name__}: {e}")

    logger.info(f"Discovered {count} task(s) in {time.time() - t0:.2f}s")

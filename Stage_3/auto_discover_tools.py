"""
Auto-discover tools.

Scans Stage_3/tools/tool_*.py for BaseTool subclasses (baked-in),
then scans SANDBOX_TOOLS for agent-created tools. Registers them
with the tool registry.

Name collisions between sandbox and baked-in tools are rejected.
"""

import importlib
import importlib.util
import inspect
import logging
import sys
from pathlib import Path

from Stage_3.BaseTool import BaseTool
from paths import SANDBOX_TOOLS

logger = logging.getLogger("Discovery")


def discover(root_dir: Path, tool_registry, config: dict, reload: bool = False):
    import time
    t0 = time.time()
    count = 0
    baked_in_names = set()

    # --- Baked-in tools ---
    tools_dir = root_dir / "Stage_3" / "tools"
    for py_file in sorted(tools_dir.glob("tool_*.py")):
        module_name = f"Stage_3.tools.{py_file.stem}"
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
            if issubclass(cls, BaseTool) and cls is not BaseTool and cls.__module__ == module_name:
                try:
                    instance = cls()
                    instance._mutable = False
                    tool_registry.register(instance)
                    baked_in_names.add(instance.name)
                    count += 1
                except Exception as e:
                    logger.error(f"Could not register tool {cls.__name__}: {e}", exc_info=True)

    # --- Sandbox tools ---
    if SANDBOX_TOOLS.exists():
        for py_file in sorted(SANDBOX_TOOLS.glob("tool_*.py")):
            module_name = f"sandbox_tools_{py_file.stem}"
            try:
                if reload and module_name in sys.modules:
                    module = importlib.reload(sys.modules[module_name])
                else:
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
            except Exception as e:
                logger.error(f"Failed to load sandbox tool {py_file.name}: {e}")
                continue
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseTool) and cls is not BaseTool and cls.__module__ == module_name:
                    try:
                        instance = cls()
                        if instance.name in baked_in_names:
                            logger.warning(f"Sandbox tool '{instance.name}' collides with baked-in — skipped")
                            continue
                        instance._mutable = True
                        instance._source_path = str(py_file)
                        tool_registry.register(instance)
                        count += 1
                    except Exception as e:
                        logger.error(f"Could not register sandbox tool {cls.__name__}: {e}")

    logger.info(f"Discovered {count} tool(s) in {time.time() - t0:.2f}s")

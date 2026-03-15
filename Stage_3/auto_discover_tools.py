"""
Auto-discover tools.

Scans Stage_3/tools/tool_*.py for BaseTool subclasses and registers them
with the tool registry.

To add a new tool, drop a tool_<name>.py file into Stage_3/tools/.
"""

import importlib
import inspect
import logging
import sys
from pathlib import Path

from Stage_3.BaseTool import BaseTool

logger = logging.getLogger("Discovery")


def discover(root_dir: Path, tool_registry, config: dict, reload: bool = False):
    import time
    t0 = time.time()
    count = 0
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
                    tool_registry.register(cls())
                    count += 1
                except Exception as e:
                    logger.error(f"Could not register tool {cls.__name__}: {e}", exc_info=True)
    logger.info(f"Discovered {count} tool(s) in {time.time() - t0:.2f}s")

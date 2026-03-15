"""
Auto-discover services.

Scans Stage_0/services/ for modules that expose a build_services(config)
function and collects all returned service instances.

To add a new service, drop a file into Stage_0/services/ and add a
module-level build_services(config) -> dict function.
"""

import importlib
import logging
from pathlib import Path

logger = logging.getLogger("Discovery")


def discover(root_dir: Path, config: dict) -> dict:
    import time
    t0 = time.time()
    services = {}
    services_dir = root_dir / "Stage_0" / "services"
    for py_file in sorted(services_dir.glob("*.py")):
        if py_file.stem.startswith("_"):
            continue
        module_name = f"Stage_0.services.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Could not import {module_name}: {e}")
            continue
        except Exception as e:
            logger.error(f"Failed to load {module_name}: {e}", exc_info=True)
            continue
        build_fn = getattr(module, "build_services", None)
        if build_fn is None:
            continue
        try:
            built = build_fn(config)
            if built:
                services.update(built)
                logger.debug(f"Services from {py_file.stem}: {list(built.keys())}")
        except Exception as e:
            logger.error(f"build_services() in {module_name} failed: {e}", exc_info=True)
    logger.info(f"Discovered {len(services)} service(s) in {time.time() - t0:.2f}s")
    return services

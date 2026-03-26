"""
Auto-discover services.

Scans Stage_0/services/ for modules that expose a build_services(config)
function (baked-in), then scans SANDBOX_SERVICES for agent-created services.
Collects all returned service instances.

Name collisions between sandbox and baked-in services are rejected.
"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

from paths import SANDBOX_SERVICES

logger = logging.getLogger("Discovery")


def discover(root_dir: Path, config: dict) -> dict:
    import time
    t0 = time.time()
    services = {}
    baked_in_names = set()

    # --- Baked-in services ---
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
                for svc_name, svc in built.items():
                    svc._mutable = False
                    services[svc_name] = svc
                    baked_in_names.add(svc_name)
                logger.debug(f"Services from {py_file.stem}: {list(built.keys())}")
        except Exception as e:
            logger.error(f"build_services() in {module_name} failed: {e}", exc_info=True)

    # --- Sandbox services ---
    if SANDBOX_SERVICES.exists():
        for py_file in sorted(SANDBOX_SERVICES.glob("*.py")):
            if py_file.stem.startswith("_"):
                continue
            module_name = f"sandbox_services_{py_file.stem}"
            try:
                if module_name in sys.modules:
                    module = importlib.reload(sys.modules[module_name])
                else:
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
            except Exception as e:
                logger.error(f"Failed to load sandbox service {py_file.name}: {e}")
                continue
            build_fn = getattr(module, "build_services", None)
            if build_fn is None:
                continue
            try:
                built = build_fn(config)
                if built:
                    for svc_name, svc in built.items():
                        if svc_name in baked_in_names:
                            logger.warning(f"Sandbox service '{svc_name}' collides with baked-in — skipped")
                            continue
                        svc._mutable = True
                        svc._source_path = str(py_file)
                        services[svc_name] = svc
                    logger.debug(f"Sandbox services from {py_file.stem}: {list(built.keys())}")
            except Exception as e:
                logger.error(f"build_services() in {module_name} failed: {e}", exc_info=True)

    logger.info(f"Discovered {len(services)} service(s) in {time.time() - t0:.2f}s")
    return services

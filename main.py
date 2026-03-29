"""Unix-friendly entry point. Delegates to main.pyw which is the canonical source."""
import importlib.util, sys, pathlib

spec = importlib.util.spec_from_file_location("_main", pathlib.Path(__file__).parent / "main.pyw")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

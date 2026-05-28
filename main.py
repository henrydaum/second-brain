"""Unix-friendly entry point. Delegates to main.pyw which is the canonical source."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(Path(__file__).with_name("main.pyw"), run_name="__main__")

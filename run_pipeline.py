"""Entry point. Run from this directory:

    python run_pipeline.py
"""

from pathlib import Path

from core.pipeline import run_all


if __name__ == "__main__":
    run_all(Path(__file__).resolve().parent, rebuild=True)

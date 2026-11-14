"""Run both differential-drive evaluations."""

import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ["exp1_obs_init", "exp2_prior_init"]


def main():
    """Run both initialization panels and stop on failure."""
    env = dict(os.environ, MPLBACKEND="Agg")
    for script in SCRIPTS:
        result = subprocess.run([sys.executable, "-m", f"experiments.differential_drive.{script}"], cwd=ROOT, env=env)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

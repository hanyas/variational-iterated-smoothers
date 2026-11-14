"""Run the stochastic-volatility evaluations."""

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SCRIPTS = ["exp1_comparison.py", "exp2_sweep.py"]


def main():
    """Run both stochastic-volatility evaluations and stop on failure."""
    env = dict(os.environ, MPLBACKEND="Agg")
    for script in SCRIPTS:
        result = subprocess.run([sys.executable, script], cwd=HERE, env=env)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

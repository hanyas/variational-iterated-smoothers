"""Run the full linear-Gaussian experiment suite and write all outputs to ./outputs"""

import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    ("exp1_recovery.py", "exact recovery + modularity   -> table_linear_gaussian.tex"),
    ("exp2_proximal.py", "proximal / trust-region       -> fig_lg_convergence.pdf, fig_lg_damping.pdf"),
    ("exp3_interpolation.py", "beta-interpolation       -> fig_lg_interpolation.pdf"),
]


def main():
    env = dict(os.environ, MPLBACKEND="Agg")
    failures = []
    for script, blurb in SCRIPTS:
        print(f"\n{'=' * 74}\n# {script}  --  {blurb}\n{'=' * 74}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run([sys.executable, "-u", script], cwd=HERE, env=env).returncode
        print(f"  [{script} finished in {time.perf_counter() - t0:.1f}s, rc={rc}]")
        if rc != 0:
            failures.append(script)

    print(f"\nAll outputs in: {HERE / 'outputs'}")
    if failures:
        print("FAILED:", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()

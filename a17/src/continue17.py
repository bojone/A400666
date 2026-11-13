#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    while not (RESULTS / "main_d23.audit.txt").exists():
        if (RESULTS / "answer.txt").exists() or (RESULTS / "main_answer.txt").exists():
            print("EARLIER_CANDIDATE_D24_NOT_RESUMED", flush=True)
            return 0
        time.sleep(30)
    if (RESULTS / "stop").exists():
        print("D24_RESUME_ABORTED_BY_STOP", flush=True)
        return 0
    print("D24_RESUMED_AFTER_LOW_PROOF", flush=True)
    with (ROOT / "campaign17_main.log").open("a") as log:
        return subprocess.run([sys.executable, str(ROOT / "run17.py"), "--digits", "24"],
                              cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    sys.exit(main())

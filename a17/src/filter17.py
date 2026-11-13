#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

from residues17 import generate
from verify17 import audit_filter_digit, self_check

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BINARY = ROOT / "filter17"
SOURCE = ROOT / "filter17.cpp"


def build():
    if not BINARY.exists() or SOURCE.stat().st_mtime > BINARY.stat().st_mtime:
        subprocess.run(["g++", "-O3", "-march=native", "-fopenmp", str(SOURCE), "-o", str(BINARY)],
                       cwd=ROOT, check=True)
    with (RESULTS / "filter_selftest.log").open("w") as log:
        subprocess.run([str(BINARY), "selftest"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print((RESULTS / "filter_selftest.log").read_text(), end="", flush=True)


def run(d):
    if d < 7:
        _, candidates = audit_filter_digit(d)
    else:
        generate(d)
        name = f"filter_d{d:02d}"
        with (RESULTS / f"{name}.log").open("a") as log:
            log.write(f"\nCOMMAND {BINARY} run {d}\n")
            log.flush()
            rc = subprocess.run([str(BINARY), "run", str(d)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            raise RuntimeError(f"filter d={d} failed rc={rc}")
        _, candidates = audit_filter_digit(d)
    if candidates:
        answer = min(candidates)
        (RESULTS / "filter_answer.txt").write_text(
            f"a(17)={answer}\nfilter_digit_complete={d}\n"
            "All earlier filter intervals completed and audited; no smaller solution exists.\n")
        print(f"FILTER_CANDIDATE_FOUND d={d} k={answer}", flush=True)
        return answer
    return None


def status():
    for path in sorted(RESULTS.glob("filter_d*.state")):
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        d, _, groups = map(int, values["config"].split(":"))
        done = int(values["done"])
        print(f"{path.stem} groups={done}/{groups} fraction={done/groups:.4%} searched={values['searched']} hits={values['hits']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--digits", type=int, nargs="+", default=list(range(2, 24)))
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    for d in args.digits:
        if not 2 <= d <= 28:
            parser.error("filter supports digits 2..28")
    self_check()
    generate(18)
    build()
    for d in args.digits:
        if run(d) is not None:
            return 0
    print("FILTER_INTERVALS_COMPLETE_WITHOUT_SOLUTION", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"FILTER17_ERROR {error}", flush=True)
        raise

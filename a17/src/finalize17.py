#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

from verify17 import M17, audit_answer

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def candidates():
    values = []
    for name in ("main_answer.txt", "filter_answer.txt"):
        path = RESULTS / name
        if path.exists():
            fields = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
            values.append(int(fields["a(17)"]))
    return sorted(values)


def finish(answer):
    path = RESULTS / "answer.txt"
    temporary = RESULTS / "answer.tmp"
    temporary.write_text(
        f"a(17)={answer}\nc(17)={M17}\nfull_modulus_remainder=0\n"
        "minimality_audit=PENDING\n")
    temporary.replace(path)
    audit_answer()
    temporary.write_text(
        f"a(17)={answer}\nc(17)={M17}\nfull_modulus_remainder=0\n"
        "All lower intervals required for minimality have been completed and audited.\n")
    temporary.replace(path)
    print(f"A17_COMPLETE {answer}", flush=True)
    return 0


def filter_complete_through_23():
    return (RESULTS / "main_d23.audit.txt").exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        values = candidates()
        if values:
            answer = values[0]
            if answer < 10**24:
                rc = finish(answer)
                (RESULTS / "stop").touch()
                return rc
            if filter_complete_through_23():
                return finish(answer)
        if not args.watch:
            print("FINALIZATION_NOT_READY candidates=" + repr(values), flush=True)
            return 2
        if not values and filter_complete_through_23() and (RESULTS / "main_d24.audit.txt").exists():
            print("NO_SOLUTION_THROUGH_24", flush=True)
            return 2
        time.sleep(30)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import subprocess
import sys
import threading
from pathlib import Path

from verify import KNOWN, concat_mod, literal

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "oeis_results"


def odd_part(value):
    while value % 2 == 0:
        value //= 2
    return value


def worker(device):
    log_path = RESULTS / f"dev{device}.log"
    with log_path.open("a") as log:
        log.write(f"\nCOMMAND verify_oeis run {device}\n")
        log.flush()
        rc = subprocess.run([str(ROOT / "verify_oeis"), "run", str(device)],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"DEVICE_EXIT device={device} rc={rc}", flush=True)
    return rc


def read_state(path):
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def audit():
    assert KNOWN[1] == 2 and literal(1) == 1
    lines = ["n=1 a(1)=2 remainder=0 earlier_solutions=0 status=VERIFIED (2 is the least integer greater than 1)"]
    print("VERIFIED n=1 a(1)=2 (trivial minimality)")
    all_valid = True
    for n in range(2, 14):
        modulus = literal(n)
        known = KNOWN[n]
        remainder = concat_mod(known, modulus)
        assert remainder == 0, (n, known, remainder)
        lo, hi = n + 1, known
        total = hi - lo
        searched = 0
        for device in range(8):
            begin = lo + total * device // 8
            end = lo + total * (device + 1) // 8
            state = read_state(RESULTS / f"n{n:02d}_dev{device}.state")
            expected_config = f"{n}:{begin}:{end}:{odd_part(modulus)}"
            assert state["config"] == expected_config, (n, device, state["config"], expected_config)
            assert int(state["next_k"]) == end, (n, device, state["next_k"], end)
            searched += end - begin
            log = (RESULTS / f"dev{device}.log").read_text()
            assert f"SHARD_DONE n={n} device={device} begin={begin} end={end}" in log, (n, device)
        assert searched == known - n - 1
        early = set()
        for path in RESULTS.glob(f"n{n:02d}_dev*.early"):
            for line in path.read_text().splitlines():
                if line.startswith("n="):
                    fields = dict(field.split("=", 1) for field in line.split())
                    k = int(fields["k"])
                    assert n < k < known
                    assert concat_mod(k, modulus) == 0
                    early.add(k)
        if early:
            all_valid = False
            print(f"NOT_MINIMAL n={n} published={known} earlier={min(early)}")
            lines.append(f"n={n} a({n})={known} remainder=0 earlier_solutions={len(early)} earliest={min(early)} status=NOT_MINIMAL")
        else:
            print(f"VERIFIED n={n} a({n})={known} remainder=0 searched={searched} earlier_solutions=0")
            lines.append(f"n={n} a({n})={known} remainder=0 searched={searched} earlier_solutions=0 status=VERIFIED")
    for device in range(8):
        assert f"DEVICE_DONE device={device}" in (RESULTS / f"dev{device}.log").read_text()
    if all_valid:
        lines.append("ALL_FIRST_13_VERIFIED: divisibility and minimality confirmed")
    (RESULTS / "verification.txt").write_text("\n".join(lines) + "\n")
    print("ALL_FIRST_13_VERIFIED" if all_valid else "DISCREPANCY_FOUND")
    return 0 if all_valid else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        return audit()
    RESULTS.mkdir(exist_ok=True)
    threads = []
    results = [None] * 8
    for device in range(8):
        thread = threading.Thread(target=lambda d=device: results.__setitem__(d, worker(d)))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if any(results):
        print(f"FAILURES {results}", flush=True)
        return 1
    return audit()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import subprocess
import sys
import threading
from pathlib import Path

from verify import concat_mod

L = 823045260674080876
M15 = 123456789101112131415
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results15"


def command(device):
    begin, end = L * device // 8, L * (device + 1) // 8
    return [str(ROOT / "search15_d21_fast"), "run", f"d21_x{device}", str(device), str(begin), str(end),
            "65536", "16384"]


def worker(device):
    with (RESULTS / f"d21_x{device}.log").open("a") as log:
        cmd = command(device)
        log.write(f"\nCOMMAND {' '.join(cmd)}\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"DEVICE_EXIT device={device} rc={rc}", flush=True)
    return rc


def candidates():
    values = set()
    for path in RESULTS.glob("d21_x*.candidates"):
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                values.add(int(line.split()[0][2:]))
    return sorted(values)


def status():
    for device in range(8):
        path = RESULTS / f"d21_x{device}.state"
        if not path.exists():
            continue
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        begin, end = map(int, values["config"].split(":"))
        next_x = int(values["next_x"])
        print(f"d21_x{device} next={next_x}/{end} fraction={(next_x-begin)/(end-begin):.4%} hits={values['hits']}")
    values = candidates()
    if values:
        print("CANDIDATES", *values)


def audit():
    for device in range(8):
        begin, end = L * device // 8, L * (device + 1) // 8
        values = dict(line.split("=", 1) for line in (RESULTS / f"d21_x{device}.state").read_text().splitlines() if "=" in line)
        assert values["config"] == f"{begin}:{end}"
        assert int(values["next_x"]) == end
        assert f"DONE name=d21_x{device} next_x={end}" in (RESULTS / f"d21_x{device}.log").read_text()
    values = candidates()
    for k in values:
        assert 10**20 <= k < 10**21
        assert k % 15 in (0, 5)
        assert concat_mod(k, M15) == 0, k
        print(f"VERIFIED_D21_CANDIDATE k={k} remainder_c15=0")
    print(f"D21_SEARCH_COMPLETE candidates={len(values)}")
    if values:
        print(f"SMALLEST_D21_CANDIDATE={values[0]}; lower intervals still required for a(15)")
    else:
        print("NO_D21_SOLUTION")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    if args.audit:
        return audit()
    RESULTS.mkdir(exist_ok=True)
    results = [None] * 8
    threads = [threading.Thread(target=lambda d=device: results.__setitem__(d, worker(d))) for device in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(results):
        print(f"FAILURES {results}", flush=True)
        return 1
    return audit()


if __name__ == "__main__":
    sys.exit(main())

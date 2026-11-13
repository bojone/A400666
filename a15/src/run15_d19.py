#!/usr/bin/env python3
import argparse
import subprocess
import sys
import threading
from pathlib import Path

from prepare15_coset4 import audit as audit_coset4
from verify import concat_mod

L = 823045260674080876
M15 = 123456789101112131415
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results15"


def command(device):
    begin, end = L * device // 8, L * (device + 1) // 8
    return [str(ROOT / "search15_d19_fast"), "run", f"d19_x{device}", str(device), str(begin), str(end),
            "65536", "16384"]


def worker(device):
    with (RESULTS / f"d19_x{device}.log").open("a") as log:
        cmd = command(device)
        log.write(f"\nCOMMAND {' '.join(cmd)}\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"DEVICE_EXIT device={device} rc={rc}", flush=True)
    return rc


def candidates():
    values = set()
    for path in RESULTS.glob("d19_x*.candidates"):
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                values.add(int(line.split()[0][2:]))
    return sorted(values)


def status():
    done = 0
    total = 0
    for device in range(8):
        path = RESULTS / f"d19_x{device}.state"
        if not path.exists():
            continue
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        begin, end = map(int, values["config"].split(":"))
        next_x = int(values["next_x"])
        done += next_x - begin
        total += end - begin
        mode = values.get("mode", "serial")
        print(f"d19_x{device} next={next_x}/{end} fraction={(next_x-begin)/(end-begin):.4%} hits={values['hits']} mode={mode}")
    print(f"TOTAL_D19_PROGRESS {done}/{total} fraction={done/total:.4%}")
    values = candidates()
    if values:
        print("CANDIDATES", *values)


def audit():
    if any("mode=coset4" in (RESULTS / f"d19_x{d}.state").read_text() for d in range(8)):
        assert audit_coset4()
    for device in range(8):
        begin, end = L * device // 8, L * (device + 1) // 8
        values = dict(line.split("=", 1) for line in (RESULTS / f"d19_x{device}.state").read_text().splitlines() if "=" in line)
        assert values["config"] == f"{begin}:{end}"
        assert int(values["next_x"]) == end
        assert f"DONE name=d19_x{device} next_x={end}" in (RESULTS / f"d19_x{device}.log").read_text()
    values = candidates()
    for k in values:
        assert 10**18 <= k < 10**19
        assert k % 15 in (0, 5)
        assert concat_mod(k, M15) == 0, k
        print(f"VERIFIED_D19_CANDIDATE k={k} remainder_c15=0")
    print(f"D19_SEARCH_COMPLETE candidates={len(values)}")
    if values:
        print(f"SMALLEST_D19_CANDIDATE={values[0]}; lower intervals still required for a(15)")
    else:
        print("NO_D19_SOLUTION")
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


if (RESULTS / "d19_balance_jobs.txt").exists():
    from balance15_d19 import main, status

    def audit():
        from balance15_d19 import audit as audit_balanced
        assert audit_balanced()
        return 0


if __name__ == "__main__":
    sys.exit(main())

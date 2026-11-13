#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

Q = 3810394725342967
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHUNK = 65536
BLOCKS = 16384


def task(name, device, period_mode, digits, offset, step, period, x_begin, x_end, k_start, k_end):
    return dict(name=name, device=device, period_mode=period_mode, digits=digits,
                offset=offset, step=step, period=period, x_begin=x_begin, x_end=x_end,
                chunk=CHUNK, blocks=BLOCKS, k_start=k_start, k_end=k_end)


def make_tasks():
    queues = {i: [] for i in range(8)}
    mid = Q // 2
    for device, (begin, end) in enumerate(((0, mid), (mid, Q))):
        queues[device].append(task(f"d18_x{device}", device, 1, 18, 0, 1, Q, begin, end,
                                   10**17, 10**18 - 1))
    for offset, devices in ((0, (2, 3)), (2, (4, 5))):
        for device, (begin, end) in zip(devices, ((0, mid), (mid, Q))):
            queues[device].append(task(f"d17_a{offset}_x{device}", device, 1, 17, offset, 6, Q,
                                       begin, end, 10**16, 10**17 - 1))
    for d in range(2, 16):
        lo, hi = max(15, 10**(d - 1)), 10**d - 1
        for offset in (0, 2):
            begin = max(0, (lo - offset + 5) // 6)
            end = (hi - offset) // 6 + 1
            queues[6].append(task(f"d{d:02d}_a{offset}", 6, 0, d, offset, 6, 0,
                                  begin, end, lo, hi))
    for device, offset in ((7, 0), (6, 2)):
        lo, hi = 10**15, 10**16 - 1
        begin = max(0, (lo - offset + 5) // 6)
        end = (hi - offset) // 6 + 1
        queues[device].append(task(f"d16_a{offset}", device, 0, 16, offset, 6, 0,
                                   begin, end, lo, hi))
    return queues


def command(t):
    fields = ["run", t["name"], t["device"], t["period_mode"], t["digits"], t["offset"],
              t["step"], t["period"], t["x_begin"], t["x_end"], t["chunk"], t["blocks"],
              t["k_start"], t["k_end"]]
    return [str(ROOT / "search")] + [str(x) for x in fields]


def worker(device, tasks, failures):
    for t in tasks:
        log_path = RESULTS / f"{t['name']}.log"
        print(f"START device={device} task={t['name']} work={t['x_end']-t['x_begin']}", flush=True)
        with log_path.open("a") as log:
            log.write(f"\nCOMMAND {' '.join(command(t))}\n")
            log.flush()
            rc = subprocess.run(command(t), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            failures.append((t["name"], rc))
            print(f"FAIL device={device} task={t['name']} rc={rc}", flush=True)
            return
        print(f"FINISH device={device} task={t['name']}", flush=True)


def status():
    rows = []
    for path in sorted(RESULTS.glob("*.state")):
        values = {}
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        config = values.get("config", "").split(":")
        x_end = int(config[6]) if len(config) >= 7 else 0
        next_x = int(values.get("next_x", 0))
        rows.append((path.stem, next_x, x_end, next_x / x_end if x_end else 0,
                     int(values.get("hits", 0))))
    for name, next_x, x_end, fraction, hits in rows:
        print(f"{name:24s} {next_x:20d}/{x_end:<20d} {fraction:8.3%} hits={hits}")
    candidates = set()
    for path in RESULTS.glob("*.candidates"):
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                candidates.add(int(line.split()[0][2:]))
    if candidates:
        print("CANDIDATES", *sorted(candidates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    RESULTS.mkdir(exist_ok=True)
    queues = make_tasks()
    failures = []
    threads = [threading.Thread(target=worker, args=(d, queues[d], failures)) for d in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print(f"FAILURES {failures}", flush=True)
        return 1
    print("ALL SEARCH QUEUES FINISHED", flush=True)
    status()
    return 0


if __name__ == "__main__":
    sys.exit(main())

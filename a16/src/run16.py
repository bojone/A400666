#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading

from verify16 import L, M16, audit_answer, audit_digit, self_check

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BINARY = ROOT / "search16"
SOURCE = ROOT / "search16.cu"
CHUNK = 65536
BLOCKS = 512


def command(d, device):
    limit = min(L, (10**d - 1) // 4 + 1)
    begin = limit * device // 8
    end = limit * (device + 1) // 8
    name = f"d{d:02d}_x{device}"
    return [str(BINARY), "run", name, str(d), str(device), str(begin), str(end), str(CHUNK), str(BLOCKS)]


def shard_complete(d, device):
    name = f"d{d:02d}_x{device}"
    path = RESULTS / f"{name}.state"
    if not path.exists():
        return False
    limit = min(L, (10**d - 1) // 4 + 1)
    begin = limit * device // 8
    end = limit * (device + 1) // 8
    values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return (values.get("config") == f"{d}:{begin}:{end}" and int(values["next_x"]) == end
            and f"DONE name={name} next_x={end}" in (RESULTS / f"{name}.log").read_text())


def worker(d, device):
    name = f"d{d:02d}_x{device}"
    if shard_complete(d, device):
        print(f"TASK_ALREADY_COMPLETE d={d} device={device}", flush=True)
        return 0
    with (RESULTS / f"{name}.log").open("a") as log:
        cmd = command(d, device)
        log.write(f"\nCOMMAND {' '.join(cmd)}\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"TASK_EXIT d={d} device={device} rc={rc}", flush=True)
    return rc


def run_digit(d):
    print(f"DIGIT_START d={d}", flush=True)
    if all(shard_complete(d, device) for device in range(8)):
        pre, candidates = audit_digit(d)
    else:
        threads = []
        results = [None] * 8
        for device in range(8):
            thread = threading.Thread(target=lambda device=device: results.__setitem__(device, worker(d, device)))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        if any(results):
            raise RuntimeError(f"digit {d} worker failures: {results}")
        pre, candidates = audit_digit(d)
    summary = f"d={d}\npre_hits={pre}\nfull_candidates={len(candidates)}\n"
    if candidates:
        summary += "values=" + ",".join(map(str, candidates)) + "\n"
    (RESULTS / f"d{d:02d}.summary").write_text(summary)
    return candidates


def build():
    nvcc = os.environ.get("NVCC", "/usr/local/cuda-12.9/bin/nvcc")
    if not BINARY.exists() or SOURCE.stat().st_mtime > BINARY.stat().st_mtime:
        subprocess.run([nvcc, "-O3", "-std=c++17", "-arch=native", str(SOURCE), "-o", str(BINARY)],
                       cwd=ROOT, check=True)
    with (RESULTS / "selftest.log").open("w") as log:
        subprocess.run([str(BINARY), "selftest", "0"], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print((RESULTS / "selftest.log").read_text(), end="", flush=True)


def status():
    for path in sorted(RESULTS.glob("d*_x*.state")):
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        d, begin, end = map(int, values["config"].split(":"))
        next_x = int(values["next_x"])
        fraction = 1 if begin == end else (next_x - begin) / (end - begin)
        print(f"{path.stem} next={next_x}/{end} fraction={fraction:.4%} pre={values['pre_hits']} hits={values['hits']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--through", type=int, default=28)
    args = parser.parse_args()
    if not 2 <= args.through <= 28:
        parser.error("--through must be between 2 and 28; digit 29 uses a different period")
    if args.status:
        status()
        return 0
    if (RESULTS / "answer.txt").exists():
        return 0 if audit_answer() else 1
    self_check()
    build()
    for d in range(2, args.through + 1):
        candidates = run_digit(d)
        if candidates:
            answer = min(candidates)
            (RESULTS / "answer.txt").write_text(
                f"a(16)={answer}\nc(16)={M16}\nfull_modulus_remainder=0\n"
                "All lower digit intervals completed and audited; no smaller solution exists.\n")
            audit_answer()
            print(f"A16_COMPLETE {answer}", flush=True)
            return 0
    print(f"NO_SOLUTION_THROUGH_DIGIT_{args.through}", flush=True)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"CAMPAIGN16_ERROR {error}", flush=True)
        raise

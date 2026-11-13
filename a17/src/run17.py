#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading

from verify17 import M17, audit_answer, audit_main_digit, period, self_check

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BINARY = ROOT / "search17"
SOURCE = ROOT / "search17.cu"
CHUNK = 262144
BLOCKS = 32768


def command(d, a, shard):
    device = shard if a == 0 else shard + 4
    limit = min(period(d), (10**d - 1 - a) // 9 + 1)
    begin = limit * shard // 4
    end = limit * (shard + 1) // 4
    name = f"d{d:02d}_a{a}_x{device}"
    return [str(BINARY), "run", name, str(d), str(a), str(device), str(begin), str(end), str(CHUNK), str(BLOCKS)]


def shard_complete(d, a, shard):
    device = shard if a == 0 else shard + 4
    name = f"d{d:02d}_a{a}_x{device}"
    path = RESULTS / f"{name}.state"
    if not path.exists():
        return False
    limit = min(period(d), (10**d - 1 - a) // 9 + 1)
    begin = limit * shard // 4
    end = limit * (shard + 1) // 4
    values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return (values.get("config") == f"{d}:{a}:{begin}:{end}" and int(values["next_x"]) == end
            and f"DONE name={name} next_x={end}" in (RESULTS / f"{name}.log").read_text())


def worker(d, a, shard):
    device = shard if a == 0 else shard + 4
    name = f"d{d:02d}_a{a}_x{device}"
    if shard_complete(d, a, shard):
        print(f"TASK_ALREADY_COMPLETE d={d} a={a} device={device}", flush=True)
        return 0
    with (RESULTS / f"{name}.log").open("a") as log:
        cmd = command(d, a, shard)
        log.write(f"\nCOMMAND {' '.join(cmd)}\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"TASK_EXIT d={d} a={a} device={device} rc={rc}", flush=True)
    return rc


def run_digit(d):
    print(f"MAIN_DIGIT_START d={d}", flush=True)
    if all(shard_complete(d, a, shard) for a in (0, 8) for shard in range(4)):
        return audit_main_digit(d)[1]
    threads = []
    results = [None] * 8
    for a in (0, 8):
        for shard in range(4):
            thread = threading.Thread(target=lambda a=a, shard=shard: results.__setitem__(shard if a == 0 else shard + 4, worker(d, a, shard)))
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    if any(results):
        raise RuntimeError(f"digit {d} worker failures: {results}")
    candidates = audit_main_digit(d)[1]
    if candidates:
        answer = min(candidates)
        (RESULTS / "main_answer.txt").write_text(
            f"a(17)={answer}\nc(17)={M17}\nfull_modulus_remainder=0\n"
            f"main_digit_complete={d}\nlower_filter_audit_required=YES\n")
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
    for path in sorted(RESULTS.glob("d*_a*_x*.state")):
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        d, a, begin, end = map(int, values["config"].split(":"))
        next_x = int(values["next_x"])
        fraction = 1 if begin == end else (next_x - begin) / (end - begin)
        print(f"{path.stem} next={next_x}/{end} fraction={fraction:.4%} pre={values['pre_hits']} hits={values['hits']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--digits", type=int, nargs="+", default=[24])
    args = parser.parse_args()
    if args.status:
        status()
        return 0
    if (RESULTS / "answer.txt").exists():
        return 0 if audit_answer() else 1
    for d in args.digits:
        if not 18 <= d <= 28:
            parser.error("main search supports digits 18..28")
    self_check()
    build()
    for d in args.digits:
        candidates = run_digit(d)
        if candidates:
            print(f"MAIN_CANDIDATE_FOUND d={d} k={min(candidates)}", flush=True)
            return 0
    print("NO_MAIN_CANDIDATE", flush=True)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"CAMPAIGN17_ERROR {error}", flush=True)
        raise

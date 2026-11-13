#!/usr/bin/env python3
import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

from verify import concat_mod

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results15"
M15 = 123456789101112131415
LO, HI = 10**17, 10**18 - 1


def bounds(a):
    begin = max(0, (LO - a + 14) // 15)
    end = (HI - a) // 15 + 1
    return begin, end


CPU_RESULTS = ROOT / "cpu_results"


def cpu_state(a=0):
    path = CPU_RESULTS / f"d18_a{a}_shared.state"
    if not path.exists():
        return None
    values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    begin, end = bounds(a)
    assert values["config"] == f"18:{a}:{begin}:{end}:shared272"
    assert values["clean_checkpoint"] == "1"
    next_x = int(values["next_x"])
    assert begin <= next_x <= end
    assert values["stopped"] in ("0", "1")
    return values


def cpu_split(a):
    state = cpu_state(a)
    return bounds(a)[0] if state is None else int(state["next_x"])


def stop_cpu(a=0):
    if cpu_state(a) is None:
        return
    (CPU_RESULTS / f"d18_a{a}_shared.stop").touch()
    for _ in range(300):
        state = cpu_state(a)
        if state["stopped"] == "1":
            print(f"CPU18_CLEAN_PREFIX a={a} next={state['next_x']}/{bounds(a)[1]}", flush=True)
            return
        time.sleep(1)
    raise TimeoutError("shared CPU runner did not reach a clean stopped checkpoint")


def command(a, device):
    begin = cpu_split(a)
    end = bounds(a)[1]
    total = end - begin
    x0, x1 = begin + total * device // 8, begin + total * (device + 1) // 8
    return [str(ROOT / "search15_direct_signed"), "run", f"d18_a{a}_x{device}", str(device), "18", str(a),
            str(x0), str(x1), "65536", "16384"]


def worker(device):
    for a in (0, 5):
        stop_cpu(a)
        with (RESULTS / f"d18_a{a}_x{device}.log").open("a") as log:
            cmd = command(a, device)
            log.write(f"\nCOMMAND {' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"TASK_EXIT d18_a{a}_x{device} rc={rc}", flush=True)
        if rc:
            return rc
    return 0


def candidates():
    values = set()
    paths = list(RESULTS.glob("d18_a*_x*.candidates")) + [CPU_RESULTS / f"d18_a{a}.candidates" for a in (0, 5)]
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                values.add(int(line.split()[0][2:]))
    return sorted(values)


def status():
    for a in (0, 5):
        state = cpu_state(a)
        if state is not None:
            print(f"CPU d=18 a={a} shared_prefix={state['next_x']}/{bounds(a)[1]} stopped={state['stopped']}")
    for a in (0, 5):
        for device in range(8):
            path = RESULTS / f"d18_a{a}_x{device}.state"
            if not path.exists():
                continue
            values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
            begin, end = map(int, values["config"].split(":"))
            next_x = int(values["next_x"])
            print(f"d18_a{a}_x{device} fraction={(next_x-begin)/(end-begin):.4%} hits={values['hits']}")
    values = candidates()
    if values:
        print("CANDIDATES", *values)


def audit():
    for a in (0, 5):
        begin, end = bounds(a)
        split = cpu_split(a)
        if split > begin:
            state = cpu_state(a)
            assert state is not None and state["stopped"] == "1"
            print(f"CPU18_PREFIX_COMPLETE a={a} searched={split-begin} clean_stop=YES")
        total = end - split
        for device in range(8):
            x0, x1 = split + total * device // 8, split + total * (device + 1) // 8
            name = f"d18_a{a}_x{device}"
            values = dict(line.split("=", 1) for line in (RESULTS / f"{name}.state").read_text().splitlines() if "=" in line)
            assert values["config"] == f"{x0}:{x1}"
            assert int(values["next_x"]) == x1
            assert f"DONE name={name} next_x={x1}" in (RESULTS / f"{name}.log").read_text()
    values = candidates()
    for k in values:
        assert LO <= k <= HI
        assert k % 15 in (0, 5)
        assert concat_mod(k, M15) == 0, k
        print(f"VERIFIED_D18_CANDIDATE k={k} remainder_c15=0")
    print(f"D18_SEARCH_COMPLETE candidates={len(values)}")
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
    benchmark_result = RESULTS / "direct_odd_benchmark.txt"
    if not benchmark_result.exists() and (ROOT / "benchmark15_direct_odd.py").exists():
        subprocess.run([sys.executable, str(ROOT / "benchmark15_direct_odd.py")], cwd=ROOT, check=True)
    stop_cpu()
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

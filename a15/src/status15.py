#!/usr/bin/env python3
from pathlib import Path

from verify import concat_mod

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results15"
CPU = ROOT / "cpu_results"
L20 = 205761315168520219
M15 = 123456789101112131415
GPU_RATE = 5.133e11
CPU_RATE = 9.62e10


def hours(seconds):
    return f"{seconds/3600:.2f}h"


def gpu_status():
    seconds = {}
    for d in (20, 19, 21):
        if d == 19 and (RESULTS / "d19_balance_jobs.txt").exists():
            from balance15_d19 import status as balanced_status
            seconds[d] = balanced_status()
            continue
        remaining = 0
        rates = []
        for dev in range(8):
            path = RESULTS / f"d{d}_x{dev}.state"
            if not path.exists():
                continue
            values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
            rates.append(6.740e11 if values.get("mode") == "coset4" else GPU_RATE)
            begin, end = map(int, values["config"].split(":"))
            next_x = int(values["next_x"])
            remaining += end - next_x
            print(f"GPU d{d}_x{dev}: {(next_x-begin)/(end-begin):.3%}, hits={values['hits']}")
        if rates:
            seconds[d] = remaining / sum(rates)
            print(f"GPU d{d} remaining: {hours(seconds[d])}")
    from run15_d18 import bounds as bounds18, cpu_split
    d18_remaining = 0
    for a in (0, 5):
        begin = cpu_split(a)
        end = bounds18(a)[1]
        total = end - begin
        for dev in range(8):
            x0, x1 = begin + total * dev // 8, begin + total * (dev + 1) // 8
            path = RESULTS / f"d18_a{a}_x{dev}.state"
            next_x = x0
            if path.exists():
                values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
                assert values["config"] == f"{x0}:{x1}"
                next_x = int(values["next_x"])
                print(f"GPU d18_a{a}_x{dev}: {(next_x-x0)/(x1-x0):.3%}, hits={values['hits']}")
            d18_remaining += x1 - next_x
    seconds[18] = d18_remaining / (8 * 5.716e11)
    print(f"GPU d18 remaining: {hours(seconds[18])}")
    return sum(seconds.get(d, 0) for d in (18, 19, 20))


def cpu_status():
    rows = []
    for path in CPU.glob("d*_a*.state"):
        lines = path.read_text().splitlines()
        d, a, begin, end = map(int, lines[0].split("=", 1)[1].split(":")[:4])
        if path.name in ("d18_a0_shared.state", "d18_a5_shared.state"):
            values = dict(line.split("=", 1) for line in lines[1:] if "=" in line)
            assert lines[0] == f"config={d}:{a}:{begin}:{end}:shared272"
            assert values["clean_checkpoint"] == "1"
            done = int(values["next_x"]) - begin
            print(f"CPU d=18 a={a} shared_prefix={done}/{end-begin} stopped={values['stopped']}")
            continue
        threads = int(lines[0].split(":")[-1])
        total = end - begin
        done = 0
        for i, line in enumerate(lines[1:]):
            next_j = int(line.split("=", 1)[1])
            lo = begin + total * i // threads
            hi = begin + total * (i + 1) // threads
            done += max(0, min(hi, next_j) - lo)
        rows.append((path.stat().st_mtime, d, a, done, total))
    for _, d, a, done, total in sorted(rows)[-4:]:
        print(f"CPU d={d} a={a}: {done/total:.3%}")
    active = [row for row in rows if row[3] < row[4]]
    if active:
        _, d, a, done, total = max(active, key=lambda row: (row[1], row[2]))
        remaining = (total - done) / CPU_RATE
        print(f"CPU active d={d} a={a} estimated remaining: {hours(remaining)}")
        return remaining
    return 0


def candidates():
    values = set()
    paths = list(RESULTS.glob("d*_x*.candidates")) + list(CPU.glob("d*_a*.candidates"))
    for path in paths:
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                k = int(line.split()[0][2:])
                if k <= 15:
                    continue
                assert concat_mod(k, M15) == 0
                values.add(k)
    if values:
        print("CANDIDATES", *sorted(values))
    else:
        print("CANDIDATES none")


def campaign_status():
    path = ROOT / "campaign15.log"
    if path.exists():
        lines = path.read_text().splitlines()
        print("CAMPAIGN", lines[-1] if lines else "no log")


def main():
    gpu_seconds = gpu_status()
    from prepare15_coset4 import audit as audit_coset4
    audit_coset4()
    cpu_seconds = cpu_status()
    print(f"ESTIMATED_REMAINING_IF_NO_D19_SOLUTION {hours(max(gpu_seconds, cpu_seconds))}")
    candidates()
    campaign_status()


if __name__ == "__main__":
    main()

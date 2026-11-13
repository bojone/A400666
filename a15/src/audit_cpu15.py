#!/usr/bin/env python3
import argparse
from pathlib import Path

from verify import concat_mod

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "cpu_results"
M15 = 123456789101112131415
THREADS = 272


def bounds(d, a):
    lo = max(16, 10**(d - 1))
    hi = 10**d - 1
    begin = max(0, (lo - a + 14) // 15)
    end = (hi - a) // 15 + 1
    return begin, end


def audit_job(d, a):
    name = f"d{d:02d}_a{a}"
    begin, end = bounds(d, a)
    path = RESULTS / f"{name}.state"
    lines = path.read_text().splitlines()
    assert lines[0] == f"config={d}:{a}:{begin}:{end}:{THREADS}", (name, lines[0])
    assert len(lines) == THREADS + 1, (name, len(lines))
    for i, line in enumerate(lines[1:]):
        next_j = int(line.split("=", 1)[1])
        expected = begin + (end - begin) * (i + 1) // THREADS
        assert next_j == expected, (name, i, next_j, expected)
    candidates = set()
    hit_path = RESULTS / f"{name}.candidates"
    if hit_path.exists():
        for line in hit_path.read_text().splitlines():
            if line.startswith("k="):
                k = int(line.split()[0][2:])
                assert k > 15 and k % 15 in (0, 5)
                assert concat_mod(k, M15) == 0
                candidates.add(k)
    print(f"CPU_JOB_COMPLETE d={d} a={a} searched={end-begin} candidates={len(candidates)}")
    return end - begin, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", type=int, default=14)
    parser.add_argument("--extra", choices=["d15a0", "d16a0", "d17a0"], action="append", default=[])
    args = parser.parse_args()
    total = 0
    all_candidates = set()
    for d in range(2, args.through + 1):
        for a in (0, 5):
            count, values = audit_job(d, a)
            total += count
            all_candidates |= values
    for extra in args.extra:
        d, a = int(extra[1:3]), int(extra[-1])
        count, values = audit_job(d, a)
        total += count
        all_candidates |= values
    assert not all_candidates, sorted(all_candidates)
    report = f"CPU_AUDIT_OK through_digit={args.through} extras={args.extra} searched={total} earlier_solutions=0\n"
    (RESULTS / "audit.txt").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()

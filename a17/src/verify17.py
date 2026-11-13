#!/usr/bin/env python3
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FILTERS = ROOT / "filters"
P = 584538396786764503
R17 = 97423066131127417
M17 = 1234567891011121314151617


def combine(left, right, modulus):
    qa, s0a, s1a, la = left
    qb, s0b, s1b, lb = right
    return (qa * qb % modulus,
            (s0a * qb + s0b) % modulus,
            (s1a * qb + la * s0b + s1b) % modulus,
            la + lb)


def block_sums(q, length, modulus):
    result = (1, 0, 0, 0)
    block = (q % modulus, 1, 1, 1)
    while length:
        if length & 1:
            result = combine(result, block, modulus)
        length >>= 1
        if length:
            block = combine(block, block, modulus)
    return result


def concat_mod(k, modulus):
    remainder = 0
    start = 1
    while start <= k:
        end = min(k, start * 10 - 1)
        digits = len(str(start))
        q, s0, s1, length = block_sums(pow(10, digits, modulus), end - start + 1, modulus)
        remainder = (remainder * q + (start - 1) * s0 + s1) % modulus
        start *= 10
    return remainder


def literal(n):
    return int("".join(str(k) for k in range(1, n + 1)))


def self_check():
    assert M17 == 9 * 47 * 4993 * P == literal(17)
    for n in range(1, 101):
        assert concat_mod(n, M17) == literal(n) % M17, n
    for k in range(18, 1001):
        assert concat_mod(k, M17) != 0, k
    print("VERIFY17_SELFTEST_OK literal=100 earlier_scan_through=1000")


def period(d):
    if not 18 <= d <= 28:
        raise ValueError(d)
    if d % 2 == 0:
        return R17
    return 2 * R17


def check_candidate(k, d=None):
    assert k > 17 and k % 9 in (0, 8), k
    if d is not None:
        assert 10**(d - 1) <= k < 10**d, (d, k)
    assert concat_mod(k, P) == 0, ("large_prime", k)
    assert concat_mod(k, M17) == 0, ("full_modulus", k)


def audit_main_digit(d):
    candidates = set()
    pre = 0
    for a in (0, 8):
        limit = min(period(d), (10**d - 1 - a) // 9 + 1)
        for shard in range(4):
            device = shard if a == 0 else shard + 4
            begin = limit * shard // 4
            end = limit * (shard + 1) // 4
            name = f"d{d:02d}_a{a}_x{device}"
            path = RESULTS / f"{name}.state"
            values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
            assert values["config"] == f"{d}:{a}:{begin}:{end}", (name, values)
            assert int(values["next_x"]) == end, (name, values["next_x"], end)
            assert int(values["period"]) == period(d), (name, values["period"])
            assert f"DONE name={name} next_x={end}" in (RESULTS / f"{name}.log").read_text(), name
            pre += int(values["pre_hits"])
    for path in RESULTS.glob(f"d{d:02d}_a*_x*.candidates"):
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                k = int(line.split()[0][2:])
                check_candidate(k, d)
                candidates.add(k)
    candidates = sorted(candidates)
    report = f"d={d}\npre_hits={pre}\nfull_candidates={len(candidates)}\n"
    if candidates:
        report += "values=" + ",".join(map(str, candidates)) + "\n"
    (RESULTS / f"main_d{d:02d}.audit.txt").write_text(report)
    print(f"MAIN_D{d:02d}_COMPLETE pre_hits={pre} full_candidates={len(candidates)}", flush=True)
    return pre, candidates


def filter_data(d):
    lines = (FILTERS / f"d{d:02d}.res").read_text().splitlines()
    fields = dict(line.split("=", 1) for line in lines[:2])
    period_value = int(fields["period"])
    count = int(fields["count"])
    rows = [tuple(map(int, line.split())) for line in lines[2:] if line]
    assert len(rows) == count
    return period_value, rows


def bruteforce_filter_digit(d):
    lo = max(18, 10**(d - 1))
    hi = 10**d - 1
    candidates = []
    for k in range(lo, hi + 1):
        if concat_mod(k, M17) == 0:
            check_candidate(k, d)
            candidates.append(k)
    searched = hi - lo + 1
    report = f"d={d}\nmethod=bruteforce\nsearched={searched}\nfull_candidates={len(candidates)}\n"
    if candidates:
        report += "values=" + ",".join(map(str, candidates)) + "\n"
    (RESULTS / f"filter_d{d:02d}.audit.txt").write_text(report)
    print(f"BRUTEFORCE_D{d:02d}_COMPLETE searched={searched} full_candidates={len(candidates)}", flush=True)
    return searched, candidates


def audit_filter_digit(d):
    if d < 7:
        return bruteforce_filter_digit(d)
    period_value, rows = filter_data(d)
    groups = (len(rows) + 7) // 8
    path = RESULTS / f"filter_d{d:02d}.state"
    values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    assert values["config"] == f"{d}:{period_value}:{groups}", (d, values)
    assert int(values["done"]) == groups, (d, values["done"], groups)
    expected_hex = ((1 << groups) - 1).to_bytes((groups + 7) // 8, "little").hex()
    assert values["done_hex"] == expected_hex, d
    lo = max(17, 10**(d - 1))
    hi = 10**d - 1
    expected = sum(1 + (hi - k0) // period_value for k0, _ in rows if lo <= k0 <= hi)
    assert int(values["searched"]) == expected, (d, values["searched"], expected)
    candidates = set()
    path = RESULTS / f"filter_d{d:02d}.candidates"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                k = int(line.split()[0][2:])
                check_candidate(k, d)
                candidates.add(k)
    candidates = sorted(candidates)
    report = f"d={d}\nsearched={expected}\nfull_candidates={len(candidates)}\n"
    if candidates:
        report += "values=" + ",".join(map(str, candidates)) + "\n"
    (RESULTS / f"filter_d{d:02d}.audit.txt").write_text(report)
    print(f"FILTER_D{d:02d}_COMPLETE searched={expected} full_candidates={len(candidates)}", flush=True)
    return expected, candidates


def audit_answer():
    self_check()
    fields = dict(line.split("=", 1) for line in (RESULTS / "answer.txt").read_text().splitlines() if "=" in line)
    answer = int(fields["a(17)"])
    digits = len(str(answer))
    candidates = []
    if digits < 24:
        for d in range(2, digits + 1):
            _, values = audit_filter_digit(d) if d < 21 else audit_main_digit(d)
            if d < digits:
                assert not values, (d, values)
            candidates.extend(values)
    else:
        for d in range(24, digits + 1):
            _, values = audit_main_digit(d)
            if d < digits:
                assert not values, (d, values)
            candidates.extend(values)
        for d in range(2, 24):
            _, values = audit_filter_digit(d) if d < 21 else audit_main_digit(d)
            assert not values, (d, values)
    candidates = sorted(set(candidates))
    assert candidates and candidates[0] == answer, (answer, candidates)
    check_candidate(answer, digits)
    report = (f"a(17)={answer}\nc(17)={M17}\nfull_modulus_remainder=0\n"
              f"lower_digits_complete_through={digits}\nearlier_solutions=0\n"
              "minimum_of_all_saved_candidates=YES\nFINAL_A17_AUDIT_OK\n")
    (RESULTS / "final_independent_audit.txt").write_text(report)
    print(report, end="", flush=True)
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int)
    parser.add_argument("--main-digit", type=int)
    parser.add_argument("--filter-digit", type=int)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.candidate is not None:
        self_check()
        check_candidate(args.candidate)
        print(f"INDEPENDENT_CANDIDATE_OK k={args.candidate} full_modulus_remainder=0")
        print("CANDIDATE_ONLY_NO_MINIMALITY_CLAIM")
    elif args.main_digit is not None:
        self_check()
        audit_main_digit(args.main_digit)
    elif args.filter_digit is not None:
        self_check()
        audit_filter_digit(args.filter_digit)
    elif args.audit:
        audit_answer()
    else:
        self_check()


if __name__ == "__main__":
    main()

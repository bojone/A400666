#!/usr/bin/env python3
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
P1 = 1231026625769
P2 = 2507191691
L = 153878328221
M16 = 12345678910111213141516


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
    assert M16 == 4 * P1 * P2 == literal(16)
    for n in range(1, 101):
        assert concat_mod(n, M16) == literal(n) % M16, n
    for k in range(17, 1001):
        assert concat_mod(k, M16) != 0, k
    print("VERIFY16_SELFTEST_OK literal=100 earlier_scan_through=1000")


def x_limit(d):
    hi = 10**d - 1
    return min(L, hi // 4 + 1)


def audit_digit(d):
    limit = x_limit(d)
    pre = 0
    for device in range(8):
        begin = limit * device // 8
        end = limit * (device + 1) // 8
        name = f"d{d:02d}_x{device}"
        path = RESULTS / f"{name}.state"
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        assert values["config"] == f"{d}:{begin}:{end}", (name, values)
        assert int(values["next_x"]) == end, (name, values["next_x"], end)
        assert f"DONE name={name} next_x={end}" in (RESULTS / f"{name}.log").read_text(), name
        pre += int(values["pre_hits"])
    candidates = set()
    for path in RESULTS.glob(f"d{d:02d}_x*.candidates"):
        for line in path.read_text().splitlines():
            if not line.startswith("k="):
                continue
            fields = dict(field.split("=", 1) for field in line.split())
            k = int(fields["k"])
            assert max(17, 10**(d - 1)) <= k < 10**d, (d, k)
            assert k % 4 == 0
            assert concat_mod(k, P1) == 0, ("p1", k)
            assert concat_mod(k, P2) == 0, ("p2", k)
            assert concat_mod(k, M16) == 0, ("full", k)
            candidates.add(k)
    print(f"D{d:02d}_COMPLETE pre_hits={pre} full_candidates={len(candidates)}", flush=True)
    return pre, sorted(candidates)


def audit_answer():
    self_check()
    fields = dict(line.split("=", 1) for line in (RESULTS / "answer.txt").read_text().splitlines() if "=" in line)
    answer = int(fields["a(16)"])
    digits = len(str(answer))
    candidates = []
    for d in range(2, digits + 1):
        _, values = audit_digit(d)
        if d < digits:
            assert not values, (d, values)
        candidates.extend(values)
    candidates = sorted(set(candidates))
    assert candidates and candidates[0] == answer, (answer, candidates)
    assert answer > 16 and answer % 4 == 0
    assert concat_mod(answer, P1) == 0
    assert concat_mod(answer, P2) == 0
    assert concat_mod(answer, M16) == 0
    report = (f"a(16)={answer}\nc(16)={M16}\nfull_modulus_remainder=0\n"
              f"lower_digits_complete_through={digits}\nearlier_solutions=0\n"
              "minimum_of_all_saved_candidates=YES\nFINAL_A16_AUDIT_OK\n")
    (RESULTS / "final_independent_audit.txt").write_text(report)
    print(report, end="", flush=True)
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int)
    parser.add_argument("--digit", type=int)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.candidate is not None:
        self_check()
        k = args.candidate
        assert k > 16 and k % 4 == 0
        assert concat_mod(k, P1) == 0
        assert concat_mod(k, P2) == 0
        assert concat_mod(k, M16) == 0
        print(f"INDEPENDENT_CANDIDATE_OK k={k} full_modulus_remainder=0")
        print("CANDIDATE_ONLY_NO_MINIMALITY_CLAIM")
    elif args.digit is not None:
        self_check()
        audit_digit(args.digit)
    elif args.audit:
        audit_answer()
    else:
        self_check()


if __name__ == "__main__":
    main()

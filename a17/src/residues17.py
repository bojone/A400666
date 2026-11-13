#!/usr/bin/env python3
import argparse
from math import gcd
from pathlib import Path

from sympy import n_order
from verify17 import concat_mod

ROOT = Path(__file__).resolve().parent
FILTERS = ROOT / "filters"


def zero_residues(modulus, d):
    start = 10**(d - 1)
    q = pow(10, d, modulus)
    order = n_order(q, modulus)
    if order == 1:
        period = 2 * modulus
        residues = [k % period for k in range(start, start + period) if concat_mod(k, modulus) == 0]
        return period, sorted(set(residues))
    A = pow(1 - q, -1, modulus)
    B = -q * A * A % modulus
    U = (concat_mod(start - 1, modulus) - A * (start - 1) - B) * pow(q, -(start - 1), modulus) % modulus
    Z = U * pow(q, start, modulus) % modulus
    C = (A * start + B) % modulus
    D = A * order % modulus
    assert D
    period = order * modulus
    result = []
    for x in range(order):
        t = -(Z * pow(q, x, modulus) + A * x + C) * pow(D, -1, modulus) % modulus
        k = start + x + order * t
        assert start <= k < start + period
        assert concat_mod(k, modulus) == 0, (modulus, d, x, k)
        result.append(k % period)
    return period, sorted(set(result))


def merge(left, right):
    m, aa = left
    n, bb = right
    g = gcd(m, n)
    lcm = m // g * n
    inverse = pow(m // g, -1, n // g)
    output = set()
    for a in aa:
        for b in bb:
            if (b - a) % g == 0:
                output.add((a + m * ((b - a) // g * inverse % (n // g))) % lcm)
    return lcm, sorted(output)


def make_filter(d):
    period47, residues47 = zero_residues(47, d)
    period4993, residues4993 = zero_residues(4993, d)
    period, residues = merge(merge((period47, residues47), (period4993, residues4993)), (9, [0, 8]))
    start = max(17, 10**(d - 1))
    k_values = sorted(start + (r - start) % period for r in residues)
    rows = []
    for i, k0 in enumerate(k_values):
        next_k = k_values[i + 1] if i + 1 < len(k_values) else k_values[0] + period
        assert next_k > k0
        rows.append((k0, next_k - k0))
    return period, rows


def expected_count(d, period, rows):
    hi = 10**d - 1
    return sum(1 + (hi - k0) // period for k0, _ in rows if k0 <= hi)


def generate(d, force=False):
    if d < 7:
        print(f"FILTER_BRUTE_FORCE_ONLY d={d}", flush=True)
        return
    destination = FILTERS / f"d{d:02d}.res"
    audit = FILTERS / f"d{d:02d}.audit.txt"
    if destination.exists() and audit.exists() and not force:
        print(f"FILTER_ALREADY_GENERATED d={d}", flush=True)
        return
    period, rows = make_filter(d)
    lines = [f"period={period}", f"count={len(rows)}"]
    lines += [f"{k0} {gap}" for k0, gap in rows]
    temporary = destination.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(destination)
    direct_checks = 0
    hi = 10**d - 1
    for k0, _ in rows[:len(rows) if d == 18 else min(256, len(rows))]:
        if k0 > hi:
            continue
        assert k0 % 9 in (0, 8)
        assert concat_mod(k0, 47) == 0, (d, k0)
        assert concat_mod(k0, 4993) == 0, (d, k0)
        direct_checks += 1
    count = expected_count(d, period, rows)
    report = (f"d={d}\nperiod={period}\nresidue_classes={len(rows)}\n"
              f"direct_small_factor_checks={direct_checks}\nprimary_residue_formulas_verified=YES\n"
              f"exact_candidates={count}\n")
    audit.write_text(report)
    print(f"FILTER_GENERATED d={d} classes={len(rows)} period={period} candidates={count}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digits", type=int, nargs="+", default=list(range(2, 24)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    FILTERS.mkdir(exist_ok=True)
    for d in args.digits:
        if not 2 <= d <= 28:
            parser.error("filter generation supports digits 2..28")
        generate(d, args.force)


if __name__ == "__main__":
    main()

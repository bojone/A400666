#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

from verify17 import M17, P, concat_mod, literal

ROOT = Path(__file__).resolve().parent
KNOWN24 = 553943104744829340555260


def multiply(a, b, modulus):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) % modulus
             for j in range(3)] for i in range(3)]


def matrix_pow(matrix, exponent, modulus):
    result = [[int(i == j) for j in range(3)] for i in range(3)]
    while exponent:
        if exponent & 1:
            result = multiply(result, matrix, modulus)
        exponent >>= 1
        if exponent:
            matrix = multiply(matrix, matrix, modulus)
    return result


def concat_matrix(k, modulus):
    remainder = 0
    start = 1
    while start <= k:
        end = min(k, start * 10 - 1)
        q = pow(10, len(str(start)), modulus)
        step = [[q, 1, 0], [0, 1, 1], [0, 0, 1]]
        power = matrix_pow(step, end - start + 1, modulus)
        remainder = (power[0][0] * remainder + power[0][1] * start + power[0][2]) % modulus
        start = end + 1
    return remainder


def selftest():
    for n in range(1, 201):
        assert concat_matrix(n, M17) == literal(n) % M17, n
    rng = random.Random(40066617)
    values = [10**23, 10**24 - 1, KNOWN24]
    values += [rng.randrange(18, 10**24) for _ in range(100)]
    for n in values:
        assert concat_matrix(n, M17) == concat_mod(n, M17), n
    return "MATRIX17_SELFTEST_OK literal=200 independent_cross_checks=103"


def verify(k):
    remainder = concat_matrix(k, M17)
    block_remainder = concat_mod(k, M17)
    assert remainder == 0, (k, remainder)
    assert block_remainder == 0, (k, block_remainder)
    assert k > 17 and k % 9 in (0, 8)
    assert concat_matrix(k, P) == 0
    return [f"k={k}", "matrix_remainder_c17=0", "block_sums_remainder_c17=0",
            "INDEPENDENT_MATRIX17_VERIFIED full_modulus_remainder=0"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("values", nargs="*", type=int)
    args = parser.parse_args()
    lines = [selftest()]
    for k in args.values or [KNOWN24]:
        lines.extend(verify(k))
    report = "\n".join(lines) + "\n"
    if not args.values:
        (ROOT / "results" / "matrix_verification.txt").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()

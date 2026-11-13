#!/usr/bin/env python3
import argparse
import random

from verify import concat_mod, literal

M15 = 123456789101112131415
P = 8230452606740808761
KNOWN20 = 70685893246361615705


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
        assert concat_matrix(n, M15) == literal(n) % M15, n
    rng = random.Random(40066615)
    values = [10**19, 10**20 - 1, KNOWN20]
    values += [rng.randrange(16, 10**20) for _ in range(100)]
    for n in values:
        assert concat_matrix(n, M15) == concat_mod(n, M15), n
    print("MATRIX_SELFTEST_OK literal=200 independent_cross_checks=103")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("values", nargs="*", type=int)
    args = parser.parse_args()
    selftest()
    for k in args.values or [KNOWN20]:
        remainder = concat_matrix(k, M15)
        print(f"k={k} matrix_remainder_c15={remainder}")
        assert remainder == 0, (k, remainder)
        assert k % 15 in (0, 5)
        assert concat_matrix(k, P) == 0
        print(f"INDEPENDENT_MATRIX_VERIFIED k={k} remainder_c15=0")


if __name__ == "__main__":
    main()

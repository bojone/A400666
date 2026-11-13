#!/usr/bin/env python3
import sys
from pathlib import Path

M14 = 1234567891011121314
KNOWN = [None, 2, 6, 23, 1838, 15000, 6992, 2888377, 795420, 37366496,
         8803255110, 1462927131789, 36810773632272, 2879846318370304]


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
    for n in range(1, 101):
        assert concat_mod(n, M14) == literal(n) % M14, n
    for n in range(1, 14):
        modulus = literal(n)
        remainder = concat_mod(KNOWN[n], modulus)
        assert remainder == 0, (n, KNOWN[n], remainder)
    print("verified literal concatenation through n=100 and all 13 published OEIS values")


def candidates():
    values = set()
    for path in (Path(__file__).parent / "results").glob("*.candidates"):
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                values.add(int(line.split()[0][2:]))
    return sorted(values)


def main():
    self_check()
    values = [int(x) for x in sys.argv[1:]] or candidates()
    for k in values:
        remainder = concat_mod(k, M14)
        print(f"k={k} c(k)_mod_c(14)={remainder}")
        assert remainder == 0, k
    if values:
        print(f"verified {len(values)} candidate(s) modulo the full composite c(14); smallest={min(values)}")


if __name__ == "__main__":
    main()

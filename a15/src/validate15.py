#!/usr/bin/env python3
import random
import sympy as s

from verify import concat_mod

P = 8230452606740808761
M15 = 123456789101112131415
L20 = 205761315168520219
L19 = 823045260674080876


def sub(a, b):
    return (a - b) % P


def job(d, h, L):
    start = 10**(d - 1)
    q = pow(10, d, P)
    A = pow(1 - q, -1, P)
    B = -q * A * A % P
    prev = start - 1
    rprev = concat_mod(prev, P)
    U = (rprev - A * prev - B) * pow(q, -prev, P) % P
    den = A * h % P * L % P
    alpha = -U * pow(den, -1, P) % P
    beta = -pow(L, -1, P) % P
    gamma = -B * pow(den, -1, P) % P
    g = pow(q, h, P)
    return q, A, B, U, g, alpha, beta, gamma


def check_period(d, h, L, tmax, tests=64):
    q, A, B, U, g, alpha, beta, gamma = job(d, h, L)
    assert s.n_order(g, P) == L
    start = 10**(d - 1)
    hi = 10**d - 1
    jmin = (start + h - 1) // h
    jmax = hi // h
    assert jmax // L == tmax, (d, jmax // L, tmax)
    assert (L - 1) + tmax * L >= jmax
    den = A * h % P * L % P
    rng = random.Random(4006661500 + d)
    specials = [jmin, jmin + 1, jmax] + ([L - 1] if jmin <= L - 1 <= jmax else [])
    for j in specials + [rng.randrange(jmin, jmax + 1) for _ in range(tests)]:
        t, x = divmod(j, L)
        k = h * j
        remainder = (U * pow(g, j, P) + A * k + B) % P
        assert remainder == concat_mod(k, P), (d, h, j, remainder)
        tstar = (alpha * pow(g, x, P) + beta * x + gamma) % P
        assert (remainder - den * (t - tstar)) % P == 0, (d, h, j, tstar)
    print(f"PERIOD_FORMULA_OK d={d} h={h} L={L} tmax={tmax} tests={tests+5}")


def check_direct(d=18, tests=64):
    rng = random.Random(4006661518)
    for a in (0, 5):
        start = 10**(d - 1)
        q = pow(10, d, P)
        A = pow(1 - q, -1, P)
        B = -q * A * A % P
        prev = start - 1
        rprev = concat_mod(prev, P)
        U = (rprev - A * prev - B) * pow(q, -prev, P) % P
        g = pow(q, 15, P)
        jmin = (start - a + 14) // 15
        jmax = (10**d - 1 - a) // 15
        for j in [jmin, jmin + 1, jmax] + [rng.randrange(jmin, jmax + 1) for _ in range(tests)]:
            k = a + 15 * j
            remainder = (U * pow(q, k, P) + A * k + B) % P
            assert remainder == concat_mod(k, P), (d, a, j)
        print(f"DIRECT_FORMULA_OK d={d} a={a} tests={tests+3}")


def main():
    assert 15 * P == M15 and s.isprime(P)
    check_period(20, 1, L20, 486)
    check_period(19, 5, L19, 2)
    check_period(21, 5, L19, 243)
    check_direct()
    print("ALL_A15_FORMULAS_AND_BOUNDARIES_VERIFIED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import random

import sympy as s

from validate15 import L19, L20, P, job
from verify import concat_mod

Q = L20
assert L19 == 4 * Q


def initial_next(device, old_next):
    rlo, rhi = Q * device // 8, Q * (device + 1) // 8
    result = []
    for u in range(4):
        old_device = 2 * u + device // 4
        offset = Q * u + rlo
        done = max(0, old_next[old_device] - offset)
        result.append(rlo + min(done, rhi - rlo))
    return result


def check_mapping():
    rng = random.Random(400666154)
    for trial in range(100):
        old_next = [rng.randrange(L19 * d // 8, L19 * (d + 1) // 8 + 1) for d in range(8)]
        converted = [initial_next(d, old_next) for d in range(8)]
        assert sum(converted[d][u] - Q * d // 8 for d in range(8) for u in range(4)) == sum(
            old_next[d] - L19 * d // 8 for d in range(8))
        x_values = [0, L19 - 1] + [rng.randrange(L19) for _ in range(100)]
        for d in range(8):
            x_values.extend([L19 * d // 8, old_next[d], L19 * (d + 1) // 8 - 1])
        for x in x_values:
            u, r = divmod(x, Q)
            d = 8 * r // Q
            old_device = 8 * x // L19
            assert d // 4 == old_device % 2 and 2 * u + d // 4 == old_device
            assert (r < converted[d][u]) == (x < old_next[old_device])
    for d in range(8):
        assert [initial_next(d, [L19 * (j + 1) // 8 for j in range(8)])[u]
                for u in range(4)] == [Q * (d + 1) // 8] * 4
    print("COSET4_MIGRATION_MAPPING_OK: 100 random checkpoint mappings, boundaries, counts and membership")


def check_residues():
    q, A, B, U, g, alpha, beta, gamma = job(19, 5, L19)
    assert s.n_order(g, P) == L19
    h = pow(g, Q, P)
    assert s.n_order(h, P) == 4
    rng = random.Random(400666155)
    for x in [0, 1, Q - 1, Q, 2 * Q, 3 * Q, 4 * Q - 1] + [rng.randrange(L19) for _ in range(100)]:
        u, r = divmod(x, Q)
        z = alpha * pow(g, r, P) * pow(h, u, P) % P
        w = (beta * r + gamma + beta * Q * u) % P
        t = (z + w) % P
        expected = (alpha * pow(g, x, P) + beta * x + gamma) % P
        assert t == expected
        k = 5 * (x + L19 * t)
        if 10**18 <= k < 10**19:
            assert (concat_mod(k, P) == 0) == (t <= 2)
    print("COSET4_RESIDUES_OK: order(g^Q)=4, four-coset traces and hit criterion independently checked")


def coverage_limits(device, x_limit):
    rlo, rhi = Q * device // 8, Q * (device + 1) // 8
    result = []
    for u in range(4):
        offset = Q * u
        result.append(rlo if x_limit <= offset + rlo else min(rhi, x_limit - offset))
    return result


def check_pruning():
    rng = random.Random(400666156)
    for x_limit in [0, 1, Q - 1, Q, 2 * Q, L19 - 1, L19] + [rng.randrange(L19 + 1) for _ in range(100)]:
        limits = [coverage_limits(d, x_limit) for d in range(8)]
        for d in range(8):
            rlo, rhi = Q * d // 8, Q * (d + 1) // 8
            assert all(rlo <= limit <= rhi for limit in limits[d])
        for x in [0, L19 - 1, min(L19 - 1, x_limit), max(0, x_limit - 1)] + [rng.randrange(L19) for _ in range(20)]:
            u, r = divmod(x, Q)
            d = 8 * r // Q
            assert (r < limits[d][u]) == (x < x_limit)
    for k0 in [10**18, 5 * Q, 5 * L19, 10**19 - 5]:
        bound = min(L19, k0 // 5)
        for j in [0, k0 // 5 - 1] + [rng.randrange(k0 // 5) for _ in range(100)]:
            t, x = divmod(j, L19)
            assert t <= 2 and x < bound
    print("COSET4_PRUNING_OK: smaller candidates require x<min(4Q,k0/5); boundaries and coverage checked")


if __name__ == "__main__":
    check_mapping()
    check_residues()
    check_pruning()
    print("ALL_COSET4_FORMULAS_AND_MIGRATIONS_VERIFIED")

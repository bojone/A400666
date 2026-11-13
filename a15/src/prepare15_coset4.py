#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path

from validate15_coset4 import L19, Q, coverage_limits, initial_next
from verify import concat_mod

M15 = 123456789101112131415

RESULTS = Path(__file__).resolve().parent / "results15"


def values(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def prepare():
    snapshots = []
    for d in range(8):
        path = RESULTS / f"d19_x{d}.state"
        snapshot = RESULTS / f"d19_x{d}.state.serial.state"
        state = values(path)
        begin, end = L19 * d // 8, L19 * (d + 1) // 8
        assert state["config"] == f"{begin}:{end}"
        if "mode" not in state:
            data = path.read_bytes()
            if snapshot.exists():
                assert snapshot.read_bytes() == data
            else:
                temporary = RESULTS / f"d19_x{d}.state.serial.tmp"
                temporary.write_bytes(data)
                temporary.replace(snapshot)
        assert snapshot.exists()
        snapshots.append(values(snapshot))
    old_next = [int(state["next_x"]) for state in snapshots]
    report = ["mode=coset4", f"period={L19}", f"q={Q}"]
    for d, state in enumerate(snapshots):
        raw = (RESULTS / f"d19_x{d}.state.serial.state").read_bytes()
        report.append(f"serial{d}=next:{state['next_x']},hits:{state['hits']},sha256:{hashlib.sha256(raw).hexdigest()}")
    for d in range(8):
        report.append(f"new{d}_initial_r=" + ",".join(map(str, initial_next(d, old_next))))
    (RESULTS / "d19_coset4_migration.txt").write_text("\n".join(report) + "\n")
    audit()
    print("COSET4_SERIAL_CHECKPOINTS_SAVED_AND_MIGRATION_VERIFIED")


def candidate_limit():
    best = None
    for path in RESULTS.glob("d19_x*.candidates"):
        for line in path.read_text().splitlines():
            if not line.startswith("k="):
                continue
            if " independent_remainder_p=0 k_mod_15=" not in line:
                continue
            k = int(line.split()[0][2:])
            assert 10**18 <= k < 10**19 and k % 15 in (0, 5)
            assert concat_mod(k, M15) == 0, k
            best = k if best is None else min(best, k)
    return L19 if best is None else min(L19, best // 5)


def audit():
    if (RESULTS / "d19_balance_jobs.txt").exists():
        from balance15_d19 import audit as audit_balanced
        return audit_balanced()
    snapshots = [values(RESULTS / f"d19_x{d}.state.serial.state") for d in range(8)]
    old_next = [int(state["next_x"]) for state in snapshots]
    for d, state in enumerate(snapshots):
        assert state["config"] == f"{L19*d//8}:{L19*(d+1)//8}"
        assert "mode" not in state
    initial = [initial_next(d, old_next) for d in range(8)]
    initial_total = sum(initial[d][u] - Q * d // 8 for d in range(8) for u in range(4))
    assert initial_total == sum(old_next[d] - L19 * d // 8 for d in range(8))
    x_limit = candidate_limit()
    current_total = 0
    complete = True
    modes = 0
    for d in range(8):
        state = values(RESULTS / f"d19_x{d}.state")
        begin, end = L19 * d // 8, L19 * (d + 1) // 8
        assert state["config"] == f"{begin}:{end}"
        rlo, rhi = Q * d // 8, Q * (d + 1) // 8
        if "mode" not in state:
            assert state["next_x"] == snapshots[d]["next_x"]
            current_total += int(state["next_x"]) - begin
            complete = False
            continue
        modes += 1
        assert state["mode"] == "coset4" and int(state["q"]) == Q
        assert int(state["r_begin"]) == rlo and int(state["r_end"]) == rhi
        stored_limit = int(state.get("x_limit", L19))
        assert stored_limit >= x_limit
        saved_limits = coverage_limits(d, stored_limit)
        required_limits = coverage_limits(d, x_limit)
        next_r = [int(state[f"next_r{u}"]) for u in range(4)]
        assert all(min(initial[d][u], saved_limits[u]) <= next_r[u] <= saved_limits[u] for u in range(4))
        done = sum(next_r[u] - rlo + rhi - saved_limits[u] for u in range(4))
        current_total += done
        assert int(state["next_x"]) == begin + (end - begin) * done // (4 * (rhi - rlo))
        complete &= (int(state["next_x"]) == end and all(next_r[u] >= required_limits[u] for u in range(4)))
    if x_limit == L19 and modes in (0, 8):
        assert current_total >= initial_total
    if complete:
        print(f"COSET4_AUDIT_COMPLETE: every possible smaller d19 candidate has x<{x_limit}; all required exponents covered")
    else:
        print(f"COSET4_CHECKPOINT_OK: covered_or_pruned={current_total}/{L19} fraction={current_total/L19:.4%}, x_limit={x_limit}")
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        audit()

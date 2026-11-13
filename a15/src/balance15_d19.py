#!/usr/bin/env python3
import argparse
import hashlib
import subprocess
import sys
import threading
from pathlib import Path

from validate15_coset4 import L19, Q, coverage_limits
from verify import concat_mod

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results15"
M15 = 123456789101112131415
RATE = 7.051e11


def values(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def spec_line(job):
    return (f"name={job['name']};device={job['device']};source={job['source']};"
            f"r_begin={job['begin']};r_end={job['end']};initial=" + ",".join(map(str, job["initial"])))


def jobs():
    result = []
    for line in (RESULTS / "d19_balance_jobs.txt").read_text().splitlines():
        if not line:
            continue
        fields = dict(field.split("=", 1) for field in line.split(";"))
        job = {"name": fields["name"], "device": int(fields["device"]),
               "source": int(fields["source"]), "begin": int(fields["r_begin"]),
               "end": int(fields["r_end"]), "initial": tuple(map(int, fields["initial"].split(","))),
               "config": line}
        assert len(job["initial"]) == 4
        assert spec_line(job) == line
        result.append(job)
    return result


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


def source_states():
    result = []
    for device in range(8):
        path = RESULTS / f"d19_x{device}.state.pre_balance"
        state = values(path)
        assert state["config"] == f"{L19*device//8}:{L19*(device+1)//8}"
        assert state["mode"] == "coset4" and int(state["q"]) == Q
        assert int(state["r_begin"]) == Q * device // 8
        assert int(state["r_end"]) == Q * (device + 1) // 8
        assert int(state["x_limit"]) == L19
        next_r = tuple(int(state[f"next_r{u}"]) for u in range(4))
        assert len(set(next_r)) == 1
        result.append((state, next_r[0]))
    return result


def prepare():
    assert not (RESULTS / "d19_balance_jobs.txt").exists(), "balanced manifest already exists"
    assert not (RESULTS / "d19_balance_migration.txt").exists(), "balanced migration already exists"
    assert candidate_limit() == L19, "do not rebalance after a 19-digit candidate appears"
    states = []
    for device in range(8):
        path = RESULTS / f"d19_x{device}.state"
        state = values(path)
        assert state["config"] == f"{L19*device//8}:{L19*(device+1)//8}"
        assert state["mode"] == "coset4" and int(state["q"]) == Q
        assert int(state["r_begin"]) == Q * device // 8
        assert int(state["r_end"]) == Q * (device + 1) // 8
        assert int(state["x_limit"]) == L19
        next_r = tuple(int(state[f"next_r{u}"]) for u in range(4))
        assert len(set(next_r)) == 1, "unaligned source checkpoint"
        assert int(state["r_begin"]) <= next_r[0] <= int(state["r_end"])
        states.append((path, state, next_r[0]))
    for path, state, next_r in states:
        backup = path.with_name(path.name + ".pre_balance")
        assert not backup.exists(), backup
        temporary = backup.with_name(backup.name + ".tmp")
        temporary.write_bytes(path.read_bytes())
        temporary.replace(backup)
    assert source_states()
    work = [(device, next_r, int(state["r_end"])) for device, (_, state, next_r) in enumerate(states)
            if next_r < int(state["r_end"])]
    total = sum(end - begin for _, begin, end in work)
    assert total > 0
    assignments = []
    position = 0
    for device, begin, end in work:
        for gpu in range(8):
            target_begin, target_end = total * gpu // 8, total * (gpu + 1) // 8
            lo, hi = max(position, target_begin), min(position + end - begin, target_end)
            if lo < hi:
                assignments.append({"name": f"d19_xb{len(assignments)}", "device": gpu,
                                    "source": device, "begin": begin + lo - position,
                                    "end": begin + hi - position})
        position += end - begin
    assert position == total
    for job in assignments:
        job["initial"] = (job["begin"],) * 4
    manifest = ["mode=balanced", f"period={L19}", f"q={Q}", f"remaining_r={total}"]
    for device, (path, state, next_r) in enumerate(states):
        backup = path.with_name(path.name + ".pre_balance")
        manifest.append(f"source{device}=next_r:{next_r},sha256:{hashlib.sha256(backup.read_bytes()).hexdigest()}")
    atomic_write(RESULTS / "d19_balance_jobs.txt", "".join(spec_line(job) + "\n" for job in assignments))
    atomic_write(RESULTS / "d19_balance_migration.txt", "\n".join(manifest) + "\n")
    print(f"BALANCED_MANIFEST_PREPARED jobs={len(assignments)} remaining_r={total}")
    audit()
    for gpu in range(8):
        selected = [job for job in assignments if job["device"] == gpu]
        remaining = sum(4 * (job["end"] - job["begin"]) for job in selected)
        print(f"BALANCED_PLAN gpu={gpu} jobs={len(selected)} points={remaining} eta_hours={remaining/RATE/3600:.3f}")


def audit():
    sources = source_states()
    limit = candidate_limit()
    assigned = jobs()
    complete = True
    covered = 0
    for device, (old, old_next) in enumerate(sources):
        rlo, rhi = Q * device // 8, Q * (device + 1) // 8
        required = coverage_limits(device, limit)
        selected = sorted((job for job in assigned if job["source"] == device), key=lambda job: job["begin"])
        cursor = old_next
        for job in selected:
            assert job["begin"] == cursor and old_next <= job["begin"] < job["end"] <= rhi
            assert job["initial"] == (job["begin"],) * 4
            cursor = job["end"]
        assert cursor == rhi
        segments = [[] for _ in range(4)]
        for u in range(4):
            prefix_end = min(old_next, required[u])
            if prefix_end > rlo:
                segments[u].append((rlo, prefix_end))
        for job in selected:
            assert 0 <= job["device"] < 8
            path = RESULTS / f"{job['name']}.state"
            if path.exists():
                state = values(path)
                assert state["config"] == job["config"] and state["mode"] == "balanced"
                assert int(state["q"]) == Q and int(state["source"]) == device
                assert int(state["r_begin"]) == job["begin"] and int(state["r_end"]) == job["end"]
                stored_limit = int(state.get("x_limit", L19))
                assert stored_limit >= limit
                next_r = tuple(int(state[f"next_r{u}"]) for u in range(4))
                saved_limits = coverage_limits(device, stored_limit)
                saved_limits = [min(job["end"], max(job["begin"], value)) for value in saved_limits]
                for u in range(4):
                    assert min(job["begin"], saved_limits[u]) <= next_r[u] <= saved_limits[u]
                done = sum(next_r[u] - job["begin"] + job["end"] - saved_limits[u] for u in range(4))
                total = 4 * (job["end"] - job["begin"])
                expected_next = job["begin"] + (job["end"] - job["begin"]) * done // total
                assert int(state["next_x"]) == expected_next
            else:
                next_r = job["initial"]
            for u in range(4):
                searched_end = min(next_r[u], required[u])
                if searched_end > job["begin"]:
                    segments[u].append((job["begin"], searched_end))
        for u in range(4):
            cursor = rlo
            for begin, end in sorted(segments[u]):
                if begin > cursor:
                    complete = False
                    cursor = begin
                if end > cursor:
                    covered += end - cursor
                    cursor = end
            covered += max(0, rhi - max(rlo, required[u]))
            complete &= cursor >= required[u]
    for job in assigned:
        if complete:
            assert int(values(RESULTS / f"{job['name']}.state")["next_x"]) == job["end"]
            assert f"DONE name={job['name']} next_x={job['end']}" in (RESULTS / f"{job['name']}.log").read_text()
    found = candidates()
    if complete:
        assert covered == L19
        print(f"BALANCED_D19_AUDIT_COMPLETE covered_or_pruned={covered}/{L19} candidates={found}")
    else:
        assert covered <= L19
        print(f"BALANCED_D19_CHECKPOINT_OK covered_or_pruned={covered}/{L19} fraction={covered/L19:.4%} x_limit={limit}")
    return complete


def candidates():
    result = set()
    for path in RESULTS.glob("d19_x*.candidates"):
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                k = int(line.split()[0][2:])
                assert 10**18 <= k < 10**19 and k % 15 in (0, 5)
                assert concat_mod(k, M15) == 0, k
                result.add(k)
    return sorted(result)


def status():
    sources = source_states()
    assigned = jobs()
    limit = candidate_limit()
    eta = 0.0
    for gpu in range(8):
        remaining = 0
        for job in (job for job in assigned if job["device"] == gpu):
            path = RESULTS / f"{job['name']}.state"
            next_r = job["initial"] if not path.exists() else tuple(
                int(values(path)[f"next_r{u}"]) for u in range(4))
            source = job["source"]
            limits = coverage_limits(source, limit)
            limits = [min(job["end"], max(job["begin"], value)) for value in limits]
            remaining += sum(max(0, limits[u] - min(next_r[u], limits[u])) for u in range(4))
        seconds = remaining / RATE
        eta = max(eta, seconds)
        print(f"GPU balanced_d19 device={gpu} remaining={remaining} eta={seconds/3600:.3f}h")
    print(f"GPU balanced_d19 makespan={eta/3600:.3f}h")
    return eta


def command(job):
    return [str(ROOT / "search15_balanced"), "run", job["name"], str(job["device"]), "65536", "16384"]


def worker(gpu, assigned, results):
    results[gpu] = 0
    for job in (job for job in assigned if job["device"] == gpu):
        with (RESULTS / f"{job['name']}.log").open("a") as log:
            cmd = command(job)
            log.write(f"\nCOMMAND {' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"DEVICE_JOB_EXIT device={gpu} name={job['name']} rc={rc}", flush=True)
        if rc:
            results[gpu] = rc
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return 0
    if args.audit:
        return 0 if audit() else 1
    if args.status:
        status()
        audit()
        return 0
    assigned = jobs()
    assert assigned
    results = [None] * 8
    threads = [threading.Thread(target=lambda d=gpu: worker(d, assigned, results)) for gpu in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(results):
        print(f"FAILURES {results}", flush=True)
        return 1
    return 0 if audit() else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import contextlib
import hashlib
import io
import time
from datetime import datetime, timezone
from pathlib import Path

import status15
from verify import concat_mod

ROOT = Path(__file__).resolve().parent
DESTINATIONS = ROOT / "results15" / "snapshots"
EXPERIMENT_ROOT = ROOT.parent / "experiments"
EXPERIMENT_SOURCES = {
    "benchmark15_direct_odd.py", "search15_coarse.cu", "search15_direct.cu", "search15_direct_counter.cu",
    "search15_direct_pair.cu", "search15_grouped.cu", "search15_lazyfilter.cu", "search15_minmod.cu",
    "search15_negstep.cu", "search15_pair18.cu", "search15_signed.cu",
}
SOURCES = [
    "audit15_final.py", "audit_cpu15.py", "balance15_d19.py", "campaign15.py", "cpu15_avx.cpp", "cpu15_shared.cpp",
    "prepare15_coset4.py", "run15_d18.py", "run15_d19.py", "run15_d20_fast.py", "benchmark15_direct_odd.py",
    "search15_balanced.cu", "search15_block.cu", "search15_coarse.cu", "search15_direct.cu", "search15_direct_odd.cu", "search15_direct_counter.cu", "search15_lazyfilter.cu",
    "search15_minmod.cu", "search15_negstep.cu", "search15_direct_pair.cu", "search15_grouped.cu", "search15_pair18.cu", "search15_period_fast.cu", "search15_prune.cu", "search15_signed.cu",
    "snapshot15.py", "status15.py", "transition15_a5.py",
    "validate15.py", "validate15_coset4.py", "verify.py", "verify15_matrix.py",
]


def source_path(name):
    base = EXPERIMENT_ROOT if name in EXPERIMENT_SOURCES else ROOT
    return base / name


def relative_path(path):
    if path.is_relative_to(ROOT):
        return path.relative_to(ROOT)
    return path.relative_to(ROOT.parent)


def copy_atomic(source, destination):
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)


def copy_log_tail_atomic(source, destination, limit=262144):
    size = source.stat().st_size
    with source.open("rb") as input_file:
        if size > limit:
            input_file.seek(-limit, 2)
            data = input_file.read(limit)
            data = data.partition(b"\n")[2]
        else:
            data = input_file.read()
    if data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1]
    header = f"SNAPSHOT_LOG_TAIL original_bytes={size} retained_bytes={len(data)}\n".encode()
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(header + data)
    temporary.replace(destination)
    return size, len(data)


def candidates():
    paths = list((ROOT / "results15").glob("d*_x*.candidates"))
    paths += list((ROOT / "cpu_results").glob("d*_a*.candidates"))
    values = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                k = int(line.split()[0][2:])
                if k <= 15:
                    continue
                assert concat_mod(k, status15.M15) == 0, (path, k)
                values.add(k)
    return sorted(values)


def snapshot():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = DESTINATIONS / stamp
    assert not destination.exists(), destination
    files = [source_path(name) for name in SOURCES]
    files += [ROOT.parent / "campaign15.log", ROOT.parent / "cpu_manager.log"]
    for directory, patterns in [
        (ROOT / "results15", ["d*_x*.state", "d*_a*_x*.state", "d*_x*.state.serial.state",
                              "d*_x*.candidates", "d*_a*_x*.candidates", "*.audit.txt",
                              "d19_coset4_migration.txt", "d19_balance*.txt", "d19_x*.pre_balance",
                              "speed_history.log", "known*.txt", "balanced*_verification.txt",
                              "signed*_verification.txt", "direct_odd_*.txt",
                              "answer.txt", "final_independent_audit.txt", "final_independent_audit.log"]),
        (ROOT / "cpu_results", ["d*_a*.state", "d*_a*.candidates", "*.audit.txt"]),
    ]:
        for pattern in patterns:
            files.extend(directory.glob(pattern))
    log_files = list((ROOT / "results15").glob("d*.log"))
    log_files += list((ROOT / "results15").glob("cpu15_*.log"))
    log_files += list((ROOT / "results15").glob("search15_*.log"))
    log_files += list((ROOT / "cpu_results").glob("d*.log"))
    log_files += list(ROOT.parent.glob("campaign*.log")) + list(ROOT.parent.glob("run15*.log"))
    log_files = sorted(set(log_files).difference(files))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status15.main()
    for name in sorted(set(files)):
        if not name.is_file():
            continue
        if name.parent in (ROOT, EXPERIMENT_ROOT) and name.suffix in {".py", ".cu", ".cpp"}:
            target = destination / "source" / name.name
        else:
            target = destination / relative_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_atomic(name, target)
    tail_records = []
    for name in log_files:
        if not name.is_file():
            continue
        relative = relative_path(name)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        original_bytes, retained_bytes = copy_log_tail_atomic(name, target)
        tail_records.append(f"{relative} original_bytes={original_bytes} retained_bytes={retained_bytes}")
    (destination / "log_tails.txt").write_text("\n".join(tail_records) + "\n")
    for name in SOURCES:
        assert (destination / "source" / name).read_bytes() == source_path(name).read_bytes()
    (destination / "status.txt").write_text(output.getvalue())
    entries = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(f"{digest}  {path.relative_to(destination)}")
    (destination / "SHA256SUMS").write_text("\n".join(entries) + "\n")
    for line in (destination / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    found = candidates()
    message = f"{stamp} files={len(entries)} candidates={found} hourly_checkpoint=OK\n"
    with (DESTINATIONS / "snapshot.log").open("a") as log:
        log.write(message)
    print(message, end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=3600)
    args = parser.parse_args()
    snapshot()
    while not args.once:
        time.sleep(args.interval)
        snapshot()


if __name__ == "__main__":
    main()

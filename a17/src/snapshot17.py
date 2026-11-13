#!/usr/bin/env python3
import argparse
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

from verify17 import check_candidate

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DESTINATIONS = RESULTS / "snapshots"
SOURCES = [
    "run17.py", "search17.cu", "verify17.py", "verify17_matrix.py",
    "filter17.py", "filter17.cpp", "residues17.py", "continue17.py",
    "finalize17.py", "snapshot17.py", "report17.py",
]


def copy_atomic(source, destination, tail=None):
    data = source.read_bytes()
    if tail is not None and len(data) > tail:
        data = data[-tail:]
        data = data.partition(b"\n")[2]
    temporary = destination.with_name(destination.name + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(data)
    temporary.replace(destination)


def saved_candidates():
    answer = set()
    for path in RESULTS.glob("d*_a*_x*.candidates"):
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                k = int(line.split()[0][2:])
                check_candidate(k)
                answer.add(k)
    return sorted(answer)


def snapshot():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = DESTINATIONS / stamp
    assert not destination.exists(), destination
    files = []
    for pattern in ["d*_a*_x*.state", "d*_a*_x*.candidates", "*.audit.txt",
                    "speed_history.log", "orders17.audit.txt", "answer.txt",
                    "final_independent_audit.txt", "boundary_current.log", "boundary_current.state"]:
        files.extend(RESULTS.glob(pattern))
    for name in SOURCES:
        copy_atomic(ROOT / name, destination / "source" / name)
    for source in sorted(set(files)):
        if source.is_file():
            copy_atomic(source, destination / "results" / source.name)
    logs = sorted(set(RESULTS.glob("d*_a*_x*.log")) | set(ROOT.glob("*17*.log")))
    for source in logs:
        copy_atomic(source, destination / "logs" / source.name, tail=262144)
    status = ["timestamp=" + stamp, "binary_sha256=" + hashlib.sha256((ROOT / "search17").read_bytes()).hexdigest()]
    for source in sorted(RESULTS.glob("d*_a*_x*.state")):
        status.append(source.name + "\n" + source.read_text().rstrip())
    (destination / "status.txt").write_text("\n".join(status) + "\n")
    entries = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(f"{digest}  {path.relative_to(destination)}")
    (destination / "SHA256SUMS").write_text("\n".join(entries) + "\n")
    for line in entries:
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    message = f"{stamp} files={len(entries)} candidates={saved_candidates()} manifest=OK\n"
    with (DESTINATIONS / "snapshot.log").open("a") as log:
        log.write(message)
    print(message, end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=3600)
    args = parser.parse_args()
    while True:
        snapshot()
        if args.once or (RESULTS / "final_independent_audit.txt").exists():
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

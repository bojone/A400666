#!/usr/bin/env python3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from verify17 import period

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def values(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def latest_rate(path):
    rates = []
    for line in reversed(path.read_text().splitlines()):
        if not line.startswith("progress "):
            continue
        fields = {}
        try:
            for token in line.split()[1:]:
                key, value = token.split("=", 1)
                fields[key] = value
            rate = float(fields["rate"])
            float(fields["elapsed"])
            int(fields["pre"])
            int(fields["hits"])
            int(fields["next_x"].split("/")[0])
            int(fields["next_x"].split("/")[1])
        except (KeyError, ValueError):
            continue
        if rate > 0:
            rates.append(rate)
            if len(rates) == 16:
                break
    if not rates:
        return None
    rates.sort()
    middle = len(rates) // 2
    if len(rates) % 2:
        return rates[middle]
    return (rates[middle - 1] + rates[middle]) / 2


def active_positions():
    result = {}
    for path in RESULTS.glob("d*_a*_x*.state"):
        state = values(path)
        _, _, begin, end = map(int, state["config"].split(":"))
        next_x = int(state["next_x"])
        age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        if next_x < end and age < 300:
            result[path.name] = next_x
    return result


def main():
    sample = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    if sample < 0:
        raise SystemExit("sample seconds must be nonnegative")
    started = time.monotonic()
    start_positions = active_positions() if sample else {}
    if sample:
        time.sleep(sample)
    elapsed = time.monotonic() - started
    end_positions = active_positions() if sample else {}
    rates = []
    if sample:
        for name, next_x in end_positions.items():
            if name in start_positions and next_x > start_positions[name]:
                rates.append((next_x - start_positions[name]) / elapsed)
    print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    remaining = 0
    for d in range(21, 25):
        done = total = pre = hits = 0
        for a in (0, 8):
            for shard in range(4):
                device = shard + 4 * (a // 8)
                path = RESULTS / f"d{d:02d}_a{a}_x{device}.state"
                if path.exists():
                    state = values(path)
                    _, _, begin, end = map(int, state["config"].split(":"))
                    next_x = int(state["next_x"])
                    pre += int(state["pre_hits"])
                    hits += int(state["hits"])
                    age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
                    if next_x < end and age < 300 and not sample:
                        rate = latest_rate(path.with_suffix(".log"))
                        if rate is not None:
                            rates.append(rate)
                else:
                    begin = period(d) * shard // 4
                    end = period(d) * (shard + 1) // 4
                    next_x = begin
                done += next_x - begin
                total += end - begin
        assert total == 2 * period(d), (d, total, period(d))
        remaining += total - done
        print(f"d={d} completed={100 * done / total:.9f}% pre_hits={pre} "
              f"full_candidates={hits} remaining_points={total - done}")
    if rates:
        per_gpu = sum(rates) / len(rates)
        print(f"kernel_samples={len(rates)} mean_per_gpu={per_gpu:.10g} "
              f"aggregate_8gpu={8 * per_gpu:.10g}")
        print(f"est_remaining_hours={remaining / (8 * per_gpu) / 3600:.5f}")
        print("rate_source=" + ("state_delta" if sample else "logs"))
    else:
        print("kernel_samples=0 rate_unavailable=yes")
    snapshot_log = RESULTS / "snapshots" / "snapshot.log"
    lines = [line for line in snapshot_log.read_text().splitlines() if line] if snapshot_log.exists() else []
    print("latest_snapshot", lines[-1] if lines else "NONE")
    print("answer_exists", (RESULTS / "answer.txt").exists())


if __name__ == "__main__":
    main()

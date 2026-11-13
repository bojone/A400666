#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path

from run_all import make_tasks
from verify import M14, concat_mod, self_check

ROOT = Path(__file__).resolve().parent


def state_values(path):
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main():
    self_check()
    tasks = [task for queue in make_tasks().values() for task in queue]
    for task in tasks:
        name = task["name"]
        values = state_values(ROOT / "results" / f"{name}.state")
        expected_config = ":".join(str(task[key]) for key in
            ("period_mode", "digits", "offset", "step", "period", "x_begin", "x_end", "k_start", "k_end"))
        assert values["config"] == expected_config, name
        assert int(values["next_x"]) == task["x_end"], (name, values["next_x"], task["x_end"])
        log = (ROOT / "results" / f"{name}.log").read_text()
        assert f"DONE name={name} next_x={task['x_end']}" in log, name
    candidates = defaultdict(set)
    for path in (ROOT / "results").glob("*.candidates"):
        for line in path.read_text().splitlines():
            if not line.startswith("k="):
                continue
            fields = dict(field.split("=", 1) for field in line.split())
            k = int(fields["k"])
            assert k > 14 and k % 6 in (0, 2)
            assert concat_mod(k, M14) == 0, k
            candidates[len(str(k))].add(k)
    for digits, values in sorted(candidates.items()):
        print(f"digit interval {digits}: {len(values)} distinct solutions; smallest={min(values)}")
    all_values = sorted(set().union(*candidates.values())) if candidates else []
    assert all_values, "no solution below 10^18; run the prepared 19-digit fallback"
    print(f"ALL SHARDS COMPLETE; a(14)={all_values[0]}; independently verified c(k) mod c(14)=0")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import hashlib
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results15"
EXPECTED = {
    "search15_direct_signed": "0975586248710d39ea02005140341241b0c48a6198dd78664fcf1c8c6692e489",
    "search15_direct_odd_candidate": "f6e02bd31e729e4f9581e966244f8274ec12f8b75ef93d12a5e2edd8cd34e073",
}


def main():
    for name, expected in EXPECTED.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    rates = {name: [] for name in EXPECTED}
    with (RESULTS / "direct_odd_benchmark.log").open("a") as log:
        for repeat in range(2):
            for name in EXPECTED:
                result = subprocess.run(
                    [str(ROOT / name), "benchmark", "0", "8192"], cwd=ROOT,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
                log.write(f"ROUND {repeat} BINARY {name}\n{result.stdout}")
                log.flush()
                found = re.findall(r"^D18_BENCHMARK.*rate=([0-9.eE+]+)$", result.stdout, re.MULTILINE)
                assert len(found) == 4, result.stdout
                rates[name].extend(map(float, found))
    control = rates["search15_direct_signed"]
    candidate = rates["search15_direct_odd_candidate"]
    speedup = statistics.mean(candidate) / statistics.mean(control) - 1
    report = (f"D18_ODD_FIXED_BENCHMARK repeats_per_binary=8 blocks_per_launch=8192\n"
              f"control_mean={statistics.mean(control):.12g} candidate_mean={statistics.mean(candidate):.12g}\n"
              f"speedup_percent={100 * speedup:.9f}\n"
              f"control_rates={control}\ncandidate_rates={candidate}\n"
              f"control_sha256={EXPECTED['search15_direct_signed']}\n"
              f"candidate_sha256={EXPECTED['search15_direct_odd_candidate']}\n"
              "production_changed=NO\n")
    temporary = RESULTS / "direct_odd_benchmark.txt.tmp"
    temporary.write_text(report)
    temporary.replace(RESULTS / "direct_odd_benchmark.txt")
    print(report, end="", flush=True)


if __name__ == "__main__":
    main()

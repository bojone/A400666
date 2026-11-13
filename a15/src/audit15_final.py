#!/usr/bin/env python3
import argparse
from pathlib import Path

import audit_cpu15
import run15_d18
import run15_d19
import run15_d20_fast
from verify import concat_mod, literal
from verify15_matrix import concat_matrix

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results15"
M15 = 123456789101112131415


def candidates():
    values = set()
    paths = list(RESULTS.glob("d*_x*.candidates")) + list((ROOT / "cpu_results").glob("d*_a*.candidates"))
    for path in paths:
        for line in path.read_text().splitlines():
            if line.startswith("k="):
                k = int(line.split()[0][2:])
                if k > 15:
                    values.add(k)
    return sorted(values)


def verify_candidate(k):
    assert k > 15 and k % 15 in (0, 5), k
    assert concat_mod(k, M15) == 0, ("block_sums", k)
    assert concat_matrix(k, M15) == 0, ("matrix", k)
    print(f"INDEPENDENT_CANDIDATE_OK k={k} block_sums=0 matrix=0", flush=True)


def final_audit():
    assert literal(15) == M15
    print(f"INPUT_MODULUS_LITERAL_OK c(15)={M15}", flush=True)
    answer_path = RESULTS / "answer.txt"
    fields = dict(line.split("=", 1) for line in answer_path.read_text().splitlines() if "=" in line)
    answer = int(fields["a(15)"])
    values = candidates()
    assert values and answer == values[0], (answer, values)
    for k in values:
        verify_candidate(k)
    digits = len(str(answer))
    assert 18 <= digits <= 20, f"unsupported answer interval: {digits}"
    cpu_count = 0
    for d in range(2, 18):
        for a in (0, 5):
            count, found = audit_cpu15.audit_job(d, a)
            assert not found, (d, a, found)
            cpu_count += count
    print(f"FINAL_CPU_COVERAGE_OK searched={cpu_count} earlier_solutions=0", flush=True)
    if digits >= 18:
        assert run15_d18.audit() == 0
    if digits >= 19:
        assert run15_d19.audit() == 0
    if digits >= 20:
        assert run15_d20_fast.audit() == 0
    verify_candidate(answer)
    report = (f"a(15)={answer}\nblock_sums_remainder=0\nmatrix_remainder=0\n"
              f"cpu_searched={cpu_count}\nlower_intervals_complete_through_digit={digits}\n"
              "minimum_of_all_saved_candidates=YES\nFINAL_INDEPENDENT_AUDIT_OK\n")
    temporary = RESULTS / "final_independent_audit.txt.tmp"
    temporary.write_text(report)
    temporary.replace(RESULTS / "final_independent_audit.txt")
    print(report, end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int)
    args = parser.parse_args()
    if args.candidate is not None:
        verify_candidate(args.candidate)
        print("CANDIDATE_ONLY_NO_MINIMALITY_CLAIM")
    else:
        final_audit()


if __name__ == "__main__":
    main()

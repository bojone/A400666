#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

from audit_cpu15 import bounds as cpu_bounds
from verify import concat_mod

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results15"
M15 = 123456789101112131415


def log(message):
    print(message, flush=True)


def states_complete(prefix, shards):
    for suffix in shards:
        path = RESULTS / f"{prefix}{suffix}.state"
        if not path.exists():
            return False
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        end = int(values["config"].split(":")[1])
        if int(values["next_x"]) != end:
            return False
    return True


def wait_states(prefix, shards):
    log(f"WAIT {prefix} shards={len(shards)}")
    while not states_complete(prefix, shards):
        time.sleep(60)
    log(f"STATES_COMPLETE {prefix}")


def stop_phase(marker):
    log(f"STOP_PHASE {marker}")
    subprocess.run(["pkill", "-f", marker], check=False)
    for _ in range(120):
        if subprocess.run(["pgrep", "-f", marker], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode:
            log(f"PHASE_STOPPED {marker}")
            return
        time.sleep(1)
    raise RuntimeError(f"failed to stop {marker}")


def wait_candidate_or_states(prefix, shards, lo, hi, marker, process=None):
    log(f"WAIT candidate or completion {prefix}")
    while True:
        values = [k for k in candidates() if lo <= k < hi]
        if values:
            if not states_complete(prefix, shards):
                stop_phase(marker)
            log(f"CANDIDATE_FOUND {prefix} count={len(values)}")
            return True
        if states_complete(prefix, shards):
            log(f"NO_CANDIDATE {prefix}")
            return False
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"phase exited before completion: {prefix} rc={process.returncode}")
        time.sleep(60)


def resume_if_incomplete(script, prefix, shards, log_name):
    if not states_complete(prefix, shards):
        run_phase(script, log_name)
    audit_phase(script)


def run_phase(script, log_name):
    log(f"PHASE_START {script}")
    with (ROOT / log_name).open("a") as output:
        rc = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT,
                            stdout=output, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"{script} failed with rc={rc}")
    log(f"PHASE_DONE {script}")


def start_phase(script, log_name):
    log(f"PHASE_START {script}")
    output = (ROOT / log_name).open("a")
    return subprocess.Popen([sys.executable, str(ROOT / script)], cwd=ROOT,
                            stdout=output, stderr=subprocess.STDOUT)


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


def wait_cpu17():
    log("WAIT CPU d17")
    while True:
        complete = True
        for a in (0, 5):
            begin, end = cpu_bounds(17, a)
            path = ROOT / "cpu_results" / f"d17_a{a}.state"
            if not path.exists():
                complete = False
                break
            lines = path.read_text().splitlines()
            if len(lines) != 273:
                complete = False
                break
            for i, line in enumerate(lines[1:]):
                expected = begin + (end - begin) * (i + 1) // 272
                if int(line.split("=", 1)[1]) != expected:
                    complete = False
                    break
            if not complete:
                break
        if complete:
            log("CPU_D17_COMPLETE")
            return
        time.sleep(60)


def audit_phase(script):
    rc = subprocess.run([sys.executable, str(ROOT / script), "--audit"], cwd=ROOT).returncode
    if rc:
        raise RuntimeError(f"audit failed: {script}")


def main():
    d20_hit = wait_candidate_or_states("d20_x", range(8), 10**19, 10**20, "search15_b1024_signed run d20_x")
    if d20_hit:
        run_phase("run15_d19.py", "campaign_d19.log")
        d19_values = [k for k in candidates() if 10**18 <= k < 10**19]
        if not d19_values:
            resume_if_incomplete("run15_d20_fast.py", "d20_x", range(8), "campaign_d20_resume.log")
    else:
        process = start_phase("run15_d21.py", "campaign_d21.log")
        d21_hit = wait_candidate_or_states("d21_x", range(8), 10**20, 10**21,
                                           "search15_d21_fast run d21_x", process)
        rc = process.wait()
        if not d21_hit:
            if rc:
                raise RuntimeError(f"d21 failed with rc={rc}")
            log("NO_CANDIDATE_THROUGH_D21; additional research required")
            return 2
        run_phase("run15_d19.py", "campaign_d19.log")
        d19_values = [k for k in candidates() if 10**18 <= k < 10**19]
        if not d19_values:
            resume_if_incomplete("run15_d21.py", "d21_x", range(8), "campaign_d21_resume.log")
    run_phase("run15_d18.py", "campaign_d18.log")
    wait_cpu17()
    audit_phase("run15_d18.py")
    subprocess.run([sys.executable, str(ROOT / "audit_cpu15.py"), "--through", "17"], cwd=ROOT, check=True)
    values = candidates()
    if not values:
        log("NO_SOLUTION_FOUND")
        return 2
    answer = values[0]
    assert concat_mod(answer, M15) == 0
    report = (f"a(15)={answer}\nc(15)={M15}\nc(a(15)) mod c(15)=0\n"
              "All lower intervals completed and audited; no smaller solution exists.\n")
    (RESULTS / "answer.txt").write_text(report)
    log(f"A15_COMPLETE {answer}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"CAMPAIGN_ERROR {exc}")
        raise

#!/usr/bin/env python3
import argparse
import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import campaign15
import run15_d18

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results15"
AUDIT = RESULTS / "cpu18_a5_transition.audit.txt"


def phase(message):
    print(message, flush=True)
    previous = AUDIT.read_text() if AUDIT.exists() else ""
    temporary = RESULTS / "cpu18_a5_transition.audit.txt.tmp"
    temporary.write_text(previous + message + "\n")
    temporary.replace(AUDIT)


def command_exists(pattern):
    return subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def check_runner(pid):
    arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    assert str(ROOT / "run15_d18.py").encode() in arguments, (pid, arguments)


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    check_runner(args.pid)
    assert not any(RESULTS.glob("d18_a5_x*.state"))
    assert digest("search15_direct_signed") == "f6e02bd31e729e4f9581e966244f8274ec12f8b75ef93d12a5e2edd8cd34e073"
    assert digest("cpu15_shared_a5") == "a4636a3f62650c95493836d83fad283043d093cd0b1652120e6cb32f5c1cde69"
    assert digest("run15_d18.py") == "2d14fbd2571355200e59d0df9b84b505304f6ca4931bf78359e24d0e65826cc3"
    assert run15_d18.cpu_state(0)["stopped"] == "1"
    state = run15_d18.cpu_state(5)
    assert state is not None and state["stopped"] == "0" and state["clean_checkpoint"] == "1"
    if args.dry_run:
        print(f"A5_TRANSITION_DRY_RUN_OK old_runner={args.pid} cpu_prefix={state['next_x']}", flush=True)
        return
    os.kill(args.pid, signal.SIGSTOP)
    phase(f"OLD_RUNNER_SIGSTOP pid={args.pid} gpu_a0_continues=YES cpu_a5_continues=YES")
    while not campaign15.states_complete("d18_a0_x", range(8)):
        if not command_exists("search15_direct_signed run d18_a0_"):
            raise RuntimeError("a0 GPU processes exited before all checkpoints completed")
        time.sleep(15)
    phase("A0_ALL_CHECKPOINTS_COMPLETE")
    for _ in range(120):
        if not command_exists("search15_direct_signed run d18_a0_"):
            break
        time.sleep(1)
    else:
        raise RuntimeError("a0 GPU processes did not exit after completed checkpoints")
    assert not any(RESULTS.glob("d18_a5_x*.state"))
    check_runner(args.pid)
    os.kill(args.pid, signal.SIGKILL)
    phase(f"OLD_RUNNER_REMOVED pid={args.pid} no_a0_gpu_running=YES no_a5_gpu_started=YES")
    for _ in range(120):
        if not command_exists("campaign15.py$"):
            break
        time.sleep(1)
    else:
        raise RuntimeError("old campaign did not exit")
    assert campaign15.states_complete("d20_x", range(8))
    assert campaign15.states_complete("d18_a0_x", range(8))
    assert run15_d18.cpu_state(0)["stopped"] == "1"
    state = run15_d18.cpu_state(5)
    assert state is not None and state["clean_checkpoint"] == "1"
    assert digest("run15_d18.py") == "2d14fbd2571355200e59d0df9b84b505304f6ca4931bf78359e24d0e65826cc3"
    assert digest("search15_direct_signed") == "f6e02bd31e729e4f9581e966244f8274ec12f8b75ef93d12a5e2edd8cd34e073"
    phase(f"EXEC_NEW_CAMPAIGN cpu_a5_next={state['next_x']} cpu_a5_stopped={state['stopped']}")
    os.chdir(ROOT)
    output = (ROOT / "campaign15.log").open("a")
    os.dup2(output.fileno(), 1)
    os.dup2(output.fileno(), 2)
    os.execv(sys.executable, [sys.executable, "campaign15.py"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        phase(f"TRANSITION_ERROR {type(error).__name__}: {error}")
        raise

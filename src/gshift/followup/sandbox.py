"""Run untrusted (model-written) Python in a throwaway subprocess: isolated interpreter
(-I), fresh temp working directory, wall-clock timeout and address-space / CPU limits.
Used only on disposable pods or for trusted reference code in tests; it is a resource
guard, not a security boundary against a malicious program."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class RunResult:
    ok: bool              # exit code 0 within the timeout
    timeout: bool
    returncode: int | None
    stderr_tail: str


def _limits(mem_bytes: int, cpu_s: int):
    def f():
        import resource

        for lim, val in ((resource.RLIMIT_AS, mem_bytes), (resource.RLIMIT_CPU, cpu_s)):
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):   # e.g. RLIMIT_AS unsupported on macOS
                pass
    return f


def run_python(code: str, timeout_s: float = 20.0, mem_mb: int = 2048) -> RunResult:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(code)
        env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "HOME": d}
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, env=env, capture_output=True, text=True,
                               timeout=timeout_s, preexec_fn=_limits(mem_mb * 2**20, int(timeout_s) + 1))
            return RunResult(p.returncode == 0, False, p.returncode, p.stderr[-400:])
        except subprocess.TimeoutExpired:
            return RunResult(False, True, None, "timeout")

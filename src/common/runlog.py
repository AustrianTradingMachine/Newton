"""Pipeline-stage runner for the setup notebook.

Runs one scenario stage as a subprocess and does two things at once:

  * streams its output **live** (line by line, as it is produced) to the
    notebook / terminal, so a long GPU run can be watched in real time; and
  * records an **OK / ERR** health line plus the key result lines into
    ``logs/summary.txt`` (with the full per-stage log in ``logs/<slug>.log``).

It lives here, not inline in 00_setup.ipynb, so the notebook cell is a one-line
import (`from common.runlog import run`) and the helper can be unit-checked.

Live output matters: a child Python process **block-buffers** its stdout when it
is attached to a pipe (not a TTY), so without help its prints would only appear in
one chunk at the very end. We force line-by-line flushing by exporting
``PYTHONUNBUFFERED=1`` into the child's environment and flushing our own stdout
after every line.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

from common import params

# lines worth lifting into the one-file health report (logs/summary.txt)
_KEY = re.compile(
    r"(wrote |tip vertical drop|tip ratio|max .*displacement|residual RMS|"
    r"max rel|plateau|theta\*|deviation|F_max|F=|pen=|settled at frame|dolfinx )",
    re.I,
)

# label -> (ok, key_lines), kept in call order for the lifetime of the kernel session
_SUMMARY: dict[str, tuple[bool, list[str]]] = {}


def run(label: str, cmd: str) -> bool:
    """Run one pipeline stage; stream it live AND log it. Returns True on exit code 0.

    Re-running a stage updates (does not duplicate) its entry in logs/summary.txt.
    """
    os.makedirs(params.LOGS_DIR, exist_ok=True)
    slug = re.sub(r"\W+", "_", label).strip("_").lower()

    # PYTHONUNBUFFERED=1 -> the child flushes each print immediately, so we (and the
    # reader) see output as it happens instead of in a buffered chunk at the end.
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    lines: list[str] = []
    with open(os.path.join(params.LOGS_DIR, f"{slug}.log"), "w") as log:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        for line in proc.stdout:
            sys.stdout.write(line)        # live in the notebook / terminal
            sys.stdout.flush()
            log.write(line)
            lines.append(line)
        proc.wait()

    ok = proc.returncode == 0
    hits = (
        [ln.strip() for ln in lines if _KEY.search(ln)][-3:] if ok
        else [ln.strip() for ln in lines if ln.strip()][-6:]   # tail on failure
    )
    _SUMMARY[label] = (ok, hits)

    summary_path = os.path.join(params.LOGS_DIR, "summary.txt")
    with open(summary_path, "w") as fh:       # rewrite from memory -> no dups on re-run
        for lbl, (o, hs) in _SUMMARY.items():
            fh.write(f"[{'OK ' if o else 'ERR'}] {lbl}\n")
            fh.writelines(f"      {h}\n" for h in hs)

    sys.stdout.write(f"\n[{'OK ' if ok else 'ERR'}] {label}  ->  logged to {summary_path}\n")
    sys.stdout.flush()
    return ok

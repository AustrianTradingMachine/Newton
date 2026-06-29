"""One-command launcher: run the whole project on Kaggle via `kaggle kernels push`.

This is the *start* step that kagglehub cannot do. It:

  1. ships the code as a dataset  (kagglehub upload-src, unless --no-src),
  2. stages a push dir (the notebook + a generated kernel-metadata.json that
     attaches that dataset and enables GPU + Internet),
  3. `kaggle kernels push`  -> Kaggle runs 00_setup_kaggle.ipynb on a GPU,
  4. optionally polls status and pulls the outputs (--watch).

Auth (required): any ONE of these. The kaggle CLI >= 2.2 and kagglehub both read
  * the NEW OAuth token at ~/.kaggle/access_token   (Kaggle's recommended flow;
    create it once with `kagglehub login`), or
  * the classic ~/.kaggle/kaggle.json  (Account -> Create New API Token), or
  * the env vars KAGGLE_USERNAME + KAGGLE_KEY.
This script never reads or stores a key -- it only checks one is present and lets
the CLI / kagglehub authenticate themselves. With OAuth the username is resolved
automatically via kagglehub.whoami(), so --prefix is optional.

Usage:
    pip install kaggle kagglehub
    kagglehub login                       # once -> writes ~/.kaggle/access_token
    python kaggle/run_on_kaggle.py --watch
    python kaggle/run_on_kaggle.py --no-src    # code dataset already pushed

NOTE: requires the `kaggle` CLI on PATH; verify-on-Kaggle for the exact
`kernels status` text parsed below.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = "00_setup_kaggle.ipynb"
# invoke the kaggle CLI via the current interpreter so it works without
# `kaggle` being on PATH (e.g. installed into a venv)
KAGGLE = [sys.executable, "-m", "kaggle"]


def resolve_user(prefix):
    if prefix:
        return prefix
    for k in ("KAGGLE_HANDLE_PREFIX", "KAGGLE_USERNAME"):
        if os.environ.get(k):
            return os.environ[k]
    cfg = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(cfg):
        try:
            return json.load(open(cfg)).get("username")
        except Exception:
            pass
    # new-auth (OAuth access_token): ask kagglehub who we are
    try:
        import kagglehub
        who = kagglehub.whoami()
        if isinstance(who, dict) and who.get("username"):
            return who["username"]
    except Exception:
        pass
    return None


def has_auth():
    """True if any supported Kaggle auth is present (env, classic, or new OAuth)."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kdir = os.path.expanduser("~/.kaggle")
    # classic kaggle.json OR the newer OAuth token (kaggle CLI >= 2.2 reads it)
    return (os.path.exists(os.path.join(kdir, "kaggle.json"))
            or os.path.exists(os.path.join(kdir, "access_token")))


def upload_src(user):
    print("[run] uploading source code dataset via kagglehub ...")
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "kaggle", "kagglehub_sync.py"),
                    "upload-src", "--prefix", user], check=True)


def build_push_dir(user, slug, src_slug, gpu, internet):
    d = tempfile.mkdtemp(prefix="kaggle_push_")
    shutil.copy2(os.path.join(REPO_ROOT, NOTEBOOK), os.path.join(d, NOTEBOOK))
    meta = {
        "id": f"{user}/{slug}",
        "title": "Newton vs FEM (soft-body)",
        "code_file": NOTEBOOK,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": bool(gpu),
        "enable_internet": bool(internet),
        "dataset_sources": [f"{user}/{src_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(d, "kernel-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return d


def watch(kernel_id, out_dir, timeout, interval):
    print(f"[run] polling status of {kernel_id} (timeout {timeout}s) ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = subprocess.run(KAGGLE + ["kernels", "status", kernel_id],
                             capture_output=True, text=True)
        line = (res.stdout + res.stderr).strip()
        low = line.lower()
        print(f"   {line}")
        if "complete" in low:
            os.makedirs(out_dir, exist_ok=True)
            subprocess.run(KAGGLE + ["kernels", "output", kernel_id, "-p", out_dir], check=True)
            print(f"[run] outputs pulled to {out_dir}")
            return True
        if "error" in low or "cancel" in low:
            print("[run] kernel did not complete successfully.")
            return False
        time.sleep(interval)
    print("[run] timed out waiting for the kernel.")
    return False


def main():
    ap = argparse.ArgumentParser(description="Launch the project on Kaggle (kaggle kernels push)")
    ap.add_argument("--prefix", default=None, help="Kaggle username (or $KAGGLE_HANDLE_PREFIX)")
    ap.add_argument("--slug", default="newton-vs-fem", help="kernel slug")
    ap.add_argument("--src-slug", default="newton-vs-fem-src", help="code dataset slug")
    ap.add_argument("--no-src", action="store_true", help="skip the source upload (already pushed)")
    ap.add_argument("--no-internet", action="store_true", help="disable kernel Internet")
    ap.add_argument("--no-gpu", action="store_true", help="run on CPU (no GPU)")
    ap.add_argument("--watch", action="store_true", help="poll status and pull outputs when done")
    ap.add_argument("--out", default="./out", help="dir for pulled outputs (--watch)")
    ap.add_argument("--timeout", type=int, default=3600, help="status-poll timeout [s]")
    ap.add_argument("--interval", type=int, default=20, help="status-poll interval [s]")
    args = ap.parse_args()

    if subprocess.run(KAGGLE + ["--version"], capture_output=True).returncode != 0:
        sys.exit("the `kaggle` package is not importable -- run: pip install kaggle")
    if not has_auth():
        sys.exit("no Kaggle auth found -- log in with `kagglehub login` (new OAuth token), "
                 "or set KAGGLE_USERNAME + KAGGLE_KEY, or place ~/.kaggle/kaggle.json")
    user = resolve_user(args.prefix)
    if not user:
        sys.exit("could not resolve your Kaggle username -- pass --prefix <user>")

    print(f"[run] user={user}  kernel={user}/{args.slug}  src={user}/{args.src_slug}")
    if not args.no_src:
        upload_src(user)

    push_dir = build_push_dir(user, args.slug, args.src_slug,
                              gpu=not args.no_gpu, internet=not args.no_internet)
    print(f"[run] kaggle kernels push -p {push_dir}")
    subprocess.run(KAGGLE + ["kernels", "push", "-p", push_dir], check=True)
    shutil.rmtree(push_dir, ignore_errors=True)

    kernel_id = f"{user}/{args.slug}"
    print(f"[run] pushed. Watch it at https://www.kaggle.com/code/{user}/{args.slug}")
    if args.watch:
        ok = watch(kernel_id, args.out, args.timeout, args.interval)
        sys.exit(0 if ok else 1)
    else:
        print(f"[run] check: kaggle kernels status {kernel_id}")
        print(f"[run] fetch: kaggle kernels output {kernel_id} -p {args.out}")


if __name__ == "__main__":
    main()

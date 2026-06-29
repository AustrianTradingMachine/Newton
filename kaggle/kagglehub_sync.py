"""kagglehub transport for the Newton-vs-FEM project (no GitHub needed).

kagglehub is Kaggle's modern Python API for *artifacts* (datasets / models /
packages) plus auth. It is the clean way to move CODE and RESULTS in and out of
Kaggle. It does NOT start kernel execution -- launching the GPU run is still the
Kaggle notebook UI or `kaggle kernels push`. kagglehub handles the two transport
legs around that run:

    local  --upload-src-->   Kaggle Dataset  (the code)
    kernel --upload-out-->   Kaggle Dataset  (data/ + figures/)
    local  --download-out--> the results

Handles are <prefix>/<name>; prefix is your Kaggle username, from
$KAGGLE_HANDLE_PREFIX or --prefix. Auth: run `kagglehub.login()` once, or set
KAGGLE_USERNAME / KAGGLE_KEY in the environment.

Usage:
    # locally: ship the code, then run the notebook on Kaggle (UI or CLI), then:
    python kaggle/kagglehub_sync.py upload-src   --prefix <user>
    python kaggle/kagglehub_sync.py download-out --prefix <user> -o ./out
    # inside the Kaggle kernel (see 00_setup_kaggle.ipynb):
    python kaggle/kagglehub_sync.py download-src --prefix <user> -o /kaggle/working/Newton
    python kaggle/kagglehub_sync.py upload-out   --prefix <user>

NOTE: kagglehub APIs evolve; the exact `dataset_upload` / `dataset_download`
signatures are the verify-on-Kaggle spots here.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NAME = "newton-vs-fem-src"     # dataset of the code
OUT_NAME = "newton-vs-fem-out"     # dataset of data/ + figures/

# code paths shipped in the source dataset (everything needed to `pip install -e .`)
SRC_ALLOW = ["common", "newton_run", "fenics_run", "compare", "kaggle",
             "pyproject.toml", "requirements.txt", "README.md",
             "00_setup_kaggle.ipynb", "00_setup_colab.ipynb",
             "10_stage_a_analysis.ipynb", "20_stage_b_analysis.ipynb",
             "30_convergence_analysis.ipynb", "40_friction_analysis.ipynb"]


def _prefix(args) -> str:
    p = args.prefix or os.environ.get("KAGGLE_HANDLE_PREFIX")
    if not p:
        raise SystemExit("set --prefix <kaggle-username> or $KAGGLE_HANDLE_PREFIX")
    return p


def _stage(paths) -> str:
    """Copy an allowlist of repo paths into a fresh temp dir; return its path."""
    staging = tempfile.mkdtemp(prefix="kh_stage_")
    for rel in paths:
        src = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(staging, rel)
        if os.path.isdir(src):
            shutil.copytree(src, dst,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".*"))
        else:
            shutil.copy2(src, dst)
    return staging


def _upload(handle: str, local_dir: str, notes: str):
    import kagglehub
    print(f"[kagglehub] uploading {local_dir} -> {handle}")
    kagglehub.dataset_upload(handle, local_dir, version_notes=notes)
    print(f"[kagglehub] done: {handle}")


def _download(handle: str, out_dir: str):
    import kagglehub
    print(f"[kagglehub] downloading {handle}")
    cached = kagglehub.dataset_download(handle)
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(cached):
        s = os.path.join(cached, name)
        d = os.path.join(out_dir, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    print(f"[kagglehub] copied {handle} -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="kagglehub code/results transport")
    ap.add_argument("cmd", choices=["upload-src", "download-src", "upload-out", "download-out"])
    ap.add_argument("--prefix", default=None, help="Kaggle username (or $KAGGLE_HANDLE_PREFIX)")
    ap.add_argument("-o", "--out", default=None, help="target dir for download-*")
    ap.add_argument("--notes", default="automated upload", help="version notes for upload-*")
    args = ap.parse_args()
    pre = _prefix(args)

    if args.cmd == "upload-src":
        staging = _stage(SRC_ALLOW)
        _upload(f"{pre}/{SRC_NAME}", staging, args.notes)
        shutil.rmtree(staging, ignore_errors=True)
    elif args.cmd == "download-src":
        _download(f"{pre}/{SRC_NAME}", args.out or os.path.join(os.getcwd(), "Newton"))
    elif args.cmd == "upload-out":
        staging = _stage(["data", "figures"])
        _upload(f"{pre}/{OUT_NAME}", staging, args.notes)
        shutil.rmtree(staging, ignore_errors=True)
    elif args.cmd == "download-out":
        _download(f"{pre}/{OUT_NAME}", args.out or os.path.join(os.getcwd(), "out"))


if __name__ == "__main__":
    main()

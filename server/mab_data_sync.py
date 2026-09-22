"""
Single script for keeping local (D:\\Data\\mab) and server
(bapun@10.36.17.152:~/Data) data in sync via rsync over WSL.

Two directions:
  - push_csv: local -> server, *.csv files across all dataset folders.
    "Local always wins" -- no --update, always overwrites the server's copy.
  - pull_pt: server -> local, *.pt (RNN model) files across all dataset
    folders. Uses --update so a locally-newer file is never clobbered.

Manual usage:
    python mab_data_sync.py push-csv
    python mab_data_sync.py push-csv --dry-run
    python mab_data_sync.py pull-pt

As a pre-flight step before submitting a SLURM job (from another script):
    from mab_data_sync import sync_before_submit
    sync_before_submit()   # runs push_csv, logs triggered_by="preflight-sbatch"

Every run appends one line to data_sync.log (same folder as this script)
recording timestamp, direction, dry_run, triggered_by, and a summary
parsed from rsync's own --stats output.
"""

import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path

LOCAL_ROOT = "/mnt/d/Data/mab/"
REMOTE = "bapun@10.36.17.152:~/Data/"
DATASETS = ["ACdataset", "ASdataset", "BGdataset", "RNNdataset"]
LOG_PATH = Path(__file__).parent / "data_sync.log"


def _build_include_args(pattern):
    args = []
    for ds in DATASETS:
        args += [
            f"--include={ds}/",
            f"--include={ds}/**/",
            f"--include={ds}/**/{pattern}",
        ]
    args.append("--exclude=*")
    return args


def _run_rsync(src, dest, pattern, extra_flags=None, dry_run=False):
    cmd = ["wsl", "rsync", "-avP", "--stats"]
    if dry_run:
        cmd.append("--dry-run")
    if extra_flags:
        cmd += extra_flags
    cmd += _build_include_args(pattern)
    cmd += [src, dest]
    return subprocess.run(cmd, capture_output=True, text=True)


def _parse_stats(stdout):
    def grab(label):
        m = re.search(rf"{label}: ([\d,]+)", stdout)
        return m.group(1) if m else "?"

    n_files = grab("Number of regular files transferred")
    total_size = grab("Total transferred file size")
    return n_files, total_size


def _log(direction, dry_run, triggered_by, result=None, error=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if error is not None:
        line = (
            f"[{timestamp}] {direction} dry_run={dry_run} "
            f"triggered_by={triggered_by} FAILED: {error}\n"
        )
    else:
        n_files, total_size = _parse_stats(result.stdout)
        line = (
            f"[{timestamp}] {direction} dry_run={dry_run} triggered_by={triggered_by} "
            f"files_transferred={n_files} bytes_transferred={total_size} "
            f"rc={result.returncode}\n"
        )
    with open(LOG_PATH, "a") as f:
        f.write(line)
    print(line.strip())


def push_csv(dry_run=False, triggered_by="manual"):
    """Local -> server, *.csv files. Local always wins (no --update)."""
    try:
        result = _run_rsync(LOCAL_ROOT, REMOTE, "*.csv", dry_run=dry_run)
    except Exception as e:
        _log("push-csv", dry_run, triggered_by, error=str(e))
        return False
    _log("push-csv", dry_run, triggered_by, result=result)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def pull_pt(dry_run=False, triggered_by="manual"):
    """Server -> local, *.pt files. --update: never overwrite a newer local file."""
    try:
        result = _run_rsync(
            REMOTE, LOCAL_ROOT, "*.pt", extra_flags=["--update"], dry_run=dry_run
        )
    except Exception as e:
        _log("pull-pt", dry_run, triggered_by, error=str(e))
        return False
    _log("pull-pt", dry_run, triggered_by, result=result)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def sync_before_submit(dry_run=False):
    """Pre-flight hook: call this right before any sbatch submission."""
    return push_csv(dry_run=dry_run, triggered_by="preflight-sbatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync data between local D:\\Data\\mab and the server."
    )
    parser.add_argument("direction", choices=["push-csv", "pull-pt"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.direction == "push-csv":
        ok = push_csv(dry_run=args.dry_run)
    else:
        ok = pull_pt(dry_run=args.dry_run)

    raise SystemExit(0 if ok else 1)

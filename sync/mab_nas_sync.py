"""Keep the local dataset tree (D:\\Data\\mab) in step with the NAS.

The NAS holds the authoritative raw record under
Z:\\Bandit2ArmData\\BGdataset, laid out exactly like the local tree
(<Paradigm_*>/<lesion>/<animal>/raw_data), so moving data between them is a
plain mirror -- no per-file routing.

Three stages, each runnable on its own:

  pull     NAS -> local, *.csv and *.dat. Pull-only: nothing local is ever
           deleted, because the NAS is a rolling record and some sessions
           survive only on D:. Video and ATM_backups are skipped.
  rebuild  Regenerate each animal's pooled <animal>.csv from its raw_data
           with 'raw2ArmIO', which reads every session from its .csv when
           present and its .dat otherwise. Only folders whose raw files are
           newer than the pooled file are touched.
  push     Hand off to mab_data_sync.push_csv() to send the pooled files to
           the compute server.

Run from the repo root:
    python sync/mab_nas_sync.py pull --dry-run
    python sync/mab_nas_sync.py pull
    python sync/mab_nas_sync.py rebuild --animal BGF0
    python sync/mab_nas_sync.py all

The first rebuild after a change to how pooled files are built needs
'--force': staleness is judged from file mtimes, so an unchanged raw_data
folder looks up to date even when the build method has changed.

Every run appends a line to nas_sync.log next to this file.
"""

import argparse
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

NAS_ROOT = Path("Z:/Bandit2ArmData/BGdataset")
LOCAL_ROOT = Path("D:/Data/mab/BGdataset")
LOG_PATH = Path(__file__).parent / "nas_sync.log"

# Neither is behavioural data, and ATM_backups is ~800 GB of video and frames.
EXCLUDE_DIRS = ["video", "videos", "ATM_backups"]
PATTERNS = ["*.csv", "*.dat"]

CSV_SUFFIX = ".trial.csv"


def _log(stage, detail, dry_run=False, triggered_by="manual"):
    """Append one line to nas_sync.log and echo it."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {stage} dry_run={dry_run} triggered_by={triggered_by} {detail}"
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
    print(line)


# <Paradigm_*>/<lesion>/<animal>/raw_data. Globbing the exact depth rather
# than rglob keeps the walk out of ATM_backups, which is ~800 GB of video.
ANIMAL_GLOB = "*/*/*/raw_data"


def _animal_dirs(root, animal=None):
    """Every <animal> directory under 'root' that has a raw_data folder."""
    dirs = sorted(p.parent for p in root.glob(ANIMAL_GLOB) if p.is_dir())
    if animal is not None:
        dirs = [d for d in dirs if d.name == animal]
    return dirs


# ---------------------------------------------------------------- pull


def pull(dry_run=False, animal=None, triggered_by="manual"):
    """Mirror NAS -> local for *.csv and *.dat. Never deletes.

    robocopy rather than rsync: the NAS is an SMB share on Z:, which WSL (and
    so the rsync used by mab_data_sync) cannot see.

    Returns
    -------
    bool
        True when robocopy reported no failures.
    """
    if not NAS_ROOT.is_dir():
        _log("pull", f"FAILED: {NAS_ROOT} not reachable", dry_run, triggered_by)
        return False

    if animal is None:
        pairs = [(NAS_ROOT, LOCAL_ROOT)]
    else:
        pairs = [
            (d, LOCAL_ROOT / d.relative_to(NAS_ROOT)) for d in _animal_dirs(NAS_ROOT, animal)
        ]
        if not pairs:
            _log("pull", f"FAILED: no NAS folder for animal {animal}", dry_run, triggered_by)
            return False

    total_files = total_bytes = 0
    ok = True
    for src, dst in pairs:
        cmd = ["robocopy", str(src), str(dst), *PATTERNS, "/S", "/XD", *EXCLUDE_DIRS]
        cmd += ["/R:2", "/W:5", "/NFL", "/NDL", "/NJH", "/NP", "/BYTES", "/MT:8"]
        if dry_run:
            cmd.append("/L")

        result = subprocess.run(cmd, capture_output=True, text=True)
        # robocopy uses a bitmask: <8 means it did its job, >=8 means failures.
        if result.returncode >= 8:
            ok = False
            _log("pull", f"FAILED rc={result.returncode} {src}", dry_run, triggered_by)
            print(result.stdout[-2000:], file=sys.stderr)
            continue

        files, nbytes = _parse_robocopy(result.stdout)
        total_files += files
        total_bytes += nbytes

    verb = "would_copy" if dry_run else "copied"
    _log(
        "pull",
        f"{verb}_files={total_files} {verb}_bytes={total_bytes} ok={ok}",
        dry_run,
        triggered_by,
    )
    return ok


def _parse_robocopy(stdout):
    """Pull the copied file count and byte count out of robocopy's summary."""

    def grab(label):
        m = re.search(rf"^\s*{label}\s*:\s*\d+\s+(\d+)", stdout, re.MULTILINE)
        return int(m.group(1)) if m else 0

    return grab("Files"), grab("Bytes")


# ------------------------------------------------------------- rebuild


def needs_rebuild(animal_dir):
    """True when the pooled <animal>.csv is missing or older than its raw files."""
    raw = animal_dir / "raw_data"
    if not raw.is_dir():
        return False
    files = list(raw.glob(f"*{CSV_SUFFIX}")) + list(raw.glob("*.dat"))
    if not files:
        return False
    pooled = animal_dir / f"{animal_dir.name}.csv"
    if not pooled.exists():
        return True
    return max(f.stat().st_mtime for f in files) > pooled.stat().st_mtime


def rebuild(dry_run=False, animal=None, force=False, triggered_by="manual"):
    """Regenerate pooled <animal>.csv files from raw_data.

    Uses 'raw2ArmIO', so a session missing its .csv is still read from its
    .dat instead of being dropped.

    Returns
    -------
    bool
        True when every attempted rebuild succeeded.
    """
    from banditpy.io import raw2ArmIO, session_sources

    targets = [
        d
        for d in _animal_dirs(LOCAL_ROOT, animal)
        if force or needs_rebuild(d)
    ]
    if not targets:
        _log("rebuild", "nothing stale", dry_run, triggered_by)
        return True

    done = failed = 0
    for d in targets:
        raw = d / "raw_data"
        pooled = d / f"{d.name}.csv"
        rel = d.relative_to(LOCAL_ROOT)
        sources = session_sources(raw)
        n_csv = sum(1 for v in sources.values() if v.name.endswith(CSV_SUFFIX))
        tag = f"{rel} sessions={len(sources)} from_csv={n_csv} from_dat={len(sources)-n_csv}"

        if dry_run:
            print(f"  would rebuild {tag}")
            done += 1
            continue

        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                task = raw2ArmIO(raw)
            task.to_df().to_csv(pooled)
        except Exception as e:
            failed += 1
            print(f"  FAILED {rel}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        done += 1
        print(f"  rebuilt {tag} trials={task.n_trials} ({time.time()-t0:.0f}s)")

    _log("rebuild", f"rebuilt={done} failed={failed}", dry_run, triggered_by)
    return failed == 0


# ---------------------------------------------------------------- push


def push(dry_run=False, triggered_by="post-nas-rebuild"):
    """Send the pooled csv files on to the compute server."""
    from mab_data_sync import push_csv

    return push_csv(dry_run=dry_run, triggered_by=triggered_by)


# ----------------------------------------------------------------- cli


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=["pull", "rebuild", "push", "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--animal", help="restrict to one animal, e.g. BGF0")
    parser.add_argument(
        "--force", action="store_true", help="rebuild even when not stale"
    )
    args = parser.parse_args()

    if args.stage == "pull":
        ok = pull(dry_run=args.dry_run, animal=args.animal)
    elif args.stage == "rebuild":
        ok = rebuild(dry_run=args.dry_run, animal=args.animal, force=args.force)
    elif args.stage == "push":
        ok = push(dry_run=args.dry_run)
    else:
        ok = pull(dry_run=args.dry_run, animal=args.animal, triggered_by="all")
        ok = (
            rebuild(dry_run=args.dry_run, animal=args.animal, triggered_by="all") and ok
        )
        ok = push(dry_run=args.dry_run, triggered_by="all") and ok

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

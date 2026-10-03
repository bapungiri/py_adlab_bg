"""Fit VanillaRNNFit2Arm to animal sessions, one (hidden_size, session) per SLURM array task.

Task list is HIDDEN_SIZES x EXPS; run with '--list' to print the index mapping.
Submit with 'sbatch --array=0-<n_tasks-1> job_fit_rnn.slurm'.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import mab_subjects
from banditpy.models import VanillaRNNFit2Arm
from banditpy.models.rnn._mle_fit_base import _run_restart

EXPS = (
    mab_subjects.unstruc.p8020_good_intact_sess
    + mab_subjects.struc.p8020_good_intact_sess
    + mab_subjects.unstruc.p8020_lesion_OFC_post_sess
    + mab_subjects.struc.p8020_lesion_OFC_post_sess
    + mab_subjects.unstruc.p8020_lesion_mPFC_post_sess
    + mab_subjects.struc.p8020_lesion_mPFC_post_sess
)
HIDDEN_SIZES = (48, 6)
TASKS = [(h, exp) for h in HIDDEN_SIZES for exp in EXPS]

# -------- Params ---------
LR = 1e-3
SEGMENT_STARTS = "session"
# -------------------------


def model_path(exp: mab_subjects.MABData, hidden_size: int, n_epochs: int):
    fp = exp.filePrefix
    return fp.with_name(
        fp.stem + f"_RNNfit_N{hidden_size}_LR{LR}_E{n_epochs}_S{SEGMENT_STARTS}.pt"
    )


def fit_subject(
    exp: mab_subjects.MABData,
    hidden_size: int,
    n_epochs: int = 500,
    n_jobs_inner: int = 5,
    skip_existing: bool = False,
):
    model_filename = model_path(exp, hidden_size, n_epochs)
    if skip_existing and model_filename.is_file():
        print(f"Exists, skipping: {model_filename}")
        return

    task = exp.b2a
    task.auto_block_window_ids()

    if exp.lesion_tag == "intact":
        start_date = task.datetime[0]
        stop_date = task.datetime[-1]
        n_days = pd.Timedelta(stop_date - start_date).days
        if n_days > 30:
            task = task.filter_by_datetime(start=start_date + pd.Timedelta(days=30))

    task = task.filter_by_trials(min_trials=100)

    rnn_fit = VanillaRNNFit2Arm(
        task=task,
        hidden_size=hidden_size,
        segment_starts=SEGMENT_STARTS,
        device="cpu",
    )
    # Full-data fit runs in its own process alongside the CV folds, so all
    # n_jobs_inner + 1 CPUs stay busy instead of idling during the final fit.
    with ProcessPoolExecutor(max_workers=1) as pool:
        full_fit = pool.submit(
            _run_restart,
            VanillaRNNFit2Arm,
            rnn_fit._init_kwargs,
            rnn_fit.task,
            rnn_fit._seg_mask,
            "cpu",
            0,  # weight-init seed
            n_epochs,
            LR,
            1e-5,  # lr_min, same as fit() default
            False,
        )
        # evaluate generalization
        cv_results = rnn_fit.cross_validate(
            k=5, lr=LR, n_epochs=n_epochs, n_jobs=n_jobs_inner
        )
        _, final_nll, state_dict, nll_history = full_fit.result()

    rnn_fit.model.load_state_dict(state_dict)
    rnn_fit.model.eval()
    rnn_fit.nll_history = nll_history
    print(f"Full fit complete. Final NLL/trial: {final_nll:.4f}")
    rnn_fit.save(model_filename, extra={"cv": cv_results.to_dict("list")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Index into TASKS (defaults to SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument(
        "--n-jobs-inner",
        type=int,
        default=5,
        help="Parallel workers for the cross-validation folds",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--list", action="store_true", help="Print the task index mapping and exit"
    )
    args = parser.parse_args()

    if args.list:
        for i, (h, exp) in enumerate(TASKS):
            print(f"{i:3d}  N{h:<3d} {exp.group_tag:8s} {exp.lesion_tag:18s} {exp.sub_name}")
        print(f"n_tasks = {len(TASKS)}")
        raise SystemExit

    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    hidden_size, exp = TASKS[task_id]
    print(f"Task {task_id}: N{hidden_size} {exp.group_tag} {exp.lesion_tag} {exp.sub_name}")
    fit_subject(
        exp,
        hidden_size,
        n_epochs=args.n_epochs,
        n_jobs_inner=args.n_jobs_inner,
        skip_existing=args.skip_existing,
    )

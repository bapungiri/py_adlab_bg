from pathlib import Path
from joblib import Parallel, delayed
import mab_subjects
from banditpy.models import VanillaRNNFit2Arm
import pandas as pd
import argparse

EXPS = (
    mab_subjects.unstruc.p8020_good_intact_sess
    + mab_subjects.struc.p8020_good_intact_sess
    + mab_subjects.unstruc.p8020_lesion_OFC_post_sess
    + mab_subjects.struc.p8020_lesion_OFC_post_sess
    + mab_subjects.unstruc.p8020_lesion_mPFC_post_sess
    + mab_subjects.struc.p8020_lesion_mPFC_post_sess
)


def fit_subject(exp: mab_subjects.MABData, n_jobs_inner: int = 5):
    fp = exp.filePrefix
    task = exp.b2a
    task.auto_block_window_ids()

    if exp.lesion_tag == "intact":
        start_date = task.datetime[0]
        stop_date = task.datetime[-1]
        n_days = pd.Timedelta(stop_date - start_date).days
        if n_days > 30:
            task = task.filter_by_datetime(start=start_date + pd.Timedelta(days=30))

    # -------- Params ---------
    task = task.filter_by_trials(min_trials=100)
    hidden_size = 48
    lr = 1e-3
    n_epochs = 500
    segment_starts = "session"
    # -------------------------

    rnn_fit = VanillaRNNFit2Arm(
        task=task,
        hidden_size=hidden_size,
        segment_starts=segment_starts,
        device="cpu",
    )
    # evaluate generalization
    cv_results = rnn_fit.cross_validate(
        k=5, lr=lr, n_epochs=n_epochs, n_jobs=n_jobs_inner
    )

    # retrain on all data and save model
    rnn_fit.fit(n_epochs=n_epochs, lr=lr, progress_bar=False)
    model_filename = fp.with_name(
        fp.stem + f"_RNNfit_N{hidden_size}_LR{lr}_E{n_epochs}_S{segment_starts}.pt"
    )
    rnn_fit.save(model_filename, extra={"cv": cv_results.to_dict("list")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=30,
        help="Total number of models to train (split evenly between structured and unstructured)",
    )
    parser.add_argument(
        "--n-jobs-subject",
        type=int,
        default=15,
        help="Number of parallel workers (passed to joblib)",
    )
    parser.add_argument(
        "--n-jobs-inner",
        type=int,
        default=5,
        help="Reserved for inner parallelism (unused)",
    )
    args = parser.parse_args()
    n_jobs = args.n_jobs_subject

    Parallel(n_jobs=n_jobs)(
        delayed(fit_subject)(exp, args.n_jobs_inner) for exp in EXPS
    )

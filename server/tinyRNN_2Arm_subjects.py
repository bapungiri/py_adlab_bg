import sys

sys.path.extend(
    [
        "/mnt/pve/Homes/bapun/Codes/BanditPy",
        "/mnt/pve/Homes/bapun/Codes/NeuroPy",
        "/mnt/pve/Homes/bapun/Codes/py_adlab_bg",
    ]
)
from banditpy.core.mab import Bandit2Arm
from banditpy.models.rnn_models import (
    nested_cross_validation_tiny_behavior_v2,
    tiny_behavior_d_vs_weighted_nll,
)
import numpy as np
import time
import mab_subjects
from joblib import Parallel, delayed
import pandas as pd
import argparse

exps = mab_subjects.unstruc.allsess + mab_subjects.struc.allsess


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-subjects",
        type=int,
        default=2,
        help="Limit number of subjects for pilot run (default=2)",
    )
    ap.add_argument(
        "--subject-filter",
        type=str,
        default=None,
        help="Substring to filter subject names (optional)",
    )
    ap.add_argument(
        "--n-jobs-subject", type=int, default=2, help="Parallel subjects (outer level)"
    )
    ap.add_argument(
        "--n-jobs-inner",
        type=int,
        default=4,
        help="Parallel (l1,seed) combos inside nested CV",
    )
    return ap.parse_args()


def run_tinyRNN(exp, n_jobs_inner: int):
    cv_result = nested_cross_validation_tiny_behavior_v2(
        data=exp.b2a.filter_by_trials(100, 100),
        num_actions=2,
        hidden_size_grid=(1, 2, 3, 5, 8),
        l1_grid=(1e-4, 1e-3),
        seed_grid=(0, 1),
        block_size=100,
        outer_folds=5,
        patience=15,
        max_epochs=200,
        include_zero_l1=False,
        weight_inner_by_trials=True,
        derive_refit_epoch="median",
        checkpoint_path=f"{exp.sub_name}_nestedcv.json",
        resume=True,
        compute_tier1=True,
        n_jobs_inner=n_jobs_inner,
    )

    d_vs_nll = tiny_behavior_d_vs_weighted_nll(cv_result)

    print(f"{exp.sub_name} weighted mean per d:", d_vs_nll)
    df = pd.DataFrame(d_vs_nll, columns=["d", "nll"])
    df["sub_name"] = exp.sub_name
    df["grp"] = "struc" if exp.b2a.is_structured else "unstruc"
    df["first_experience"] = True if "Exp1" in exp.sub_name else False

    # Aggregate Tier1 metrics across folds per d and write JSON file per subject
    import json, numpy as _np

    tier1_summary = {}
    for d_key, info in cv_result["per_d"].items():
        folds = info["folds"]
        # collect only non-null tier1 entries
        tier_entries = [f["tier1"] for f in folds if f.get("tier1")]
        if not tier_entries:
            continue

        # Numeric arrays to aggregate: mean, std, lag1_autocorr, unit_grad_sensitivity (may be None)
        def avg_list(field):
            arrs = [e[field] for e in tier_entries if e.get(field) is not None]
            if not arrs:
                return None
            return (_np.mean(_np.array(arrs), axis=0)).tolist()

        mean_avg = avg_list("mean")
        std_avg = avg_list("std")
        lag1_avg = avg_list("lag1_autocorr")
        grad_avg = avg_list("unit_grad_sensitivity")
        participation_ratio_mean = float(
            _np.mean([e["participation_ratio"] for e in tier_entries])
        )
        pc1_var_ratio_mean = float(_np.mean([e["pc1_var_ratio"] for e in tier_entries]))
        total_timepoints_sum = int(
            _np.sum([e["total_timepoints"] for e in tier_entries])
        )
        tier1_summary[int(d_key)] = {
            "mean": mean_avg,
            "std": std_avg,
            "lag1_autocorr": lag1_avg,
            "unit_grad_sensitivity": grad_avg,
            "participation_ratio_mean": participation_ratio_mean,
            "pc1_var_ratio_mean": pc1_var_ratio_mean,
            "total_timepoints_sum": total_timepoints_sum,
            "folds_with_metrics": len(tier_entries),
        }
    out_json = {
        "subject": exp.sub_name,
        "group": "struc" if exp.b2a.is_structured else "unstruc",
        "first_experience": True if "Exp1" in exp.sub_name else False,
        "d_vs_nll": d_vs_nll,
        "tier1_summary": tier1_summary,
    }
    with open(f"{exp.sub_name}_tier1.json", "w") as f:
        json.dump(out_json, f)

    df.to_csv(f"{exp.sub_name}_tinyRNN_results.csv", index=False)

    return df


if __name__ == "__main__":
    args = parse_args()
    selected = exps
    if args.subject_filter:
        selected = [e for e in selected if args.subject_filter in e.sub_name]
    if args.max_subjects is not None:
        selected = selected[: args.max_subjects]
    print(
        f"Running {len(selected)} subjects with subject_parallel={args.n_jobs_subject}, inner_parallel={args.n_jobs_inner}"
    )
    results = Parallel(n_jobs=args.n_jobs_subject)(
        delayed(run_tinyRNN)(exp, args.n_jobs_inner) for exp in selected
    )
    params_df = pd.concat(results, ignore_index=True)
    params_df.to_csv("tinyRNN_results.csv", index=False)
    print("Pilot run complete. Consolidated CSV written to tinyRNN_results.csv")

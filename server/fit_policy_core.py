import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mab_subjects import MABData
from mab_builders.preprocess import DEFAULT_KWARGS_EXPERT
from banditpy.models import DecisionModel
from banditpy.models.policy import StaticBeta, BasePolicy
from numpy.random import default_rng
import itertools


# ---------------------------------------------------------------------
# Single-subject fitting
# ---------------------------------------------------------------------
def fit_blocks(
    exp: MABData,
    policy_ctor: BasePolicy,
    fit_kwargs,
    optimizer=None,
    filter_by_datetime: bool = True,
    test=None,
    require_expert=None,
):
    """Fit one subject on all blocks and on the high-low tier.

    require_expert : None, bool or "auto"
        None (legacy): intact sessions skip their first 30 days when they
        span more than 30 days (and 'filter_by_datetime' is True).
        True: trim to trials from the animal's expertise day onward, with the
        same criteria as mab_builders.perf_tier (DEFAULT_KWARGS_EXPERT).
        "auto": like True for intact sessions, keep lesion sessions whole.
        False: keep all trials.
    """
    task = exp.b2a

    # ------- Task filters --------
    if exp.data_tag != "RNNdataset":
        task.auto_block_window_ids()
    if require_expert is None:
        if exp.lesion_tag == "intact":
            start_date = task.datetime[0]
            stop_date = task.datetime[-1]
            n_days = pd.Timedelta(stop_date - start_date).days
            if filter_by_datetime and n_days > 30:
                task = task.filter_by_datetime(
                    start=start_date + pd.Timedelta(days=30)
                )
    else:
        if require_expert == "auto":
            require_expert = exp.lesion_tag == "intact"
        if require_expert and exp.data_tag != "RNNdataset":
            _, expert_datetime, _, _ = task.get_expertise_day(
                by="datetime", **DEFAULT_KWARGS_EXPERT
            )
            task = task.filter_by_datetime(start=expert_datetime)

    # Don't clip to prevent teleportation
    task = task.filter_by_trials(min_trials=100, clip_max=None)
    window_start = task.is_window_start

    # task_block1 = task.filter_by_block_id(start=1, stop=1)
    # task_block1_reset = task_block1.get_block_start_mask(start=1, stop=1)

    # task_block2plus = task.filter_by_block_id(start=2)
    # task_block2plus_reset = task_block2plus.get_block_start_mask(start=2, stop=2)

    # ====== Models ========

    # --------Per probability combination (too fragmented)--------
    # probs_pool = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    # probs_combinations = np.array(list(itertools.combinations(probs_pool, 2)))

    # probs = np.sort(task.probs[task.is_session_start, :], axis=1)
    # _, probs_counts = np.unique(probs, axis=0, return_counts=True)
    # probs_counts_min = probs_counts.min()

    # models = {}
    # rng = default_rng()
    # for prob in probs_combinations:
    #     prob_perms = np.array(list(itertools.permutations(prob)))

    #     try:
    #         task_prob = task.filter_by_probs(prob_perms)
    #         session_ids = np.unique(task_prob.session_ids)
    #         chosen_session_id = np.sort(
    #             rng.choice(session_ids, size=probs_counts_min, replace=False)
    #         )
    #         task_prob = task_prob.filter_by_session_id(ids=chosen_session_id)

    #         if task_prob.n_sessions > 10:
    #             session_start = task_prob.is_session_start

    #             model_prob = DecisionModel(
    #                 task_prob,
    #                 policy=policy_ctor(beta_schedule=StaticBeta()),
    #                 reset_mode=session_start,
    #             )
    #             model_prob.fit(optimizer=optimizer, **fit_kwargs)

    #             models.update({f"prob_{prob[0]}_{prob[1]}": model_prob})
    #     except Exception as e:
    #         pass

    # -------- Easy-Hard combination --------------

    # task_easy = task.filter_by_deltaprob(delta_min=0.38)  # >=0.4
    # task_hard = task.filter_by_deltaprob(delta_min=0.08, delta_max=0.35)  # <=0.3

    # model_easy = DecisionModel(
    #     task_easy,
    #     policy=policy_ctor(beta_schedule=StaticBeta()),
    #     reset_mode="session",
    # )
    # model_easy.fit(optimizer=optimizer, **fit_kwargs)

    # model_hard = DecisionModel(
    #     task_hard,
    #     policy=policy_ctor(beta_schedule=StaticBeta()),
    #     reset_mode="session",
    # )
    # model_hard.fit(optimizer=optimizer, **fit_kwargs)

    # --------low-low, low-high, high-high combinations ---------
    # task_easy = task.filter_by_deltaprob(delta_min=0.38)  # >=0.4
    # low_low_mask = (task.probs[:, 0] <= 0.4) & (task.probs[:, 1] <= 0.4)
    # high_high_mask = (task.probs[:, 0] >= 0.6) & (task.probs[:, 1] >= 0.6)

    # model_low_high = DecisionModel(
    #     task_easy,
    #     policy=policy_ctor(beta_schedule=StaticBeta()),
    #     reset_mode="session",
    # )
    # model_low_high.fit(optimizer=optimizer, **fit_kwargs)

    # model_low_low = DecisionModel(
    #     task._filtered(low_low_mask),
    #     policy=policy_ctor(beta_schedule=StaticBeta()),
    #     reset_mode="session",
    # )
    # model_low_low.fit(optimizer=optimizer, **fit_kwargs)

    # model_high_high = DecisionModel(
    #     task._filtered(high_high_mask),
    #     policy=policy_ctor(beta_schedule=StaticBeta()),
    #     reset_mode="session",
    # )
    # model_high_high.fit(optimizer=optimizer, **fit_kwargs)

    # --------- Correlated/Uncorrelated combinations ----------

    # probs_all = task.probs

    # corr_mask = probs_all.sum(axis=1).round(2) == 1.0  # correlated combinations
    # uncorr_mask = ~corr_mask  # uncorrelated combinations

    # model_corr = DecisionModel(
    #     task._filtered(corr_mask),
    #     policy=policy_ctor(beta_schedule=StaticBeta()),
    #     reset_mode="session",
    # )
    # model_corr.fit(optimizer=optimizer, **fit_kwargs)

    # model_uncorr = DecisionModel(
    #     task._filtered(uncorr_mask),
    #     policy=policy_ctor(beta_schedule=StaticBeta()),
    #     reset_mode="session",
    # )
    # model_uncorr.fit(optimizer=optimizer, **fit_kwargs)

    # -------- Block wise ----------------

    model_all = DecisionModel(
        task,
        policy=policy_ctor(),
        reset_mode=window_start,
    )
    # model_all.cross_validate(n_folds=5, optimizer=optimizer, n_jobs=5)
    model_all.fit(optimizer=optimizer, **fit_kwargs)

    # -------- High-low tier (exactly one arm >= 0.5), as in mab_builders.preprocess --------
    n_high = (task.probs >= 0.5).sum(axis=1)
    task_high_low = task._filtered(n_high == 1)
    model_high_low = DecisionModel(
        task_high_low,
        policy=policy_ctor(),
        reset_mode=task_high_low.is_window_start,
    )
    model_high_low.fit(optimizer=optimizer, **fit_kwargs)

    # model_block1 = DecisionModel(
    #     task_block1,
    #     policy=policy_ctor(),
    #     reset_mode=task_block1_reset,
    # )
    # model_block1.fit(optimizer=optimizer, **fit_kwargs)

    # model_block2 = DecisionModel(
    #     task_block2plus,
    #     policy=policy_ctor(),
    #     reset_mode=task_block2plus_reset,
    # )
    # model_block2.fit(optimizer=optimizer, **fit_kwargs)

    # ------ compiling models ----------
    models = {
        "all": model_all,
        "high_low": model_high_low,
        # "low_high": model_low_high,
        # "low_low": model_low_low,
        # "high_high": model_high_high,
    }

    param_names = []
    param_values = []
    fit_scope = []

    for scope, model in models.items():
        d = model.to_dict()
        param_names.extend(d.keys())
        param_values.extend(d.values())
        fit_scope.extend([scope] * len(d))

    temp_dict = {
        "name": exp.sub_name,
        "policy": policy_ctor.__name__,
        "param_names": param_names,
        "param_values": param_values,
        "fit_scope": fit_scope,
    }
    temp_dict.update(exp.common_kwargs)

    return pd.DataFrame(temp_dict)


# ---------------------------------------------------------------------
# Single-subject, all policies
# ---------------------------------------------------------------------
def fit_subject(
    exp: MABData,
    policies,
    fit_kwargs,
    optimizer=None,
    filter_by_datetime: bool = True,
    require_expert=None,
):
    frames = []
    for policy_ctor in policies:
        df = fit_blocks(
            exp=exp,
            policy_ctor=policy_ctor,
            fit_kwargs=fit_kwargs,
            optimizer=optimizer,
            filter_by_datetime=filter_by_datetime,
            require_expert=require_expert,
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# Batch fitting with parallel execution
# ---------------------------------------------------------------------
def fit_experiments(
    exps,
    policies,
    fit_kwargs,
    optimizer=None,
    n_jobs=1,
    verbose=True,
    filter_by_datetime: bool = True,
    require_expert=None,
):
    def _fit_one(exp):
        if verbose:
            print(f"Starting: {exp.sub_name}")

        df = fit_subject(
            exp=exp,
            policies=policies,
            fit_kwargs=fit_kwargs,
            optimizer=optimizer,
            filter_by_datetime=filter_by_datetime,
            require_expert=require_expert,
        )

        if verbose:
            print(f"Completed: {exp.sub_name}")

        return df

    results = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(exp) for exp in exps)

    return pd.concat(results, ignore_index=True)

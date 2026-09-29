from functools import wraps

import pandas as pd
from joblib import Parallel, delayed

from mab_group_data import GroupData


def _session_key(exp):
    """Unique label for one session set (names repeat across lesion/paradigm)."""
    return f"{exp.sub_name}|{exp.paradigm_tag}|{exp.lesion_tag}|{exp.group_tag}"


def _run_one(compute, exp, params):
    """Run 'compute' on one exp; return (DataFrame, None) or (None, error text)."""
    try:
        out = compute(exp, **params)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    df = out if isinstance(out, pd.DataFrame) else pd.DataFrame(out)
    for k, v in exp.common_kwargs.items():
        if k not in df:
            df[k] = v
    return df, None


def group_builder(name):
    """Turn a per-subject function into a group dataframe builder.

    The decorated function 'compute(exp, **params)' returns a dict or DataFrame
    for one subject. The returned builder 'run(exps, save=True, save_as=None,
    n_jobs=1, **params)' runs it on every exp ('n_jobs' > 1 in parallel), adds
    'exp.common_kwargs' columns (without overwriting columns 'compute' already
    set), concatenates, and saves to GroupData under 'save_as' (default 'name')
    along with the params, subject names and any failures.

    An exp that raises is skipped, not fatal: the error is printed and recorded
    under 'failed' in the metadata, so one odd session can't sink a full build.

    'exps', 'save', 'save_as' and 'n_jobs' are reserved -- a builder can't use
    them as its own parameter names. The raw per-subject function stays
    available as 'run.compute' for debugging on a single exp.
    """

    def deco(compute):
        @wraps(compute)
        def run(exps, save=True, save_as=None, n_jobs=1, **params):
            if n_jobs == 1:
                results = []
                for exp in exps:
                    print(exp.sub_name)
                    results.append(_run_one(compute, exp, params))
            else:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(_run_one)(compute, exp, params) for exp in exps
                )

            rows, failed = [], {}
            for exp, (df, err) in zip(exps, results):
                if err is None:
                    rows.append(df)
                else:
                    failed[_session_key(exp)] = err
                    print(f"[{save_as or name}] skipped {_session_key(exp)}: {err}")

            if not rows:
                raise RuntimeError(f"{save_as or name}: every exp failed: {failed}")
            df = pd.concat(rows, ignore_index=True)

            if save:
                meta = dict(
                    builder=f"{compute.__module__}.{compute.__name__}",
                    params=params,
                    exps=[_session_key(exp) for exp in exps],
                    failed=failed,
                )
                GroupData().save(df, save_as or name, meta=meta)
            return df

        run.compute = compute
        run.basename = name
        return run

    return deco

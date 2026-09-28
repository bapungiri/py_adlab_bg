from functools import wraps

import pandas as pd

from mab_group_data import GroupData


def group_builder(name):
    """Turn a per-subject function into a group dataframe builder.

    The decorated function 'compute(exp, **params)' returns a dict or DataFrame
    for one subject. The returned builder 'run(exps, save=True, save_as=None,
    **params)' loops over 'exps', adds 'exp.common_kwargs' columns (without
    overwriting columns 'compute' already set), concatenates, and saves to
    GroupData under 'save_as' (default 'name') along with the params and
    subject names.

    The raw per-subject function stays available as 'run.compute' for debugging
    on a single exp.
    """

    def deco(compute):
        @wraps(compute)
        def run(exps, save=True, save_as=None, **params):
            rows = []
            for exp in exps:
                print(exp.sub_name)
                out = compute(exp, **params)
                df = out if isinstance(out, pd.DataFrame) else pd.DataFrame(out)
                for k, v in exp.common_kwargs.items():
                    if k not in df:
                        df[k] = v
                rows.append(df)

            df = pd.concat(rows, ignore_index=True)

            if save:
                meta = dict(
                    builder=f"{compute.__module__}.{compute.__name__}",
                    params=params,
                    exps=[exp.sub_name for exp in exps],
                )
                GroupData().save(df, save_as or name, meta=meta)
            return df

        run.compute = compute
        run.basename = name
        return run

    return deco

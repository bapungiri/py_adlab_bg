"""Benchmark L-BFGS-B vs Optuna for BanditPy Qlearn fits on identical simulated data.

Usage: python optimizer_benchmark.py <out.json>  (12 subjects, ~18k trials each; ~5.4 h on 12 local cores).
Summary + figures: see the Obsidian note "adlab optimizer benchmark".
"""
import sys, time, json
import numpy as np
from joblib import Parallel, delayed

N_SUB, N_BLOCKS_PER_TIER, N_JOBS = 12, 40, 12
ES = dict(early_stop=True, es_warmup_trials=3000, es_check_every=250, es_slack=0.01)


def one_subject(i, seed_seq):
    import banditpy.models.model as mm
    from banditpy.models import DecisionModel
    from banditpy.models.policy import Qlearn
    from banditpy.models.optim import OptunaOptimizer, LBFGSOptimizer
    from scipy.optimize import minimize
    sys.path.insert(0, r"C:\Users\asheshlab\Documents\Codes\py_adlab_bg\server")
    import param_recovery as pr

    # count every likelihood evaluation (incl. early-stopped ones) through the real code path
    orig = mm._nll_core
    count = {"n": 0, "trials": 0}
    def counted(policy, choices, rewards, resets, theta, **kw):
        count["n"] += 1
        return orig(policy, choices, rewards, resets, theta, **kw)
    mm._nll_core = counted

    rng = np.random.default_rng(seed_seq)
    names = Qlearn().active_parameter_names()
    b = Qlearn().get_bounds()
    true = {n: float(np.exp(rng.uniform(np.log(b[n][0]), np.log(b[n][1])))) if n == "beta"
            else float(rng.uniform(*b[n])) for n in names}
    pol = Qlearn(); pol.set_params(true)
    task = DecisionModel.simulate_policy(policy=pol, reward_schedule=pr.generate_probs_tiers(N_BLOCKS_PER_TIER, rng),
                                         min_trials_per_block=100, prob_switch=0.02, seed=rng.integers(2**32))
    th = lambda p: np.array([p[n] for n in names])
    ref = DecisionModel(task=task, policy=Qlearn(), reset_mode="session")
    nll_true = orig(ref.policy, ref.choices, ref.rewards, ref.resets, th(true))

    def run(label, **fit):
        m = DecisionModel(task=task, policy=Qlearn(), reset_mode="session")
        count["n"] = 0; t0 = time.time()
        m.fit(seed=1000 + i, n_jobs=1, **fit)
        return dict(method=label, nll=float(m.nll), evals=count["n"], secs=time.time() - t0,
                    fvals=[float(v) for v in m.fit_fvals], params={n: float(m.params[n]) for n in names}), m

    out = []
    r, _ = run("optuna 80x5", optimizer=OptunaOptimizer(n_trials=80, log_params={"beta"}), n_starts=5, **ES); out.append(r)
    r, _ = run("optuna 300x5", optimizer=OptunaOptimizer(n_trials=300, log_params={"beta"}), n_starts=5, **ES); out.append(r)
    r, _ = run("lbfgs x5", optimizer=LBFGSOptimizer(), n_starts=5); out.append(r)
    r, _ = run("lbfgs x10", optimizer=LBFGSOptimizer(), n_starts=10); out.append(r)
    # hybrid: optuna 80x5 then one L-BFGS-B polish from its best point (no early stop)
    r, m = run("optuna 80x5 + lbfgs polish", optimizer=OptunaOptimizer(n_trials=80, log_params={"beta"}), n_starts=5, **ES)
    count["n"] = 0; t0 = time.time()
    f = lambda x: orig(m.policy, m.choices, m.rewards, m.resets, x)
    res = minimize(f, th(m.params), method="L-BFGS-B", bounds=[b[n] for n in names])
    count["n"] += res.nfev
    r.update(nll=float(min(res.fun, r["nll"])), evals=r["evals"] + res.nfev, secs=r["secs"] + time.time() - t0,
             params={n: float(v) for n, v in zip(names, res.x)} if res.fun < r["nll"] else r["params"])
    out.append(r)
    for r in out:
        r.update(sub=i, true=true, nll_true=float(nll_true), n_trials=len(task.choices))
    return out


if __name__ == "__main__":
    seeds = np.random.SeedSequence(2026).spawn(N_SUB)
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS)(delayed(one_subject)(i, s) for i, s in enumerate(seeds))
    rows = [r for sub in res for r in sub]
    json.dump(rows, open(sys.argv[1], "w"))
    print(f"done in {time.time() - t0:.0f}s, {len(rows)} fits")

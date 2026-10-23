# TODO

## Now
- [ ] Decide how to refit 9505 Qlearn (`fit_qlearn_high_low_p9505`): trim from each animal's expertise day (matches `perf_tier`) or no trimming. `fit_blocks` only skips 30 days when an animal has >30 days, so BGF9/BGF8/BGM10 are fit on everything while BGM8/BGM9/BGF7 are fit on late data only. Edit `server/fit_policy_core.py` only when no fit job is running.
- [ ] **Running** (2026-10-05, SLURM 99664 + rerun 100355 of tasks 53/54 lost in the Ceph outage; copy+merge 100356 → `param_recovery_qlearn2regime_sticky_narrowtrue`) Qlearn2Regime recovery with shared perseverance (`alpha_h`, `sticky`, true sticky [0, 5]) and learned `b_init`, 24 starts. Label alignment must also map `b_init` → `1 - b_init` under a regime swap.
- [ ] **Running** (2026-10-05, SLURM 99706 / 99708 / 99710, merges 99707 / 99709 / 99711 → `fit_qlearn2regime_sticky_{p8020_intact,p8020_lesion_mPFC,p9505}`) Qlearn2Regime + perseverance + `b_init` animal fits (presets `q2r_sticky_*`, 24 starts). Then compare with Qlearn sticky by BIC: only if it wins is there evidence for regimes. 8020 intact may need TimeLimit raised above 2 days (admin: `scontrol update JobId=99706 TimeLimit=4-00:00:00`).
- [ ] **Running** (2026-10-06, SLURM 100357 = BGM3 rerun with 4-day limit; 97364_6 BGF5 restarted after node failure; copy+merge 100358 → `fit_qlearn2regime_p8020_intact`) Qlearn2Regime (no perseverance) 8020 intact, for comparison with the sticky version.
- [ ] Analyse the Qlearn sticky recovery with true sticky in [0, 5] (job 96518, done; `param_recovery_struc_unstruc_tier_qlearn_sticky_truesticky0to5`, copied locally) and add it to the Obsidian Qlearn note.
- [ ] Check BGM9 on the rig: 100% port-1 choices in its last 3 days (possible port-2 sensor issue). Excluded from the 9505 fit plot in `mab_model_analysis.ipynb`.

## Next
- [ ] Fill gaps and conflicts in `records/` found during the 2026-10-22 migration:
  - Task end dates for BGF1, BGM3, BGM5 (their boxes were reused by BGF6, BGM8, BGF10), start dates for BGF8/BGF9/BGM10/BGF11, and `8020_short` dates of BGM5/BGF4.
  - BGM7 and BGF7 both listed on 10.36.22.15 from 2026-06-14.
  - Lesion surgeries of BGF5 and BGM7 not recorded (date, sites); BGF0's surgery sheet is a copy of BGF2's; BGM6 has no surgery sheet.
  - BGF3: lesion sheet says perfused 2026-01-18, log says removed from OpCon cage 2026-01-21.
  - BGM4 death: log/old sheet say 2026-02-05, lesion index says 2026-02-06. Old sheet's "Died during surgery" note sat on BGM5's row.
  - Neymar sex (lesion sheet: female; `mab_subjects`: male). Who is Felix?
  - BGM5 "scrap data" days 2026-01-05 to 01-08 are not excluded in `mab_subjects.py`; add an `exclude` window if they should be dropped.
- [ ] Align `fit_blocks` intact trimming (30-day skip) with `perf_tier` (expertise day) so fits and performance curves use the same trials.
- [ ] mPFC lesion fits: lesion sessions start on the first post-lesion day (may include recovery) — consider skipping early post-lesion days.
- [ ] Move the remaining builder cells in `mab_choice_performance1.ipynb` into `mab_builders`.
- [ ] `perf_probability_matrix` now holds all good animals + RNN (from `mab_builders.build_all`): add a `select(...)` filter to the cells reading it in `mab_fellowship_india_alliance.ipynb` and `mab_poster_embo.ipynb`.
- [ ] Update notebooks that still import the removed `colors_2arm` from `mab_colors` (11 notebooks) when revisiting them.
- [ ] Reward-driven regime belief for Qlearn2Regime (option): the current belief is updated only from the agent's own choices, so in simulation it can't switch strategy on reward evidence (5 Oct `b_init` simulation). E.g. regimes gain belief when their agent predicted the reward well, or a reward-rate-driven belief as in `RewardRateGatedRegime`.
- [ ] Regime belief around block transitions: from fitted Qlearn2Regime, compute the trial-by-trial belief, align on block starts, and compare transitions where the tier changes vs stays the same (prediction: the dominant regime flips only when the tier changes).
- [ ] Recovery under tier-linked regimes: simulate agents whose regime follows the block tier (with a lag) and resets only at window starts; check that a Qlearn2Regime fit recovers the switches.
- [ ] Order agent/regime labels consistently across animals before plotting Qlearn2Regime fits (generic plotting cell in `mab_model_analysis.ipynb` plots raw labels).
- [ ] Rework parameter recovery of the remaining multi-regime policies (Qlearn3Regime, MoARegime, ...).
- [ ] Move the simulation scripts behind the Obsidian "does the model learn?" sections (`qlearn_sim_learn.py`, `q2r_sim_binit.py`, currently in a Claude scratch folder) into the repo if they are worth keeping.

## Later
- [ ] Distinguish mPFC vs OFC lesions in shared panels by line style (color alone can't separate three struc shades).

## Done
- [x] 2026-10-05 Qlearn2Regime: optional shared perseverance and learned `b_init` (BanditPy `7a4fc4b`); `--variant sticky` for it in `param_recovery.py` and `q2r_sticky_*` fit presets.
- [x] 2026-10-05 Simulations "does the model learn?" for Qlearn and Qlearn2Regime on 8020 structured/unstructured schedules; added to the Obsidian notes with a model algorithm history section.
- [x] 2026-10-05 Generic fitted-parameter plotting cell in `mab_model_analysis.ipynb` (any `fit_name`, parameters read from the fit).
- [x] 2026-10-05 Qlearn2Regime 9505 and 8020 mPFC lesion fits (`fit_qlearn2regime_p9505`, `fit_qlearn2regime_p8020_lesion_mPFC`): worse than Qlearn sticky on every animal, parameters degenerate (no perseveration term).
- [x] 2026-10-04 Qlearn2Regime recovery with 24 starts (job 97334): fits short of the truth 37–49% → 21–31%; recoverable after label alignment.
- [x] 2026-10-03 `LBFGSOptimizer` bounds-clipped gradient, failed starts recorded as nan (BanditPy `7221f6a`); array/merge jobs exit with python's status (`0a88852`).
- [x] 2026-10-03 SLURM arrays default to 6 CPUs and 12 starts (`33346ea`).
- [x] 2026-10-03 6-param Qlearn (L-BFGS-B, sticky bound [0, 10]) fits: 8020 intact (expert trimming), 8020 mPFC lesion and 9505 (`fit_qlearn_sticky_*`).
- [x] 2026-10-02 Parameter recovery designs `struc_unstruc_tier` and `tier_mix` (default and sticky variants).
- [x] 2026-09-30 Parameter recovery `tier` (one model → per-tier) and `tier_params` (per-tier models → one model), plus `tier` nobias and sticky variants.
- [x] 2026-09-28 Qlearn fits on all + high-low tiers: mPFC lesion (`fit_qlearn_high_low_lesion_mPFC`) and 9505 (`fit_qlearn_high_low_p9505`).
- [x] 2026-09-28 Switch probability conditioned on whether the better port flipped (`swp_by_prev_best_arm`).
- [x] 2026-09-28 `mab_builders` package with GroupData metadata; `perf_tier` with `min_sessions` and `require_expert="auto"`.

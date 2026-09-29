# TODO

## Now
- [ ] Decide how to refit 9505 Qlearn (`fit_qlearn_high_low_p9505`): trim from each animal's expertise day (matches `perf_tier`) or no trimming. `fit_blocks` only skips 30 days when an animal has >30 days, so BGF9/BGF8/BGM10 are fit on everything while BGM8/BGM9/BGF7 are fit on late data only. Edit `server/fit_policy_core.py` only when no fit job is running.
- [ ] Check BGM9 on the rig: 100% port-1 choices in its last 3 days (possible port-2 sensor issue). Excluded from the 9505 fit plot in `mab_model_analysis.ipynb`.

## Next
- [ ] Align `fit_blocks` intact trimming (30-day skip) with `perf_tier` (expertise day) so fits and performance curves use the same trials.
- [ ] mPFC lesion fits: lesion sessions start on the first post-lesion day (may include recovery) — consider skipping early post-lesion days.
- [ ] Move the remaining builder cells in `mab_choice_performance1.ipynb` into `mab_builders`.
- [ ] `perf_probability_matrix` now holds all good animals + RNN (from `mab_builders.build_all`): add a `select(...)` filter to the cells reading it in `mab_fellowship_india_alliance.ipynb` and `mab_poster_embo.ipynb`.
- [ ] Update notebooks that still import the removed `colors_2arm` from `mab_colors` (11 notebooks) when revisiting them.
- [ ] Parameter recovery, one model → per-tier recovery: simulate mixed blocks (low-low, high-low, high-high) with a single Qlearn model, then run recovery separately on each tier.
- [ ] Parameter recovery, per-tier models → one model: simulate each tier with its own Qlearn parameters, then recover with a single Qlearn fit on all tiers.
- [ ] Rework parameter recovery of the multi-regime policies (Qlearn2Regime, Qlearn3Regime, MoARegime, ...).

## Later
- [ ] Distinguish mPFC vs OFC lesions in shared panels by line style (color alone can't separate three struc shades).

## Done
- [x] 2026-09-28 Qlearn fits on all + high-low tiers: mPFC lesion (`fit_qlearn_high_low_lesion_mPFC`) and 9505 (`fit_qlearn_high_low_p9505`).
- [x] 2026-09-28 Switch probability conditioned on whether the better port flipped (`swp_by_prev_best_arm`).
- [x] 2026-09-28 `mab_builders` package with GroupData metadata; `perf_tier` with `min_sessions` and `require_expert="auto"`.

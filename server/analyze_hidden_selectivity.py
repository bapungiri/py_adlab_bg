"""Analyze hidden state selectivity (choice & reward) at the selected dimensionality d.

For each subject:
  1. Load its nested CV checkpoint JSON (<sub>_nestedcv.json)
  2. Select best d (lowest weighted_mean_test_nll)
  3. Determine hyperparameters to refit a single model:
       - l1_recurrent: modal (most frequent) chosen_l1 across folds for that d
       - seed: chosen_seed from the fold with lowest test_nll (stable tie-break)
       - refit_epochs: median of fold['refit_epochs'] (fallback to 100 if missing)
  4. Refit model on ALL blocks (constructed with block_size) from subject data
  5. Extract hidden states over all blocks (resetting h0 each block)
  6. Compute per-unit selectivity indices:
        choice_SI = (mean(h|a=1) - mean(h|a=0)) / pooled_std
        reward_SI = (mean(h|r=1) - mean(h|r=0)) / pooled_std
     where pooled_std = sqrt(0.5*(var_a1 + var_a0)) (adds 1e-8 stability)
  7. Save per-unit CSV + summary CSV (subject-level averages) + group comparison.

Usage example:
  python analyze_hidden_selectivity.py \
      --checkpoint-dir . \
      --output-prefix hidden_selectivity \
      --block-size 100

Assumptions:
  * Subject experiment objects available via mab_subjects (same as training script)
  * Per-subject checkpoint JSON files already exist in --checkpoint-dir
  * banditpy and dependencies are on PYTHONPATH
"""

from __future__ import annotations
import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# Extend sys.path similarly to training script (user environment assumption)
import sys
sys.path.extend([
    "/mnt/pve/Homes/bapun/Codes/BanditPy",
    "/mnt/pve/Homes/bapun/Codes/NeuroPy",
    "/mnt/pve/Homes/bapun/Codes/py_adlab_bg"
])

import mab_subjects  # noqa: E402
from banditpy.models.rnn_models import TinyBehaviorRNN, TinyBehaviorRNNTrainer  # noqa: E402


@dataclass
class SubjectConfig:
    name: str
    group: str
    is_structured: bool
    first_experience: bool
    exp_obj: Any


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-dir', type=str, default='.', help='Directory containing <sub>_nestedcv.json')
    ap.add_argument('--output-prefix', type=str, default='hidden_selectivity', help='Prefix for output CSV files')
    ap.add_argument('--block-size', type=int, default=100, help='Block size used during training')
    ap.add_argument('--max-refit-epochs', type=int, default=5000, help='Upper cap for refit epochs (safety)')
    ap.add_argument('--subject-filter', type=str, default=None, help='Substring to filter subject names')
    ap.add_argument('--device', type=str, default=None, help='Torch device (None=auto CPU)')
    return ap.parse_args()


def build_subject_list(filter_substr: str | None) -> List[SubjectConfig]:
    exps = mab_subjects.unstruc.allsess + mab_subjects.struc.allsess
    out = []
    for e in exps:
        if filter_substr and filter_substr not in e.sub_name:
            continue
        out.append(SubjectConfig(
            name=e.sub_name,
            group='struc' if e.b2a.is_structured else 'unstruc',
            is_structured=e.b2a.is_structured,
            first_experience=('Exp1' in e.sub_name),
            exp_obj=e,
        ))
    return out


def load_checkpoint(checkpoint_dir: str, sub_name: str) -> Dict[str, Any]:
    path = os.path.join(checkpoint_dir, f"{sub_name}_nestedcv.json")
    with open(path, 'r') as f:
        return json.load(f)


def select_best_d(per_d: Dict[str, Any]) -> int:
    # per_d keys are stringified ints; choose d with lowest weighted_mean_test_nll
    best_d = None
    best_score = float('inf')
    for d_str, info in per_d.items():
        score = info.get('weighted_mean_test_nll', float('inf'))
        if score < best_score:
            best_score = score
            best_d = int(d_str)
    if best_d is None:
        raise ValueError("Could not determine best d from checkpoint")
    return best_d


def derive_refit_hparams(per_d_entry: Dict[str, Any]) -> Dict[str, Any]:
    folds = per_d_entry.get('folds', [])
    if not folds:
        raise ValueError("per_d entry missing folds for refit parameter derivation")
    # Modal l1
    l1_vals = [f['chosen_l1'] for f in folds if 'chosen_l1' in f]
    l1_mode = Counter(l1_vals).most_common(1)[0][0]
    # Best seed by lowest test_nll
    best_fold = min(folds, key=lambda x: x.get('test_nll', float('inf')))
    seed = best_fold.get('chosen_seed', 0)
    # Refit epochs median
    refit_epochs_list = [f.get('refit_epochs', 0) for f in folds]
    refit_epochs = int(np.median([e for e in refit_epochs_list if e is not None]))
    if refit_epochs <= 0:
        refit_epochs = 100
    return {
        'l1_recurrent': l1_mode,
        'seed': seed,
        'refit_epochs': refit_epochs,
    }


def build_blocks_from_bandit(bandit_obj, block_size: int):
    # Mirror logic used in nested CV helper
    import numpy as np  # local import for safety
    try:
        bin_choices = bandit_obj.get_binarized_choices()
    except Exception:
        bin_choices = np.where(bandit_obj.choices == 2, 1, 0)
    rewards = np.asarray(bandit_obj.rewards)
    session_ids = np.asarray(bandit_obj.session_ids)
    sessions = []
    for sid in np.unique(session_ids):
        mask = session_ids == sid
        sessions.append({'actions': bin_choices[mask], 'rewards': rewards[mask]})
    blocks = []
    for sess in sessions:
        acts = np.asarray(sess['actions'])
        rews = np.asarray(sess['rewards'])
        T = len(acts)
        start = 0
        while start < T:
            end = min(start + block_size, T)
            blocks.append({'actions': acts[start:end].copy(), 'rewards': rews[start:end].copy()})
            start = end
    return blocks


def prepare_sequence(actions: np.ndarray, rewards: np.ndarray, num_actions: int = 2):
    # Re-implement minimal previous-action + previous-reward design
    T = len(actions)
    X = np.zeros((T, num_actions + 1), dtype=np.float32)
    # at t=0 all zeros
    for t in range(1, T):
        a_prev = actions[t - 1]
        X[t, a_prev] = 1.0
        X[t, -1] = rewards[t - 1]
    y = actions.copy()
    return X, y


def refit_model(d: int, hparams: Dict[str, Any], blocks: List[Dict[str, np.ndarray]], device=None, max_refit_epochs: int = 5000):
    import torch
    import random
    # Seed control
    random.seed(hparams['seed']); np.random.seed(hparams['seed']); torch.manual_seed(hparams['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hparams['seed'])
    model = TinyBehaviorRNN(input_size=3, num_actions=2, hidden_size=d)
    trainer = TinyBehaviorRNNTrainer(
        model,
        l1_recurrent=hparams['l1_recurrent'],
        max_epochs=max_refit_epochs,
        patience=max_refit_epochs,  # disable early stopping by setting patience high
        device=device,
    )
    # Concatenate all blocks as training list and run fixed epochs
    epochs = min(hparams['refit_epochs'], max_refit_epochs)
    trainer.train_fixed_epochs(blocks, epochs)
    return trainer.model


def extract_hidden(model: TinyBehaviorRNN, blocks: List[Dict[str, np.ndarray]], device=None):
    import torch
    model.eval()
    hs_all = []
    actions_all = []
    rewards_all = []
    with torch.no_grad():
        for blk in blocks:
            acts = np.asarray(blk['actions'])
            rews = np.asarray(blk['rewards'])
            X, y = prepare_sequence(acts, rews, num_actions=2)
            x_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,D)
            h0 = model.h0_param.repeat(1,1,model.hidden_size)
            out_seq, _ = model.rnn(x_t, h0)  # (1,T,H)
            hs_all.append(out_seq.squeeze(0).cpu().numpy())
            actions_all.append(acts)
            rewards_all.append(rews)
    H = np.concatenate(hs_all, axis=0)  # (TotalT,H)
    A = np.concatenate(actions_all, axis=0)
    R = np.concatenate(rewards_all, axis=0)
    return H, A, R


def compute_selectivity(H: np.ndarray, A: np.ndarray, R: np.ndarray):
    eps = 1e-8
    res = []
    for u in range(H.shape[1]):
        h = H[:, u]
        # Choice selectivity
        mask0 = A == 0; mask1 = A == 1
        if mask0.sum() > 1 and mask1.sum() > 1:
            m0 = h[mask0].mean(); m1 = h[mask1].mean()
            v0 = h[mask0].var(); v1 = h[mask1].var()
            pooled = np.sqrt(0.5 * (v0 + v1) + eps)
            choice_si = (m1 - m0) / pooled
        else:
            choice_si = np.nan
            m0 = m1 = np.nan
        # Reward selectivity
        maskr0 = R == 0; maskr1 = R == 1
        if maskr0.sum() > 1 and maskr1.sum() > 1:
            mr0 = h[maskr0].mean(); mr1 = h[maskr1].mean()
            vr0 = h[maskr0].var(); vr1 = h[maskr1].var()
            pooled_r = np.sqrt(0.5 * (vr0 + vr1) + eps)
            reward_si = (mr1 - mr0) / pooled_r
        else:
            reward_si = np.nan
            mr0 = mr1 = np.nan
        res.append({
            'unit': u,
            'choice_si': choice_si,
            'choice_mean_a0': m0,
            'choice_mean_a1': m1,
            'reward_si': reward_si,
            'reward_mean_r0': mr0,
            'reward_mean_r1': mr1,
        })
    return res


def summarize_groups(unit_df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate by subject first to avoid unit-count bias
    subj_rows = unit_df.groupby(['sub_name','group','d'], as_index=False).agg(
        mean_choice_si=('choice_si', 'mean'),
        mean_reward_si=('reward_si', 'mean'),
        mean_abs_choice_si=('choice_si', lambda x: np.nanmean(np.abs(x))),
        mean_abs_reward_si=('reward_si', lambda x: np.nanmean(np.abs(x))),
    )
    group_rows = subj_rows.groupby('group', as_index=False).agg(
        subjects=('sub_name','nunique'),
        d_set=('d', lambda s: sorted(set(s))),
        avg_choice_si=('mean_choice_si','mean'),
        avg_reward_si=('mean_reward_si','mean'),
        avg_abs_choice_si=('mean_abs_choice_si','mean'),
        avg_abs_reward_si=('mean_abs_reward_si','mean'),
    )
    return subj_rows, group_rows


def main():
    args = parse_args()
    subjects = build_subject_list(args.subject_filter)
    if not subjects:
        print("No subjects matched filter.")
        return
    all_unit_rows = []
    summary_selection_rows = []
    for subj in subjects:
        ckpt_path = os.path.join(args.checkpoint_dir, f"{subj.name}_nestedcv.json")
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Missing checkpoint for {subj.name}, skipping.")
            continue
        try:
            ckpt = load_checkpoint(args.checkpoint_dir, subj.name)
            per_d = ckpt['per_d']
            best_d = select_best_d(per_d)
            hparams = derive_refit_hparams(per_d[str(best_d)])
            blocks = build_blocks_from_bandit(subj.exp_obj.b2a, args.block_size)
            model = refit_model(best_d, hparams, blocks, device=args.device, max_refit_epochs=args.max_refit_epochs)
            H, A, R = extract_hidden(model, blocks, device=args.device)
            unit_metrics = compute_selectivity(H, A, R)
            for row in unit_metrics:
                row.update({
                    'sub_name': subj.name,
                    'group': subj.group,
                    'first_experience': subj.first_experience,
                    'd': best_d,
                    'l1_recurrent': hparams['l1_recurrent'],
                    'seed': hparams['seed'],
                    'refit_epochs_used': hparams['refit_epochs'],
                })
            all_unit_rows.extend(unit_metrics)
            summary_selection_rows.append({
                'sub_name': subj.name,
                'group': subj.group,
                'd': best_d,
                'l1_recurrent': hparams['l1_recurrent'],
                'seed': hparams['seed'],
                'refit_epochs_used': hparams['refit_epochs'],
                'n_units': best_d,
            })
            print(f"Processed {subj.name}: d={best_d} l1={hparams['l1_recurrent']} seed={hparams['seed']} epochs={hparams['refit_epochs']}")
        except Exception as e:
            print(f"[ERROR] Failed subject {subj.name}: {e}")
    if not all_unit_rows:
        print("No data collected.")
        return
    unit_df = pd.DataFrame(all_unit_rows)
    selection_df = pd.DataFrame(summary_selection_rows)
    subj_df, group_df = summarize_groups(unit_df)
    base = args.output_prefix
    unit_df.to_csv(f"{base}_units.csv", index=False)
    selection_df.to_csv(f"{base}_selection_meta.csv", index=False)
    subj_df.to_csv(f"{base}_subject_summary.csv", index=False)
    group_df.to_csv(f"{base}_group_summary.csv", index=False)
    print("Saved:")
    print(f"  {base}_units.csv")
    print(f"  {base}_selection_meta.csv")
    print(f"  {base}_subject_summary.csv")
    print(f"  {base}_group_summary.csv")


if __name__ == '__main__':
    main()

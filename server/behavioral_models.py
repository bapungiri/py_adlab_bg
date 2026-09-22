"""Behavioral models and RNN-based agents.

This module consolidates cognitive (analytical) model-free bandit agents and
neural (model-free / model-based) recurrent agents adapted from the BanditPy
implementation. It provides:

Classes
-------
BanditModelFreeAgent2Arm
    Analytical model-free agent variants: decay (mfd), decay + perseveration (mfdp),
    and learn-all binary (mflb). Supports simulation, log-likelihood, and parameter fitting.
BanditModelBasedRNN
    Hybrid actor-critic + reward-belief recurrent agent (policy/value + belief heads).
TinyBehaviorRNN
    Supervised maximum-likelihood GRU ("tiny RNN") for behavioral fitting as in
    cognitive strategy discovery literature.

Trainers / Helpers
------------------
_ModelBasedRNNTrainer
    Trainer for BanditModelBasedRNN with combined losses (policy/value/belief/entropy).
TinyBehaviorRNNTrainer
    Supervised negative log-likelihood trainer with early stopping & weight decay.
collate_sessions
    Prepare variable-length session dictionaries into padded tensors.

Each class is self-contained (no dependency on the rest of BanditPy code), making
it easier to deploy inside this server context.

Dependencies: torch, numpy, math, random, typing, optional scipy (for refinement).
"""

from __future__ import annotations

import math
import random
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Cognitive Analytical Agent (Model-Free Variants)
# ---------------------------------------------------------------------------


class BanditModelFreeAgent2Arm:
    """Model-free cognitive-style agent for 2-armed bandit tasks.

    Supported variants (string argument `variant`):
      - 'mfd'   : Model-free with optional global decay (Q <- beta * Q) then chosen update
      - 'mfdp'  : Same as 'mfd' plus perseveration bias parameter (rho)
      - 'mflb'  : Model-free learn-all (binary rewards) — simultaneous chosen & unchosen updates

    Parameterizations:
      variant='mfd' (decay=True): params = [alpha, beta_decay, inv_temp]
      variant='mfd' (decay=False): params = [alpha, inv_temp]
      variant='mfdp': as above + perseveration rho (added last)
      variant='mflb': params = [alpha_c, util_c_r0, util_c_r1, alpha_u, util_u_r0, util_u_r1, inv_temp]

    Methods:
      simulate(n_trials, reward_probs, params=None, seed=None, greedy_eval=False)
      log_likelihood(choices, rewards, params)
      fit(choices, rewards, n_starts=50, method='auto')
    """

    def __init__(self, variant: str = "mfd", decay: bool = True):
        self.variant = variant.lower()
        self.decay = decay if self.variant in ("mfd", "mfdp") else False
        self._set_param_spec()

    def _set_param_spec(self):
        spec = []
        if self.variant == "mfd":
            if self.decay:
                spec = [("alpha", "unit"), ("beta_decay", "unit"), ("inv_temp", "pos")]
            else:
                spec = [("alpha", "unit"), ("inv_temp", "pos")]
        elif self.variant == "mfdp":
            if self.decay:
                spec = [
                    ("alpha", "unit"),
                    ("beta_decay", "unit"),
                    ("inv_temp", "pos"),
                    ("rho", "unc"),
                ]
            else:
                spec = [("alpha", "unit"), ("inv_temp", "pos"), ("rho", "unc")]
        elif self.variant == "mflb":
            spec = [
                ("alpha_c", "unit"),
                ("util_c_r0", "unc"),
                ("util_c_r1", "unc"),
                ("alpha_u", "unit"),
                ("util_u_r0", "unc"),
                ("util_u_r1", "unc"),
                ("inv_temp", "pos"),
            ]
        else:
            raise ValueError("variant must be one of {'mfd','mfdp','mflb'}")
        self.param_spec = spec

    @staticmethod
    def _softmax(logits):
        z = logits - np.max(logits)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    def _coerce_actions(self, choices):
        choices = np.asarray(choices)
        if choices.min() == 1 and choices.max() == 2:
            return choices - 1
        return choices

    def default_params(self):
        vals = []
        for _, t in self.param_spec:
            if t == "unit":
                vals.append(0.5)
            elif t == "pos":
                vals.append(5.0)
            else:
                vals.append(0.0)
        return np.array(vals, dtype=float)

    def _update_mfd(self, Q, action, reward, params):
        if self.decay:
            if self.variant == "mfdp":
                alpha, beta_decay, inv_temp, *rest = params
            else:
                alpha, beta_decay, inv_temp = params[:3]
            Q *= beta_decay
        else:
            if self.variant == "mfdp":
                alpha, inv_temp, *rest = params
            else:
                alpha, inv_temp = params[:2]
        Q[action] = (1 - alpha) * Q[action] + alpha * reward
        return Q

    def _update_mflb(self, Q, action, reward, params):
        (alpha_c, util_c_r0, util_c_r1, alpha_u, util_u_r0, util_u_r1, inv_temp) = (
            params
        )
        if reward == 0:
            Q = alpha_u * Q + util_u_r0
            Q[action] = alpha_c * Q[action] + util_c_r0
        else:
            Q = alpha_u * Q + util_u_r1
            Q[action] = alpha_c * Q[action] + util_c_r1
        return Q

    def simulate(
        self, n_trials, reward_probs, params=None, seed=None, greedy_eval=False
    ):
        rng = np.random.default_rng(seed)
        reward_probs = np.asarray(reward_probs, dtype=float)
        assert reward_probs.shape == (2,)
        if params is None:
            params = self.default_params()
        params = np.asarray(params, dtype=float)
        Q = np.zeros(2)
        choices = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials, dtype=int)
        probs_arr = np.zeros((n_trials, 2))
        prev_action = None
        for t in range(n_trials):
            if self.variant in ("mfd", "mfdp"):
                if self.decay:
                    if self.variant == "mfdp":
                        alpha, beta_decay, inv_temp, *rest = params
                    else:
                        alpha, beta_decay, inv_temp = params[:3]
                else:
                    if self.variant == "mfdp":
                        alpha, inv_temp, *rest = params
                    else:
                        alpha, inv_temp = params[:2]
                logits = inv_temp * Q
                if self.variant == "mfdp" and prev_action is not None:
                    rho = params[-1]
                    logits[prev_action] += rho
            else:
                inv_temp = params[-1]
                logits = inv_temp * Q
            p = self._softmax(logits)
            probs_arr[t] = p
            a = int(np.argmax(p) if greedy_eval else rng.choice([0, 1], p=p))
            r = int(rng.random() < reward_probs[a])
            if self.variant in ("mfd", "mfdp"):
                Q = self._update_mfd(Q, a, r, params)
            else:
                Q = self._update_mflb(Q, a, r, params)
            choices[t] = a
            rewards[t] = r
            prev_action = a
        return {
            "choices": choices,
            "rewards": rewards,
            "probs": probs_arr,
            "params_used": params,
        }

    def log_likelihood(self, choices, rewards, params):
        choices = self._coerce_actions(choices)
        rewards = np.asarray(rewards).astype(int)
        params = np.asarray(params, dtype=float)
        Q = np.zeros(2)
        prev_action = None
        ll = 0.0
        for a, r in zip(choices, rewards):
            if self.variant in ("mfd", "mfdp"):
                if self.decay:
                    if self.variant == "mfdp":
                        alpha, beta_decay, inv_temp, *rest = params
                    else:
                        alpha, beta_decay, inv_temp = params[:3]
                else:
                    if self.variant == "mfdp":
                        alpha, inv_temp, *rest = params
                    else:
                        alpha, inv_temp = params[:2]
                logits = inv_temp * Q
                if self.variant == "mfdp" and prev_action is not None:
                    rho = params[-1]
                    logits[prev_action] += rho
            else:
                inv_temp = params[-1]
                logits = inv_temp * Q
            p = self._softmax(logits)
            ll += np.log(p[a] + 1e-9)
            if self.variant in ("mfd", "mfdp"):
                Q = self._update_mfd(Q, a, r, params)
            else:
                Q = self._update_mflb(Q, a, r, params)
            prev_action = a
        return ll

    def _sample_random_params(self, rng):
        vals = []
        for _, t in self.param_spec:
            if t == "unit":
                vals.append(rng.uniform(0.01, 0.99))
            elif t == "pos":
                vals.append(np.exp(rng.uniform(np.log(0.1), np.log(20.0))))
            else:
                vals.append(rng.normal(0.0, 0.5))
        return np.array(vals)

    def fit(
        self, choices, rewards, n_starts=50, method="auto", seed=None, verbose=False
    ):
        rng = np.random.default_rng(seed)
        choices = self._coerce_actions(choices)
        rewards = np.asarray(rewards).astype(int)
        best_ll = -np.inf
        best = None
        trials = []
        for i in range(n_starts):
            p0 = self._sample_random_params(rng)
            ll = self.log_likelihood(choices, rewards, p0)
            trials.append({"start_idx": i, "params": p0, "loglik": ll})
            if ll > best_ll:
                best_ll = ll
                best = p0
        refined = False
        if method != "none":
            try:
                from scipy.optimize import minimize

                def neg_obj(raw):
                    p = raw.copy()
                    for j, (_, t) in enumerate(self.param_spec):
                        if t == "unit":
                            p[j] = 1 / (1 + np.exp(-raw[j]))
                        elif t == "pos":
                            p[j] = np.exp(raw[j])
                    return -self.log_likelihood(choices, rewards, p)

                raw0 = []
                for v, (_, t) in zip(best, self.param_spec):
                    if t == "unit":
                        v = np.clip(v, 1e-6, 1 - 1e-6)
                        raw0.append(np.log(v) - np.log(1 - v))
                    elif t == "pos":
                        raw0.append(np.log(max(v, 1e-8)))
                    else:
                        raw0.append(v)
                res = minimize(neg_obj, np.array(raw0), method="L-BFGS-B")
                if res.success:
                    p_opt = res.x.copy()
                    for j, (_, t) in enumerate(self.param_spec):
                        if t == "unit":
                            p_opt[j] = 1 / (1 + np.exp(-p_opt[j]))
                        elif t == "pos":
                            p_opt[j] = np.exp(p_opt[j])
                    ll_new = self.log_likelihood(choices, rewards, p_opt)
                    if ll_new > best_ll:
                        best_ll = ll_new
                        best = p_opt
                        refined = True
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print("Refinement skipped:", e)
        return {
            "best_params": best,
            "neg_loglik": -best_ll,
            "loglik": best_ll,
            "variant": self.variant,
            "decay": self.decay,
            "refined": refined,
            "random_trials": trials,
            "param_spec": self.param_spec,
        }


# ---------------------------------------------------------------------------
# Model-Based RNN (actor-critic + belief)
# ---------------------------------------------------------------------------
class BanditModelBasedRNN(nn.Module):
    """Model-based RNN agent for two-armed bandit tasks.
    Combines policy/value estimation with explicit reward-belief logits."""

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_actions: int = 2,
        cell: str = "gru",
        belief_weight: float = 1.0,
        entropy_weight: float = 0.01,
        value_weight: float = 0.5,
        shared_head: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.cell_type = cell.lower()
        if self.cell_type == "gru":
            self.core = nn.GRU(input_size, hidden_size, batch_first=True)
        elif self.cell_type == "lstm":
            self.core = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("cell must be gru or lstm")
        self.belief_head = nn.Linear(hidden_size, num_actions)
        self.policy_head = (
            self.belief_head if shared_head else nn.Linear(hidden_size, num_actions)
        )
        self.value_head = nn.Linear(hidden_size, 1)
        self.belief_weight = belief_weight
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        for m in [self.belief_head, self.value_head] + (
            [] if shared_head else [self.policy_head]
        ):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, hidden=None):
        core, hid = self.core(x, hidden)
        belief_logits = self.belief_head(core)
        policy_logits = self.policy_head(core)
        values = self.value_head(core).squeeze(-1)
        return belief_logits, policy_logits, values, hid

    def make_trainer(self, lr=3e-4, gamma=0.9, device=None):
        return _ModelBasedRNNTrainer(
            self,
            lr=lr,
            gamma=gamma,
            device=device,
            belief_weight=self.belief_weight,
            entropy_weight=self.entropy_weight,
            value_weight=self.value_weight,
        )


class _ModelBasedRNNTrainer:
    def __init__(
        self,
        model: BanditModelBasedRNN,
        lr=3e-4,
        gamma=0.9,
        device=None,
        belief_weight=1.0,
        entropy_weight=0.01,
        value_weight=0.5,
    ):
        self.model = model
        self.gamma = gamma
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.belief_weight = belief_weight
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.history = []

    def _discount(self, rewards):
        G = []
        acc = 0.0
        for r in reversed(rewards):
            acc = r + self.gamma * acc
            G.insert(0, acc)
        return torch.tensor(G, dtype=torch.float32, device=self.device)

    def _build_input(self, prev_action, prev_reward):
        x = torch.zeros(self.model.input_size, device=self.device)
        if prev_action is not None:
            x[prev_action] = 1.0
        x[self.model.num_actions] = prev_reward if prev_reward is not None else 0.0
        return x

    def train_env(
        self,
        mode="Structured",
        n_sessions=500,
        n_trials=150,
        clip_grad=1.0,
        progress=True,
    ):
        def sample_probs():
            if isinstance(mode, str):
                if mode.lower().startswith("s"):
                    p = np.random.uniform(0, 1)
                    return np.array([p, 1 - p])
                else:
                    return np.random.uniform(0, 1, size=2)
            arr = np.asarray(mode)
            assert arr.shape == (2,)
            return arr

        self.model.train()
        records = []
        for s in range(n_sessions):
            rps = sample_probs()
            prev_action = None
            prev_reward = 0.0
            inputs = []
            actions = []
            rewards = []
            hidden = None
            for t in range(n_trials):
                x_t = (
                    self._build_input(prev_action, prev_reward)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                inputs.append(x_t)
                with torch.no_grad():
                    _, policy_logits, _, hidden = self.model(x_t, hidden)
                probs = torch.softmax(policy_logits.squeeze(0), dim=-1)
                act = torch.distributions.Categorical(probs).sample().item()
                rew = 1.0 if random.random() < rps[act] else 0.0
                actions.append(act)
                rewards.append(rew)
                prev_action = act
                prev_reward = rew
                records.append(
                    {
                        "session": s + 1,
                        "trial": t + 1,
                        "action": act + 1,
                        "reward": rew,
                        "p1": float(rps[0]),
                        "p2": float(rps[1]),
                    }
                )
            x_seq = torch.cat(inputs, dim=1)
            belief_logits_seq, policy_logits_seq, value_seq, _ = self.model(x_seq)
            policy_logits_seq = policy_logits_seq.squeeze(0)
            belief_logits_seq = belief_logits_seq.squeeze(0)
            value_seq = value_seq.squeeze(0)
            G = self._discount(rewards)
            actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
            log_probs = torch.distributions.Categorical(
                logits=policy_logits_seq
            ).log_prob(actions_t)
            advantage = G - value_seq.detach()
            policy_loss = -(log_probs * advantage).mean()
            value_loss = self.value_weight * (G - value_seq).pow(2).mean()
            entropy = (
                torch.distributions.Categorical(logits=policy_logits_seq)
                .entropy()
                .mean()
            )
            entropy_term = -self.entropy_weight * entropy
            chosen_logits = belief_logits_seq[torch.arange(len(actions)), actions_t]
            rew_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            belief_loss = self.belief_weight * F.binary_cross_entropy_with_logits(
                chosen_logits, rew_t
            )
            loss = policy_loss + value_loss + entropy_term + belief_loss
            self.opt.zero_grad()
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.opt.step()
            self.history.append(
                {
                    "session": s + 1,
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "belief_loss": float(belief_loss.item()),
                    "entropy": float(entropy.item()),
                }
            )
        import pandas as pd

        return pd.DataFrame(records)

    @torch.no_grad()
    def evaluate(self, reward_probs, n_trials=200, greedy=True):
        rps = np.asarray(reward_probs)
        prev_action = None
        prev_reward = 0.0
        hidden = None
        actions = []
        rewards = []
        for _ in range(n_trials):
            x_t = self._build_input(prev_action, prev_reward).unsqueeze(0).unsqueeze(0)
            _, policy_logits, _, hidden = self.model(x_t, hidden)
            probs = torch.softmax(policy_logits.squeeze(0), dim=-1)
            if greedy:
                act = int(torch.argmax(probs).item())
            else:
                act = torch.distributions.Categorical(probs).sample().item()
            rew = 1.0 if random.random() < rps[act] else 0.0
            actions.append(act)
            rewards.append(rew)
            prev_action = act
            prev_reward = rew
        return {"actions": np.array(actions), "rewards": np.array(rewards)}


# ---------------------------------------------------------------------------
# TinyBehaviorRNN (Supervised behavioral fitting)
# ---------------------------------------------------------------------------
class TinyBehaviorRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_actions: int,
        hidden_size: int,
        cell_type: str = "gru",
        diagonal_readout: bool = False,
        learn_h0: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.cell_type = cell_type.lower()
        self.diagonal_readout = diagonal_readout and (hidden_size == num_actions)
        self.learn_h0 = learn_h0
        if self.cell_type != "gru":
            raise ValueError("Only vanilla GRU supported.")
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        if self.diagonal_readout:
            self.theta = nn.Parameter(torch.zeros(num_actions))
            self.bias = nn.Parameter(torch.zeros(num_actions))
            nn.init.normal_(self.theta, 0.0, 1.0 / math.sqrt(hidden_size))
        else:
            self.readout = nn.Linear(hidden_size, num_actions)
            nn.init.xavier_uniform_(self.readout.weight)
            nn.init.constant_(self.readout.bias, 0.0)
        if learn_h0:
            self.h0_param = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.register_buffer("h0_param", torch.zeros(1, 1, hidden_size))

    def forward(self, x, h0=None):
        if h0 is None:
            h0 = self.h0_param.repeat(1, x.size(0), 1)
        out, h_n = self.rnn(x, h0)
        logits = (
            self.theta * out + self.bias if self.diagonal_readout else self.readout(out)
        )
        return logits, h_n

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# Helpers for session data


def _prepare_sequence(session, num_actions: int, state_dim: int | None = None):
    actions = np.asarray(session["actions"])
    if actions.min() == 1:
        actions = actions - 1
    rewards = np.asarray(session["rewards"])
    T = len(actions)
    assert len(rewards) == T
    has_states = "states" in session and state_dim is not None
    if has_states:
        states = np.asarray(session["states"])
        assert len(states) == T
    X = []
    for t in range(T):
        if t == 0:
            a_prev = np.zeros(num_actions)
            r_prev = 0.0
            s_prev = np.zeros(state_dim) if has_states else None
        else:
            a_prev = np.zeros(num_actions)
            a_prev[actions[t - 1]] = 1.0
            r_prev = rewards[t - 1]
            s_prev = np.zeros(state_dim) if has_states else None
            if has_states:
                s_prev[states[t - 1]] = 1.0
        parts = [a_prev, [r_prev]]
        if has_states:
            parts.append(s_prev)
        X.append(np.concatenate(parts))
    X = np.stack(X, 0)
    y = actions.copy()
    return X, y


def collate_sessions(
    sessions, num_actions: int, state_dim: int | None = None, device=None
):
    processed = [_prepare_sequence(s, num_actions, state_dim) for s in sessions]
    lengths = [p[0].shape[0] for p in processed]
    Tmax = max(lengths)
    D = processed[0][0].shape[1]
    B = len(processed)
    inputs = np.zeros((B, Tmax, D), dtype=np.float32)
    targets = np.zeros((B, Tmax), dtype=np.int64)
    mask = np.zeros((B, Tmax), dtype=np.float32)
    for i, (X, y) in enumerate(processed):
        L = X.shape[0]
        inputs[i, :L] = X
        targets[i, :L] = y
        mask[i, :L] = 1.0
    return {
        "inputs": torch.tensor(inputs, device=device),
        "targets": torch.tensor(targets, device=device),
        "mask": torch.tensor(mask, device=device),
        "lengths": lengths,
    }


class TinyBehaviorRNNTrainer:
    def __init__(
        self,
        model: TinyBehaviorRNN,
        lr=5e-3,
        weight_decay=5e-4,
        max_epochs=500,
        batch_size=16,
        patience=30,
        grad_clip=5.0,
        device=None,
    ):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.grad_clip = grad_clip
        self.history = {"train_nll": [], "val_nll": []}

    def _epoch_batches(self, sessions):
        idx = np.arange(len(sessions))
        np.random.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            yield [sessions[i] for i in idx[start : start + self.batch_size]]

    def _compute_nll(self, batch):
        logits, _ = self.model(batch["inputs"])
        log_probs = F.log_softmax(logits, dim=-1)
        gather = log_probs.gather(-1, batch["targets"].unsqueeze(-1)).squeeze(-1)
        masked = gather * batch["mask"]
        return -masked.sum() / batch["mask"].sum().clamp_min(1.0)

    def fit(self, train_sessions, val_sessions):
        best = float("inf")
        best_state = None
        no_improve = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            tl = []
            for bs in self._epoch_batches(train_sessions):
                batch = collate_sessions(bs, self.model.num_actions, device=self.device)
                loss = self._compute_nll(batch)
                self.opt.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.opt.step()
                tl.append(float(loss.item()))
            self.model.eval()
            with torch.no_grad():
                vb = self._compute_nll(
                    collate_sessions(
                        val_sessions, self.model.num_actions, device=self.device
                    )
                )
            self.history["train_nll"].append(float(np.mean(tl)))
            self.history["val_nll"].append(float(vb.item()))
            if vb < best - 1e-6:
                best = float(vb.item())
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    @torch.no_grad()
    def evaluate(self, test_sessions):
        self.model.eval()
        batch = collate_sessions(
            test_sessions, self.model.num_actions, device=self.device
        )
        nll = self._compute_nll(batch)
        return {"test_nll": float(nll.item())}

    def hidden_trajectories(self, session):
        self.model.eval()
        batch = collate_sessions([session], self.model.num_actions, device=self.device)
        with torch.no_grad():
            logits, _ = self.model(batch["inputs"])
        return logits.squeeze(0).cpu().numpy()


__all__ = [
    "BanditModelFreeAgent2Arm",
    "BanditModelBasedRNN",
    "TinyBehaviorRNN",
    "TinyBehaviorRNNTrainer",
    "collate_sessions",
]

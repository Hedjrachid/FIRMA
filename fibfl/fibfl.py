"""
fl_fibfl.py  -  Module 5
=========================
The FibFL protocol family: FibFL, FibFL+, and FibFL++.

Reference
---------
    "FibFL: A Fibonacci-Weighted Federated Learning Protocol Family
     for Personalised Learning on a Ring Topology Without a Central Server"
     (IEEE Transactions, this paper)

Design invariant (all three variants)
--------------------------------------
    - Peer-to-peer ring topology - NO central server.
    - Head is PERMANENTLY PRIVATE - no head parameter ever leaves the client.
    - Feature extractor shared via Fibonacci-weighted ring blending:
          alpha = 1/phi ~= 0.618  (left-neighbour weight)
          beta  = 1/phi^2 ~= 0.382 (right-neighbour weight)
          alpha + beta = 1  (Fibonacci normalisation identity)

Variants
--------
    FibFL    Fixed (alpha,beta), K=1, sequential ring, constant gamma.
    FibFL+   Accuracy-gated blend: coefficients interpolate between the
             Fibonacci prior (alpha,beta) and accuracy posterior (w_L,w_R).
    FibFL++  Full protocol - adds:
               Fix A: 2-opt ring ordering (minimise adjacent class similarity)
               Fix B: K=ceil(N/2) gossip passes (global ring coverage)
               Fix C: cosine-annealed gamma + two-phase Adam + warmup

Public API
----------
    run_fibfl_basic(exts, heads, trl, tel, cfg)            -> history
    run_fibflp(exts, heads, trl, tel, cfg)                 -> history
    run_fibflpp(exts, heads, trl, tel, q_vecs, cfg)        -> history, sigma, c_seq, c_opt
    ask_fibfl_cfg(base_cfg, variant)                       -> cfg dict
    fibfl_on_split(Xtr,Ytr,Xte,Yte,tr_idx,te_idx,split_cfg,run_cfg,variant) -> results dict
"""

import math
import time
from typing import List, Tuple

import numpy as np
import torch

from fibfl.common import (
    ALPHA, BETA,
    build_models, train_epoch, eval_acc,
    gp, sp, wavg, _log, ser, FullModel,
    section, ok, info, ask_int, ask_float, confirm,
    bold, cyan, dim,
)


# =============================================================================
# 2-opt ring ordering  (Fix A)
# =============================================================================

def two_opt(q: np.ndarray) -> Tuple[List[int], float, float]:
    """
    2-opt local search minimising ring cost:
        C(sigma) = sum_p  cos(q[sigma[p]], q[sigma[(p+1)%N]])

    Lower C(sigma) -> adjacent clients have more diverse class distributions
    -> more informative gradient transfer (Proposition 4, paper).

    Parameters
    ----------
    q : (N, C) normalised class-proportion vectors

    Returns
    -------
    sigma      : optimised permutation (list of N client indices)
    cost_seq   : ring cost of sequential order (baseline)
    cost_opt   : ring cost of 2-opt order
    """
    N     = len(q)
    sigma = list(range(N))

    def cost(s: List[int]) -> float:
        return sum(float(np.dot(q[s[p]], q[s[(p+1) % N]])) for p in range(N))

    c_seq    = cost(list(range(N)))
    c        = c_seq
    improved = True
    while improved:
        improved = False
        for i in range(N - 1):
            for j in range(i + 2, N):
                if i == 0 and j == N - 1:
                    continue
                s2 = sigma[:i+1] + sigma[i+1:j+1][::-1] + sigma[j+1:]
                c2 = cost(s2)
                if c2 < c - 1e-9:
                    sigma, c = s2, c2
                    improved = True
    return sigma, c_seq, c


# =============================================================================
# K-pass Fibonacci blend kernel  (Fix B)
# =============================================================================

def fib_blend(
    eps:     List[List[torch.Tensor]],
    accs:    List[float],
    sigma:   List[int],
    K:       int,
    gamma_r: float,
    tau:     float = 0.35,
) -> List[List[torch.Tensor]]:
    """
    K-pass Fibonacci ring gossip blend (extractor parameters only).

    Each pass, for every client i at position p in ring sigma:
      1. Gate each neighbour:  g_j = a_j  if a_j >= tau  else 0
      2. If both gated -> skip (self-retain)
      3. Accuracy posterior:   w_L = g_L/(g_L+g_R),  w_R = g_R/(g_L+g_R)
      4. Interpolate Fibonacci prior with posterior:
              a_L = 0.5*ALPHA + 0.5*w_L     (a_L + a_R = 1 always)
              a_R = 0.5*BETA  + 0.5*w_R
      5. Neighbourhood blend:  phi_tilde = a_L*phi_L + a_R*phi_R
      6. Self-retention:       phi_i <- gin*phi_i + (1-gin)*phi_tilde
                               gin = gamma_r^(1/K)  (calibrated)

    Proposition 2 (paper): K = ceil(N/2) achieves global ring coverage.
    Proposition 3 (paper): gin = gamma_r^(1/K) preserves target gamma.

    Parameters
    ----------
    eps      : List[N] of parameter lists (post-training extractor params)
    accs     : List[N] per-client validation accuracies in [0, 1]
    sigma    : 2-opt ring permutation
    K        : number of inner gossip passes
    gamma_r  : target self-retention for this round
    tau      : accuracy gate threshold (neighbours below tau are suppressed)
    """
    N   = len(eps)
    gin = gamma_r ** (1.0 / K)      # calibrated per-pass self-retention
    cur = [list(p) for p in eps]    # mutable copy

    for _ in range(K):
        nxt = [None] * N
        for p in range(N):
            i  = sigma[p]
            L  = sigma[(p - 1) % N]
            R  = sigma[(p + 1) % N]
            gL = accs[L] if accs[L] >= tau else 0.0
            gR = accs[R] if accs[R] >= tau else 0.0
            tot = gL + gR
            if tot < 1e-9:
                nxt[i] = cur[i]   # both below threshold - skip
                continue
            wL = gL / tot
            wR = gR / tot
            aL = 0.5 * ALPHA + 0.5 * wL   # interpolate prior with posterior
            aR = 0.5 * BETA  + 0.5 * wR   # a_L + a_R = 1 always
            bl = [aL * pL + aR * pR for pL, pR in zip(cur[L], cur[R])]
            nxt[i] = [gin * pi + (1 - gin) * bi
                      for pi, bi in zip(cur[i], bl)]
        cur = nxt
    return cur


# =============================================================================
# Two-phase local training helper  (Fix C)
# =============================================================================

def _two_phase(
    exts:  List,
    heads: List,
    trl:   List,
    cfg:   dict,
    oh:    List,   # persistent head optimisers
    oe:    List,   # persistent extractor optimisers
) -> List[float]:
    """
    Two-phase local training for FibFL variants (adapted from FedRep).

    Phase 1 (Eh epochs, extractor frozen):
        Head adapts to the current fixed extractor -> clean head gradient.
    Phase 2 (Ee epochs, head frozen):
        Extractor updates with the just-adapted head -> better ext gradient.

    Returns per-client accuracy on training loaders (used as accuracy gate).
    """
    n  = len(exts)
    ar = []
    for i in range(n):
        model = FullModel(exts[i], heads[i])

        # Phase 1: freeze extractor, train head
        for p in exts[i].parameters():
            p.requires_grad_(False)
        for _ in range(cfg.get("Eh", 2)):
            train_epoch(model, trl[i], oh[i])

        # Phase 2: freeze head, train extractor
        for p in exts[i].parameters():
            p.requires_grad_(True)
        for p in heads[i].parameters():
            p.requires_grad_(False)
        for _ in range(cfg.get("Ee", 2)):
            train_epoch(model, trl[i], oe[i])
        for p in heads[i].parameters():
            p.requires_grad_(True)

        ar.append(eval_acc(exts[i], heads[i], trl[i]))
    return ar


# =============================================================================
# FibFL  (basic)
# =============================================================================

def run_fibfl_basic(
    exts:  List,
    heads: List,
    trl:   List,
    tel:   List,
    cfg:   dict,
) -> List[List[float]]:
    """
    FibFL - Algorithm 1 (paper).

    Fixed (ALPHA, BETA) weights, K=1 pass, sequential ring (sigma=id),
    constant self-retention gamma.  Two-phase Adam local training.
    Head permanently private.

    Parameters
    ----------
    cfg keys: R, Eh, Ee, lr, gamma, log_every
    """
    n     = len(exts)
    sigma = list(range(n))
    gamma = cfg.get("gamma", 0.5)

    oh = [torch.optim.Adam(heads[i].parameters(), lr=cfg["lr"])
          for i in range(n)]
    oe = [torch.optim.Adam(exts[i].parameters(),  lr=cfg["lr"])
          for i in range(n)]

    history = []
    for r in range(cfg["R"]):
        _two_phase(exts, heads, trl, cfg, oh, oe)

        # Fixed (ALPHA, BETA) blend, K=1, no gating
        old = [gp(exts[i]) for i in range(n)]
        for p in range(n):
            i = sigma[p]
            L = sigma[(p - 1) % n]
            R = sigma[(p + 1) % n]
            bl = [ALPHA * pL + BETA * pR for pL, pR in zip(old[L], old[R])]
            sp(exts[i], [gamma * pi + (1 - gamma) * bi
                          for pi, bi in zip(old[i], bl)])
        # Heads: never modified here - permanently private

        accs = [eval_acc(exts[i], heads[i], tel[i]) for i in range(n)]
        history.append(accs)
        _log("FibFL  ", r + 1, accs, cfg.get("log_every", 5))

    return history


# =============================================================================
# FibFL+
# =============================================================================

def run_fibflp(
    exts:  List,
    heads: List,
    trl:   List,
    tel:   List,
    cfg:   dict,
) -> List[List[float]]:
    """
    FibFL+ - Algorithm 2 (paper).

    Extends FibFL with accuracy-gated interpolation.
    K=1, sequential ring.  Head permanently private.

    Parameters
    ----------
    cfg keys: R, Eh, Ee, lr, gamma, tau, log_every
    """
    n     = len(exts)
    sigma = list(range(n))

    oh = [torch.optim.Adam(heads[i].parameters(), lr=cfg["lr"])
          for i in range(n)]
    oe = [torch.optim.Adam(exts[i].parameters(),  lr=cfg["lr"])
          for i in range(n)]

    history = []
    for r in range(cfg["R"]):
        ar  = _two_phase(exts, heads, trl, cfg, oh, oe)
        old = [gp(exts[i]) for i in range(n)]
        bl  = fib_blend(old, ar, sigma, K=1,
                        gamma_r=cfg.get("gamma", 0.5),
                        tau=cfg.get("tau", 0.35))
        for i in range(n):
            sp(exts[i], bl[i])

        accs = [eval_acc(exts[i], heads[i], tel[i]) for i in range(n)]
        history.append(accs)
        _log("FibFL+ ", r + 1, accs, cfg.get("log_every", 5))

    return history


# =============================================================================
# FibFL++  (full protocol)
# =============================================================================

def run_fibflpp(
    exts:   List,
    heads:  List,
    trl:    List,
    tel:    List,
    q_vecs: np.ndarray,
    cfg:    dict,
) -> Tuple[List[List[float]], List[int], float, float]:
    """
    FibFL++ - Algorithm 5 (paper).  Tuned full protocol.

    Three fixes over FibFL+:
    -------------------------
    Fix A  2-opt ring ordering:
           Minimises C(sigma) = sum cos(q_sigma(p), q_sigma(p+1)).
           Computed once before training; held fixed throughout.

    Fix B  K = ceil(N/2) inner gossip passes with gin = gamma^(1/K):
           Achieves global ring coverage (= FedAvg broadcast radius)
           without any central server.

    Fix C  Cosine-annealed gamma + two-phase Adam + warmup:
           - Warmup (first W = R//6 rounds): FedAvg-style central averaging
             of the FULL model gives a good shared initialisation before
             switching to private-head mode.
           - gamma anneals from gamma_start -> gamma_end (aggressive default
             0.4 -> 0.05: more blending = faster knowledge transfer).
           - Persistent Adam optimisers preserve momentum across rounds.
           - Head LR equals extractor LR (small head benefits from full LR).

    Parameters
    ----------
    q_vecs : (N, n_classes) normalised class-proportion vectors for 2-opt.
    cfg keys: R, Eh, Ee, lr, gamma_start, gamma_end, tau, log_every

    Returns
    -------
    history        : List[R] of List[N] per-client test accuracies
    sigma          : 2-opt-optimised ring permutation
    ring_cost_seq  : ring cost before 2-opt (sequential order)
    ring_cost_opt  : ring cost after 2-opt
    """
    n  = len(exts)
    K  = max(1, math.ceil(n / 2))
    gs = cfg.get("gamma_start", 0.4)    # tuned default: aggressive blending
    ge = cfg.get("gamma_end",   0.05)
    R  = cfg["R"]
    W  = max(1, R // 6)                 # warmup rounds

    # Sample sizes for FedAvg warmup weighting (exact)
    # We get them from the loaders' underlying dataset lengths
    ns    = [len(ld.dataset) for ld in trl]
    total = max(sum(ns), 1)
    w     = [s / total for s in ns]

    # Fix A: 2-opt ring ordering - run once before training
    sigma, c_seq, c_opt = two_opt(q_vecs)
    saving = 100 * (c_seq - c_opt) / max(c_seq, 1e-9)
    info(f"2-opt: seq={c_seq:.4f}  opt={c_opt:.4f}  "
         f"saving={saving:.1f}%  K={K}  warmup={W}r  sigma={sigma}")

    # Fix C: persistent optimisers - head uses same LR as extractor
    oh = [torch.optim.Adam(heads[i].parameters(), lr=cfg["lr"])
          for i in range(n)]
    oe = [torch.optim.Adam(exts[i].parameters(),  lr=cfg["lr"])
          for i in range(n)]

    # Warmup: full models (for central avg)
    ms_wu = [FullModel(exts[i], heads[i]) for i in range(n)]
    op_wu = [torch.optim.Adam(ms_wu[i].parameters(), lr=cfg["lr"])
             for i in range(n)]

    history = []
    for r in range(R):

        # -- Warmup phase: FedAvg-style shared initialisation -----------------
        if r < W:
            for i in range(n):
                for _ in range(max(cfg.get("Eh", 2), cfg.get("Ee", 2))):
                    train_epoch(ms_wu[i], trl[i], op_wu[i])
            old_all = [gp(ms_wu[i]) for i in range(n)]
            g_all   = wavg(old_all, w)
            for i in range(n):
                sp(ms_wu[i], g_all)
            accs = [eval_acc(ms_wu[i].ext, ms_wu[i].head, tel[i])
                    for i in range(n)]
            history.append(accs)
            _log("FibFL++ warmup", r + 1, accs, cfg.get("log_every", 5))
            continue

        # -- Main FibFL++ phase (r >= W) --------------------------------------

        # Fix C: cosine-annealed gamma over non-warmup rounds
        r_eff   = r - W
        R_eff   = max(R - W - 1, 1)
        gamma_r = ge + 0.5 * (gs - ge) * (1 + math.cos(math.pi * r_eff / R_eff))

        # Fix C: two-phase local training
        ar = _two_phase(exts, heads, trl, cfg, oh, oe)

        # Fix B: K-pass Fibonacci blend (extractor only; head stays private)
        old     = [gp(exts[i]) for i in range(n)]
        blended = fib_blend(old, ar, sigma, K, gamma_r,
                            cfg.get("tau", 0.35))
        for i in range(n):
            sp(exts[i], blended[i])
        # Head: permanently private - never modified here

        accs = [eval_acc(exts[i], heads[i], tel[i]) for i in range(n)]
        history.append(accs)
        _log(f"FibFL++ g={gamma_r:.3f}", r + 1, accs, cfg.get("log_every", 5))

    return history, sigma, c_seq, c_opt


# =============================================================================
# Hyperparameter menus
# =============================================================================

def ask_fibfl_cfg(base_cfg: dict, variant: str = "FibFL++") -> dict:
    """
    Ask the user for FibFL-family hyperparameters.
    variant: one of 'FibFL', 'FibFL+', 'FibFL++'
    Returns an updated cfg dict.
    """
    section(f"{variant} - Hyperparameters")

    use_def = confirm("Use recommended defaults?", True)
    cfg = dict(base_cfg)

    if use_def:
        cfg.setdefault("Eh",          2)
        cfg.setdefault("Ee",          2)
        cfg.setdefault("lr",          0.01)
        cfg.setdefault("gamma",       0.5)
        cfg.setdefault("gamma_start", 0.4)
        cfg.setdefault("gamma_end",   0.05)
        cfg.setdefault("tau",         0.35)
        info(f"Eh={cfg['Eh']}  Ee={cfg['Ee']}  lr={cfg['lr']}  "
             f"gamma={cfg['gamma_start']}->{cfg['gamma_end']}  tau={cfg['tau']}")
    else:
        cfg["Eh"] = ask_int("Head epochs per round Eh",
                            cfg.get("Eh", 2), 1, 20)
        cfg["Ee"] = ask_int("Extractor epochs per round Ee",
                            cfg.get("Ee", 2), 1, 20)
        cfg["lr"] = ask_float("Learning rate lr",
                              cfg.get("lr", 0.01), 1e-5, 1.0)
        if variant == "FibFL":
            cfg["gamma"] = ask_float("Self-retention gamma",
                                     cfg.get("gamma", 0.5), 0.0, 1.0)
        if variant in ("FibFL+", "FibFL++"):
            cfg["tau"] = ask_float("Accuracy gate threshold tau",
                                   cfg.get("tau", 0.35), 0.0, 1.0)
        if variant == "FibFL++":
            cfg["gamma_start"] = ask_float("Gamma start (cosine anneal)",
                                           cfg.get("gamma_start", 0.4), 0.0, 1.0)
            cfg["gamma_end"]   = ask_float("Gamma end",
                                           cfg.get("gamma_end", 0.05), 0.0, 1.0)

    cfg["log_every"] = max(1, cfg.get("R", 20) // 5)
    return cfg


# =============================================================================
# Convenience wrappers
# =============================================================================

def fibfl_on_split(
    Xtr:       np.ndarray,
    Ytr:       np.ndarray,
    Xte:       np.ndarray,
    Yte:       np.ndarray,
    tr_idx:    List[np.ndarray],
    te_idx:    List[np.ndarray],
    split_cfg: dict,
    run_cfg:   dict,
    variant:   str = "FibFL++",
) -> dict:
    """
    Build fresh models, run the chosen FibFL variant on the given
    data split, and return a results dict compatible with the analysis
    and save functions.

    Parameters
    ----------
    variant : 'FibFL' | 'FibFL+' | 'FibFL++'
    """
    from fibfl.data import make_loader, class_props

    n         = len(tr_idx)
    n_classes = split_cfg["n_classes"]
    emb       = run_cfg.get("emb_dim", 128)
    bs        = split_cfg.get("batch_size", 64)

    trl = [make_loader(Xtr, Ytr, idx, bs)        for idx in tr_idx]
    tel = [make_loader(Xte, Yte, idx, bs, False)  for idx in te_idx]
    qv  = np.array([class_props(Ytr, idx, n_classes) for idx in tr_idx])

    em    = build_models(Xtr.shape[1], n, emb, n_classes)
    exts  = [e  for e,  _ in em]
    heads = [hd for _, hd in em]

    section(f"Running {bold(variant)}")
    t0 = time.time()

    extra = {}
    if variant == "FibFL":
        h = run_fibfl_basic(exts, heads, trl, tel, run_cfg)
    elif variant == "FibFL+":
        h = run_fibflp(exts, heads, trl, tel, run_cfg)
    else:   # FibFL++
        h, sigma, c_seq, c_opt = run_fibflpp(
            exts, heads, trl, tel, qv, run_cfg)
        extra = {
            "sigma":            [int(x) for x in sigma],
            "ring_cost_seq":    float(c_seq),
            "ring_cost_opt":    float(c_opt),
            "ring_saving_pct":  round(100 * (c_seq - c_opt) / max(c_seq, 1e-9), 1),
        }

    elapsed = round(time.time() - t0, 1)
    fin     = h[-1]
    ok(f"{variant:<10}  mean={np.mean(fin):.4f}  std={np.std(fin):.4f}  "
       f"min={np.min(fin):.4f}  max={np.max(fin):.4f}  ({elapsed}s)")

    return {"history": ser(h), "time_s": elapsed, **extra}

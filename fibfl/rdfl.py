"""
fl_rdfl.py  
========================
RDFL: Modified version of Ring Decentralised Federated Learning (Wang et al., 2021).

Reference
---------
    Wang et al., "Efficient Ring-Topology Decentralized Federated Learning
    with Deep Generative Models", arXiv 2104.08100, 2021.

Design
------
    - Peer-to-peer ring topology - NO central server.
    - Full model shared (extractor + head) - NO head privacy.
    - Uniform blend weights (1/2, 1/2) between left and right neighbours.
    - Self-retention gamma after blending:
          new_params = gamma * own + (1-gamma) * mean(left, right)
    - Sequential ring order (no 2-opt optimisation).
    - Local optimiser: SGD with momentum=0.9.

Public API
----------
    run_rdfl(exts, heads, trl, tel, cfg) -> history
    ask_rdfl_cfg(base_cfg)               -> cfg dict
    rdfl_on_split(...)                   -> results dict
"""

import time
from typing import List

import numpy as np
import torch

from fibfl.common import (
    build_models, train_epoch, eval_acc,
    gp, sp, wavg, lerp, _log, ser, FullModel,
    section, ok, info, ask_int, ask_float, confirm,
    bold, dim, green,
)


# =============================================================================
# RDFL runner
# =============================================================================

def run_rdfl(
    exts:  List,
    heads: List,
    trl:   List,
    tel:   List,
    cfg:   dict,
) -> List[List[float]]:
    """
    RDFL (Wang et al. 2021).

    Every round:
      1.  Each client trains the FULL model for E epochs with SGD.
      2.  Ring aggregation with UNIFORM (1/2, 1/2) weights on the FULL model:
              blend = 0.5 * params_left + 0.5 * params_right
              new   = gamma * own + (1-gamma) * blend
      3.  No server, no head privacy.

    Parameters
    ----------
    exts, heads  : per-client model components
    trl, tel     : train / test DataLoaders per client
    cfg          : run configuration (R, E, lr, gamma, log_every)

    Returns
    -------
    history : List[R] of List[N] per-client test accuracies
    """
    n     = len(exts)
    gamma = cfg.get("gamma", 0.5)
    ms    = [FullModel(exts[i], heads[i]) for i in range(n)]
    opts  = [torch.optim.SGD(m.parameters(),
                             lr=cfg["lr"], momentum=0.9)
             for m in ms]

    history = []
    for r in range(cfg["R"]):
        # Local training
        for i in range(n):
            for _ in range(cfg.get("E", 2)):
                train_epoch(ms[i], trl[i], opts[i])

        # Ring aggregation - snapshot before update
        old = [gp(m) for m in ms]
        for i in range(n):
            L  = (i - 1) % n
            R  = (i + 1) % n
            bl = wavg([old[L], old[R]], [0.5, 0.5])   # uniform blend
            sp(ms[i], lerp(old[i], bl, gamma))         # self-retention

        accs = [eval_acc(m.ext, m.head, tel[i])
                for i, m in enumerate(ms)]
        history.append(accs)
        _log("RDFL", r + 1, accs, cfg.get("log_every", 5))

    return history


# =============================================================================
# Hyperparameter menu
# =============================================================================

def ask_rdfl_cfg(base_cfg: dict) -> dict:
    """
    Ask the user for RDFL-specific hyperparameters.
    Returns an updated cfg dict ready to pass to run_rdfl().
    """
    section("RDFL - Hyperparameters")

    use_def = confirm("Use recommended defaults?", True)
    cfg = dict(base_cfg)

    if use_def:
        cfg.setdefault("E",     2)
        cfg.setdefault("lr",    0.01)
        cfg.setdefault("gamma", 0.5)
        info(f"E={cfg['E']}  lr={cfg['lr']}  gamma={cfg['gamma']}")
    else:
        cfg["E"]     = ask_int("Local epochs E", cfg.get("E", 2), 1, 30)
        cfg["lr"]    = ask_float("Learning rate lr",
                                 cfg.get("lr", 0.01), 1e-5, 1.0)
        cfg["gamma"] = ask_float("Self-retention gamma (0=full blend, 1=no blend)",
                                 cfg.get("gamma", 0.5), 0.0, 1.0)

    cfg["log_every"] = max(1, cfg.get("R", 20) // 5)
    return cfg


# =============================================================================
# Convenience wrapper
# =============================================================================

def rdfl_on_split(
    Xtr: np.ndarray, Ytr: np.ndarray,
    Xte: np.ndarray, Yte: np.ndarray,
    tr_idx: List[np.ndarray],
    te_idx: List[np.ndarray],
    split_cfg: dict,
    run_cfg:   dict,
) -> dict:
    """
    Build fresh models, run RDFL on the given data split, and return
    a results dict compatible with the analysis / save functions.
    """
    from fibfl.data import make_loader

    n         = len(tr_idx)
    n_classes = split_cfg["n_classes"]
    emb       = run_cfg.get("emb_dim", 128)
    bs        = split_cfg.get("batch_size", 64)

    trl = [make_loader(Xtr, Ytr, idx, bs)        for idx in tr_idx]
    tel = [make_loader(Xte, Yte, idx, bs, False)  for idx in te_idx]

    em    = build_models(Xtr.shape[1], n, emb, n_classes)
    exts  = [e  for e,  _ in em]
    heads = [hd for _, hd in em]

    section(f"Running {bold('RDFL')}")
    t0 = time.time()
    h  = run_rdfl(exts, heads, trl, tel, run_cfg)
    elapsed = round(time.time() - t0, 1)

    fin = h[-1]
    ok(f"RDFL    mean={np.mean(fin):.4f}  std={np.std(fin):.4f}  "
       f"min={np.min(fin):.4f}  max={np.max(fin):.4f}  ({elapsed}s)")

    return {"history": ser(h), "time_s": elapsed}

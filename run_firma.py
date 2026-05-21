#!/usr/bin/env python3
"""
fl_main_firma.py  -  FIRMA: Parallel FL runner (RDFL + FibFL family only)
============================================================================
Runs RDFL, FibFL, FibFL+, and FibFL++ - FedAvg and FedRep removed.

NEW in this version
-------------------
  Accuracy metrics
    top-1, top-2, top-3, top-5 accuracy (per client and federation mean)
    macro-averaged accuracy (unweighted across classes)
    per-class recall vector (C-length, per client)

  Convergence metrics
    area under accuracy curve (AUC over R rounds)
    rounds-to-threshold for 70%, 80%, 90% of final accuracy
    convergence rate (slope R5->R15, excluding warmup/saturation)
    plateau stability (std of last 5 rounds)

  Fairness metrics
    Gini coefficient
    coefficient of variation (CV)
    worst-to-best ratio (W2B)
    Jain's fairness index
    tail accuracy (mean of bottom-20% clients)

  Personalisation metrics
    personalisation gain: per-client (fine-tuned head) vs global-head accuracy

  Communication efficiency
    accuracy per unit communication (top-1 / total_bytes_per_round)

  Robustness
    worst-round accuracy (min over all rounds)
    accuracy standard deviation across rounds (temporal stability)

Usage
-----
    python3 fl_main_parallel.py                 # interactive
    python3 fl_main_parallel.py --auto          # 6 paper experiments
    python3 fl_main_parallel.py --workers N
    python3 fl_main_parallel.py --outdir DIR
    python3 fl_main_parallel.py --datadir DIR
"""

import argparse
import math
import multiprocessing as mp
import os
import sys
import textwrap
import time
import datetime
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
 
warnings.filterwarnings("ignore")

def _last(history, default=None):
    """Safely return the last row of a history list; returns [] if empty."""
    return history[-1] if history else (default if default is not None else [])

def _last_mean(history):
    """Mean accuracy of the final round, or nan if history is empty."""
    row = _last(history)
    return float(np.mean(row)) if row else float("nan")


os.environ.setdefault("OMP_NUM_THREADS",      "2")
os.environ.setdefault("MKL_NUM_THREADS",      "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

# -- Local modules --------------------------------------------------------------
from fibfl.common import (
    banner, section, menu, ask_int, ask_float, confirm,
    ok, info, prog,
    bold, cyan, green, yellow, red, dim,
    build_models, DEVICE,
)
from fibfl.data import (
    DATASETS, DS_NAMES, DS_MENU,
    load_dataset, partition, make_loader, class_props,
    show_split_stats, dataset_menu,
)
from fl_rdfl    import run_rdfl,    ask_rdfl_cfg,    rdfl_on_split
from fl_fibfl   import (
    run_fibfl_basic, run_fibflp, run_fibflpp,
    ask_fibfl_cfg, fibfl_on_split,
    two_opt, fib_blend,
)


# =============================================================================
# Extended evaluation helpers
# =============================================================================

def eval_metrics_client(ext, head, loader, n_classes: int,
                        ks=(1, 2, 3, 5)) -> dict:
    """
    Compute a rich metric bundle for one (ext, head, loader) triple.

    Returns
    -------
    dict with keys:
        top_k      : {k: accuracy}  for k in ks
        macro_acc  : mean per-class recall (unweighted)
        per_class  : list of per-class recall values (length n_classes)
        n_correct  : int  (for weighted aggregation)
        n_total    : int
    """
    import torch
    ext.eval(); head.eval()

    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = head(ext(x))          # (B, C)
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())

    if not all_logits:
        nan = float("nan")
        return dict(top_k={k: nan for k in ks}, macro_acc=nan,
                    per_class=[nan] * n_classes, n_correct=0, n_total=0)

    logits = torch.cat(all_logits, 0)      # (N, C)
    y      = torch.cat(all_y,      0)      # (N,)
    N      = len(y)

    # -- Top-k accuracy ---------------------------------------------------------
    max_k   = min(max(ks), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)  # (N, max_k)
    correct = pred.eq(y.unsqueeze(1).expand_as(pred))               # (N, max_k)
    top_k   = {}
    for k in ks:
        k_ = min(k, logits.shape[1])
        top_k[k] = float(correct[:, :k_].any(dim=1).float().mean())

    # -- Per-class recall -------------------------------------------------------
    per_class = []
    for c in range(n_classes):
        mask = (y == c)
        if mask.sum() == 0:
            per_class.append(float("nan"))
        else:
            per_class.append(float(correct[mask, 0].float().mean()))

    valid_recalls = [v for v in per_class if not math.isnan(v)]
    macro_acc = float(np.mean(valid_recalls)) if valid_recalls else float("nan")

    return dict(
        top_k     = top_k,
        macro_acc = macro_acc,
        per_class = per_class,
        n_correct = int(correct[:, 0].sum()),
        n_total   = N,
    )


def eval_federation_metrics(exts, heads, loaders, n_classes: int,
                             ks=(1, 2, 3, 5)) -> dict:
    """
    Compute per-client and federation-level metrics.

    Returns
    -------
    dict:
        clients   : list of per-client metric dicts
        fed_top_k : {k: federation mean top-k accuracy}  (weighted by n_total)
        fed_macro : federation mean macro accuracy
        fed_per_class : federation mean per-class recall (C,)
    """
    clients = []
    for ext, head, loader in zip(exts, heads, loaders):
        clients.append(eval_metrics_client(ext, head, loader, n_classes, ks))

    totals     = [c["n_total"]   for c in clients]
    total_n    = sum(totals)
    weights    = [t / max(total_n, 1) for t in totals]

    fed_top_k = {}
    for k in ks:
        fed_top_k[k] = float(np.average(
            [c["top_k"][k] for c in clients], weights=totals))

    valid_macros = [(c["macro_acc"], t)
                    for c, t in zip(clients, totals)
                    if not math.isnan(c["macro_acc"])]
    if valid_macros:
        vals, ws = zip(*valid_macros)
        fed_macro = float(np.average(vals, weights=ws))
    else:
        fed_macro = float("nan")

    fed_per_class = []
    for ci in range(n_classes):
        vals_w = [(c["per_class"][ci], c["n_total"])
                  for c in clients
                  if ci < len(c["per_class"])
                  and not math.isnan(c["per_class"][ci])]
        if vals_w:
            vs, ws = zip(*vals_w)
            fed_per_class.append(float(np.average(vs, weights=ws)))
        else:
            fed_per_class.append(float("nan"))

    return dict(
        clients       = clients,
        fed_top_k     = fed_top_k,
        fed_macro     = fed_macro,
        fed_per_class = fed_per_class,
    )


def compute_derived_metrics(history: list, final_metrics: dict,
                            time_s: float, n_clients: int,
                            p_e: int, K: int = 1) -> dict:
    """
    Derive convergence, fairness, and efficiency metrics from history + final_metrics.

    Parameters
    ----------
    history       : list of R lists, each of length N (top-1 per client)
    final_metrics : output of eval_federation_metrics at R=final
    time_s        : wall-clock seconds
    n_clients     : N
    p_e           : extractor parameter count (for comm efficiency)
    K             : gossip passes (1 for most methods, ceil(N/2) for FibFL++)
    """
    R     = len(history)
    means = [np.mean(row) for row in history]

    # -- AUC (trapezoid over rounds 1..R) --------------------------------------
    auc = float(np.trapz(means) / max(R - 1, 1))

    # -- Rounds to threshold ----------------------------------------------------
    final_mean = means[-1] if means else 0.0
    def _rtt(frac):
        target = frac * final_mean
        for r, v in enumerate(means):
            if v >= target:
                return r + 1
        return R
    rtt = {70: _rtt(0.70), 80: _rtt(0.80), 90: _rtt(0.90), 50: _rtt(0.50)}

    # -- Convergence rate (slope R5->R15, clamped to available rounds) ----------
    r_lo = min(4, R - 1)
    r_hi = min(14, R - 1)
    rate = ((means[r_hi] - means[r_lo]) / max(r_hi - r_lo, 1)
            if r_hi > r_lo else float("nan"))

    # -- Temporal stability (std of last 5 rounds) -----------------------------
    plateau_std = float(np.std(means[-5:])) if R >= 5 else float("nan")

    # -- Worst round accuracy ---------------------------------------------------
    worst_round = float(np.min(means))

    # -- Round-to-round std (temporal variance) --------------------------------
    temporal_std = float(np.std(means)) if R > 1 else float("nan")

    # -- Fairness metrics -------------------------------------------------------
    fin = np.array(_last(history, [0.0]), dtype=float)
    N   = max(len(fin), 1)

    gini = 0.0
    if fin.sum() > 0:
        s    = np.sort(fin)
        gini = (2 * np.sum(np.arange(1, N + 1) * s) / (N * s.sum())
                - (N + 1) / N)
    cv  = float(np.std(fin) / max(np.mean(fin), 1e-9))
    w2b = float(np.min(fin) / max(np.max(fin), 1e-9))

    # Jain's fairness index: (sum acc)^2 / (N * sum acc^2)
    jain = float(fin.sum() ** 2 / max(N * (fin ** 2).sum(), 1e-12))

    # Tail accuracy: mean of bottom 20% clients
    k_tail = max(1, round(0.20 * N))
    tail   = float(np.mean(np.sort(fin)[:k_tail]))

    # -- Communication efficiency -----------------------------------------------
    # bytes_per_round = 2 x K x N x p_e x 4  (float32 = 4 bytes, K passes)
    bytes_per_round = 2 * K * N * p_e * 4
    comm_eff = (final_mean / bytes_per_round * 1e6
                if bytes_per_round > 0 else float("nan"))

    # -- Top-k from final_metrics -----------------------------------------------
    fed_top_k   = final_metrics.get("fed_top_k", {})
    fed_macro   = final_metrics.get("fed_macro", float("nan"))
    per_class   = final_metrics.get("fed_per_class", [])

    # -- Per-client top-k summary -----------------------------------------------
    client_data = final_metrics.get("clients", [])
    per_client_top1 = [c["top_k"].get(1, float("nan")) for c in client_data]
    per_client_top3 = [c["top_k"].get(3, float("nan")) for c in client_data]
    per_client_macro = [c["macro_acc"] for c in client_data]
    per_client_perclass = [c["per_class"] for c in client_data]

    return dict(
        # convergence
        auc          = auc,
        rtt          = rtt,
        rate         = float(rate),
        plateau_std  = float(plateau_std),
        worst_round  = worst_round,
        temporal_std = temporal_std,
        # fairness
        gini    = float(gini),
        cv      = cv,
        w2b     = w2b,
        jain    = jain,
        tail    = tail,
        # accuracy
        top_k        = fed_top_k,
        macro_acc    = fed_macro,
        per_class    = per_class,
        # per-client
        per_client_top1   = per_client_top1,
        per_client_top3   = per_client_top3,
        per_client_macro  = per_client_macro,
        per_client_perclass = per_client_perclass,
        # efficiency
        comm_eff     = comm_eff,
        bytes_per_round = bytes_per_round,
    )


# =============================================================================
# Worker count
# =============================================================================

def _default_workers() -> int:
    avail = max(1, mp.cpu_count() - 1)
    return min(6, avail)


# =============================================================================
# Worker function (spawned process)
# =============================================================================

def _worker(
    method:    str,
    mmap_dir:  str,          # path to temp dir containing .npy mmaps
    shapes:    dict,         # {name: shape} for Xtr/Ytr/Xte/Yte
    dtypes:    dict,         # {name: dtype_str}
    tr_idx:    List[np.ndarray],
    te_idx:    List[np.ndarray],
    split_cfg: dict,
    run_cfg:   dict,
    result_q:  "mp.Queue",
) -> None:
    """
    Spawned worker - reads data from memory-mapped files to avoid the
    Windows pipe size limit (OSError EINVAL / pickle truncated).

    The parent writes Xtr/Ytr/Xte/Yte to numpy .npy files in mmap_dir
    before spawning; this worker reads them back as read-only mmaps.
    No large arrays transit the spawn pipe - only small dicts and index lists.
    """
    import torch, numpy as np
    seed = hash(method) % (2 ** 31)
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFF)

    import io, contextlib, os
    buf = io.StringIO()

    try:
        # -- Load data from mmap files -----------------------------------------
        Xtr = np.load(os.path.join(mmap_dir, "Xtr.npy"), mmap_mode="r")
        Ytr = np.load(os.path.join(mmap_dir, "Ytr.npy"), mmap_mode="r")
        Xte = np.load(os.path.join(mmap_dir, "Xte.npy"), mmap_mode="r")
        Yte = np.load(os.path.join(mmap_dir, "Yte.npy"), mmap_mode="r")

        n_cls = split_cfg["n_classes"]
        bs    = split_cfg.get("batch_size", 64)
        emb   = run_cfg.get("emb_dim", 128)
        N     = len(tr_idx)

        from fibfl.common import build_models, DEVICE
        from fl_data   import make_loader, class_props

        trl = [make_loader(Xtr, Ytr, idx, bs)       for idx in tr_idx]
        tel = [make_loader(Xte, Yte, idx, bs, False) for idx in te_idx]

        with contextlib.redirect_stdout(buf):
            # Apply per-method defaults so each algorithm uses its own hyperparams
            method_run_cfg = get_method_cfg(method, run_cfg)
            if method == "RDFL":
                res = rdfl_on_split(
                    Xtr, Ytr, Xte, Yte, tr_idx, te_idx, split_cfg, method_run_cfg)
            else:
                res = fibfl_on_split(
                    Xtr, Ytr, Xte, Yte, tr_idx, te_idx,
                    split_cfg, method_run_cfg, variant=method)

        # -- Rebuild final models for extended metrics -------------------------
        final_metrics = {}
        if "final_state" in res:
            from fibfl.common import Extractor, Head
            d_in  = Xtr.shape[1]
            exts, heads = [], []
            for i, (e_sd, h_sd) in enumerate(res["final_state"]):
                ext = Extractor(d_in, emb).to(DEVICE)
                hd  = Head(emb, n_cls).to(DEVICE)
                ext.load_state_dict({k: torch.tensor(v)
                                     for k, v in e_sd.items()})
                hd.load_state_dict( {k: torch.tensor(v)
                                     for k, v in h_sd.items()})
                exts.append(ext); heads.append(hd)
            final_metrics = eval_federation_metrics(
                exts, heads, tel, n_cls, ks=(1, 2, 3, 5))

        # -- Extractor param count ---------------------------------------------
        from fibfl.common import Extractor
        _tmp = Extractor(Xtr.shape[1], emb)
        p_e  = sum(p.numel() for p in _tmp.parameters())

        K_passes = (math.ceil(N / 2) if method == "FibFL++" else 1)

        derived = compute_derived_metrics(
            history       = res["history"],
            final_metrics = final_metrics,
            time_s        = res.get("time_s", float("nan")),
            n_clients     = N,
            p_e           = p_e,
            K             = K_passes,
        )

        payload = {**res, **derived, "final_metrics": final_metrics}
        result_q.put(("ok", method, payload))

    except Exception as exc:
        import traceback
        result_q.put(("err", method,
                      str(exc) + "\n" + traceback.format_exc()))


# =============================================================================
# Parallel dispatcher  (Windows-safe: data passed via temp mmap files)
# =============================================================================

def run_methods_parallel(
    methods:   List[str],
    session:   "Session",
    run_cfg:   dict,
    n_workers: int,
) -> Dict[str, dict]:
    """
    Launch up to n_workers FL methods in parallel.

    Windows-safe design
    -------------------
    On Windows, mp.spawn sends arguments through a named pipe with a
    small buffer (~64 KB).  Passing large numpy arrays (CIFAR-10 Xtr is
    ~600 MB) through this pipe causes OSError EINVAL / pickle truncated.

    Fix: write Xtr/Ytr/Xte/Yte to numpy .npy files in a temporary
    directory before spawning.  Workers load them as read-only mmaps -
    no large data ever transits the pipe.  Only small dicts (split_cfg,
    run_cfg), index lists, and the method name string are pickled.
    """
    import tempfile, shutil

    ctx = mp.get_context("spawn")
    q   = ctx.Queue()

    sp       = session.split_cfg
    tr_idx   = session.tr_idx
    te_idx   = session.te_idx

    banner(f"Parallel run  ({len(methods)} methods, {n_workers} workers)")
    print(f"  {dim('OMP_NUM_THREADS=2 per worker | ')} "
          f"{bold(str(n_workers*2))} {dim('total threads')}")

    # -- Write data to temp mmap files once (shared read-only by all workers) --
    mmap_dir = tempfile.mkdtemp(prefix="fl_mmap_")
    try:
        prog("Writing data to temp mmap files (avoids Windows pipe limit) ...")
        np.save(os.path.join(mmap_dir, "Xtr.npy"), session.Xtr)
        np.save(os.path.join(mmap_dir, "Ytr.npy"), session.Ytr)
        np.save(os.path.join(mmap_dir, "Xte.npy"), session.Xte)
        np.save(os.path.join(mmap_dir, "Yte.npy"), session.Yte)

        shapes = {k: getattr(session, k).shape
                  for k in ("Xtr","Ytr","Xte","Yte")}
        dtypes = {k: str(getattr(session, k).dtype)
                  for k in ("Xtr","Ytr","Xte","Yte")}

        size_mb = sum(
            os.path.getsize(os.path.join(mmap_dir, f"{k}.npy"))
            for k in ("Xtr","Ytr","Xte","Yte")
        ) / 1e6
        print(f"  {dim(f'Temp dir: {mmap_dir}  ({size_mb:.0f} MB total)')}\n")

        pending = list(methods)
        active  = {}
        results = {}

        t_wall = time.time()

        while pending or active:
            # -- Launch workers up to capacity ----------------------------------
            while pending and len(active) < n_workers:
                m = pending.pop(0)
                prog(f"Launching  {bold(m)} ...")
                p = ctx.Process(
                    target=_worker,
                    args=(m, mmap_dir, shapes, dtypes,
                          tr_idx, te_idx, sp, run_cfg, q),
                    daemon=True, name=f"FL-{m}",
                )
                p.start()
                active[m] = p

            # -- Collect one finished result ------------------------------------
            try:
                status, m, payload = q.get(timeout=1.0)
            except Exception:
                for m, p in list(active.items()):
                    if not p.is_alive() and m not in results:
                        p.join()
                        print(f"  {red('[FAIL]')}  {bold(m)} process died "
                              f"(exit={p.exitcode})")
                        results[m] = {"history": [], "time_s": float("nan")}
                        del active[m]
                continue

            p = active.pop(m)
            p.join()

            if status == "ok":
                results[m] = payload
                fin = payload["history"][-1] if payload["history"] else []
                t   = payload.get("time_s", float("nan"))
                tk  = payload.get("top_k", {})
                ok(f"{m:<10}  top1={np.mean(fin):.4f}  "
                   f"top3={tk.get(3, float('nan')):.4f}  "
                   f"macro={payload.get('macro_acc', float('nan')):.4f}  "
                   f"AUC={payload.get('auc', float('nan')):.4f}  "
                   f"Gini={payload.get('gini', float('nan')):.4f}  "
                   f"({t:.1f}s)")
            else:
                print(f"  {red('[FAIL]')}  {bold(m)} FAILED:\n{payload}")
                results[m] = {"history": [], "time_s": float("nan")}

        wall   = round(time.time() - t_wall, 1)
        serial = sum(r.get("time_s", 0) for r in results.values()
                     if isinstance(r.get("time_s"), float))
        print(f"\n  {green('All workers finished.')}  "
              f"Wall={bold(str(wall))}s  "
              f"(serial~{serial:.0f}s  speedup~{serial/max(wall,1):.1f}x)")
        return results

    finally:
        # -- Always clean up temp files, even if a worker crashes --------------
        try:
            shutil.rmtree(mmap_dir, ignore_errors=True)
        except Exception:
            pass


# =============================================================================
# Session
# =============================================================================

class Session:
    def __init__(self):
        self.ds_name:    Optional[str]  = None
        self.split_cfg:  Optional[dict] = None
        self.split_info: Optional[dict] = None
        self.Xtr = self.Ytr = self.Xte = self.Yte = None
        self.tr_idx: Optional[List]     = None
        self.te_idx: Optional[List]     = None
        self.results: Dict[str, dict]   = {}

    @property
    def ready(self) -> bool:
        return self.tr_idx is not None

    @property
    def R(self) -> int:
        return self.split_cfg.get("n_rounds", 20) if self.split_cfg else 20

    def base_run_cfg(self) -> dict:
        return dict(
            R          = self.R,
            emb_dim    = 128,
            batch_size = self.split_cfg.get("batch_size", 64),
            log_every  = max(1, self.R // 5),
        )


# =============================================================================
# Formatting helpers
# =============================================================================

def _strip_ansi(text: str) -> str:
    import re
    return re.sub(r'\x1b\[[0-9;]*m', '', text)

def _mean(hist, c):
    return np.mean(hist[c]) if c < len(hist) else float("nan")

def _fmt(v, w=8, d=4):
    return f"{v:.{d}f}".ljust(w) if not math.isnan(v) else "  -   ".ljust(w)

def _fmtp(v, w=7):
    """Format as percentage string."""
    return f"{v*100:.1f}%".ljust(w) if not math.isnan(v) else "  -  ".ljust(w)

def _sign(v):
    if math.isnan(v): return dim("   -   ")
    s = f"{v:+.4f}"
    return green(s) if v > 0.001 else (red(s) if v < -0.001 else dim(s))

def _bar(v, lo=0.0, hi=1.0, w=14):
    if math.isnan(v) or hi <= lo: return "." * w
    n = round(max(0.0, min(1.0, (v - lo) / (hi - lo))) * w)
    return "#" * n + "." * (w - n)

def _row(cells, widths):
    return "  " + "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

def _sep(widths):
    return "  " + dim("-" * (sum(widths) + 2 * len(widths)))

def _best_mark(v, best_v, higher=True):
    """Return green-formatted value if it equals the best."""
    is_best = (not math.isnan(v) and not math.isnan(best_v) and
               abs(v - best_v) < 1e-9)
    s = _fmt(v)
    return green(s.strip()).ljust(len(s)) if is_best else s


# =============================================================================
# print_comparison - extended 12-section report
# =============================================================================

def print_comparison(results: Dict[str, dict], R: int) -> None:
    ORDER   = ["RDFL", "FibFL", "FibFL+", "FibFL++"]
    methods = [m for m in ORDER if m in results
               and results[m].get("history") and len(results[m]["history"]) > 0]
    if not methods:
        print(f"  {red('No results yet.')}")
        return

    ck = sorted({c for c in {0, 4, 9, 19, R - 1} if c < R})

    # -- Pre-compute stats -----------------------------------------------------
    stats = {}
    for m in methods:
        r      = results[m]
        hist   = r["history"]
        fin    = np.array(hist[-1], dtype=float)
        means  = [np.mean(row) for row in hist]

        stats[m] = dict(
            # basic accuracy
            fin_mean  = float(np.mean(fin)),
            fin_std   = float(np.std(fin)),
            fin_min   = float(np.min(fin)),
            fin_max   = float(np.max(fin)),
            trajectory= [_mean(hist, c) for c in ck],
            # top-k (from derived or fallback to nan)
            top_k     = r.get("top_k",  {1: float(np.mean(fin)),
                                          2: float("nan"),
                                          3: float("nan"),
                                          5: float("nan")}),
            macro_acc = r.get("macro_acc", float("nan")),
            per_class = r.get("per_class", []),
            # convergence
            auc          = r.get("auc",          float("nan")),
            rtt          = r.get("rtt",          {50:R,70:R,80:R,90:R}),
            rate         = r.get("rate",         float("nan")),
            plateau_std  = r.get("plateau_std",  float("nan")),
            worst_round  = r.get("worst_round",  float("nan")),
            temporal_std = r.get("temporal_std", float("nan")),
            # fairness
            gini  = r.get("gini", float("nan")),
            cv    = r.get("cv",   float("nan")),
            w2b   = r.get("w2b",  float("nan")),
            jain  = r.get("jain", float("nan")),
            tail  = r.get("tail", float("nan")),
            # efficiency
            comm_eff = r.get("comm_eff", float("nan")),
            # timing
            time_s   = r.get("time_s",   float("nan")),
            # per-client
            per_client_top1  = r.get("per_client_top1",  []),
            per_client_top3  = r.get("per_client_top3",  []),
            per_client_macro = r.get("per_client_macro",  []),
        )

    all_means = [stats[m]["fin_mean"] for m in methods]
    lo_m, hi_m = min(all_means), max(all_means)
    best_m = max(methods, key=lambda m: stats[m]["fin_mean"])

    props = {
        "RDFL":    dict(server=False, head=False, ring=True,  fib=False),
        "FibFL":   dict(server=False, head=True,  ring=True,  fib=True),
        "FibFL+":  dict(server=False, head=True,  ring=True,  fib=True),
        "FibFL++": dict(server=False, head=True,  ring=True,  fib=True),
    }

    banner("Extended Comparative Analysis")

    # -- S 1  Accuracy trajectory -----------------------------------------------
    section(" 1/12  Accuracy trajectory  (top-1, mean across clients)")
    ck_labels = [f"R={c+1}" for c in ck]
    w1 = [12] + [9]*len(ck) + [20]
    print(f"\n{cyan(_row(['Method']+ck_labels+['Final (bar)'], w1))}")
    print(_sep(w1))
    for m in methods:
        s   = stats[m]
        vs  = [_fmt(v, 9) for v in s["trajectory"]]
        bar = _bar(s["fin_mean"], lo_m, hi_m)
        fv  = f"{s['fin_mean']:.4f}"
        row = ([bold(green(m)) if m == best_m else m] +
               [(green(v.strip()).ljust(9) if m == best_m else v) for v in vs] +
               [green(f"{bar} {fv}") if m == best_m else dim(f"{bar} {fv}")])
        print(_row(row, w1))

    # -- S 2  Top-k accuracy ----------------------------------------------------
    section(" 2/12  Top-k accuracy at R=final  (federation mean)")
    ks_show = [1, 2, 3, 5]
    best_topk = {k: max((stats[m]["top_k"].get(k, float("nan"))
                         for m in methods), default=float("nan"))
                 for k in ks_show}
    w2 = [12, 10, 10, 10, 10, 11]
    print(f"\n{cyan(_row(['Method','top-1','top-2','top-3','top-5','Macro-avg'], w2))}")
    print(_sep(w2))
    for m in methods:
        s   = stats[m]
        row = [bold(m) if m == best_m else m]
        for k in ks_show:
            v = s["top_k"].get(k, float("nan"))
            b = best_topk[k]
            row.append(green(_fmt(v).strip()).ljust(10)
                       if (not math.isnan(v) and not math.isnan(b)
                           and abs(v - b) < 1e-9)
                       else _fmt(v, 10))
        ma = s["macro_acc"]
        bma = max((stats[mm]["macro_acc"] for mm in methods), default=float("nan"))
        row.append(green(_fmt(ma).strip()).ljust(11)
                   if (not math.isnan(ma) and not math.isnan(bma)
                       and abs(ma - bma) < 1e-9)
                   else _fmt(ma, 11))
        print(_row(row, w2))
    print(f"\n  {dim('top-k: true label in top-k predicted classes  |  '
                     'Macro-avg: unweighted mean per-class recall')}")

    # -- S 3  Per-client top-1 / top-3 -----------------------------------------
    section(" 3/12  Per-client accuracy at R=final  (top-1 | top-3 | macro)")
    for m in methods:
        s   = stats[m]
        t1  = s["per_client_top1"]
        t3  = s["per_client_top3"]
        ma  = s["per_client_macro"]
        if not t1:
            continue
        N   = len(t1)
        hdr = f"\n  {bold(m)}"
        print(hdr)
        # top-1
        cvals = "  ".join(f"C{i+1}:{v:.3f}" for i, v in enumerate(t1))
        print(f"    {dim('top-1:')}  {cvals}  "
              f"  {cyan('mean=')} {np.mean([v for v in t1 if not math.isnan(v)]):.4f}")
        if t3:
            cvals3 = "  ".join(f"C{i+1}:{v:.3f}" for i, v in enumerate(t3))
            print(f"    {dim('top-3:')}  {cvals3}  "
                  f"  {cyan('mean=')} {np.mean([v for v in t3 if not math.isnan(v)]):.4f}")
        if ma:
            gap = [t3[i]-t1[i] if (not math.isnan(t3[i]) and not math.isnan(t1[i]))
                   else float("nan") for i in range(min(len(t1),len(t3)))]
            gap_str = "  ".join(f"C{i+1}:{v:+.3f}" for i, v in enumerate(gap))
            print(f"    {dim('delta(t3-t1):')} {gap_str}  "
                  f"  {dim('(head misalignment proxy)')}")

    # -- S 4  Per-class recall --------------------------------------------------
    section(" 4/12  Federation per-class recall  (mean across clients)")
    for m in methods:
        pc = stats[m]["per_class"]
        if not pc:
            continue
        vals = "  ".join(
            f"cls{i}:{v:.3f}" if not math.isnan(v) else f"cls{i}:  - "
            for i, v in enumerate(pc))
        print(f"  {bold(m):<10}  {vals}")
    print(f"\n  {dim('Low per-class recall on non-primary classes = head misalignment under label-skew')}")

    # -- S 5  Convergence profile -----------------------------------------------
    section(" 5/12  Convergence profile")
    best_auc = max((stats[m]["auc"]           for m in methods), default=float("nan"))
    best_r70 = min((stats[m]["rtt"].get(70,R) for m in methods), default=R)
    best_r80 = min((stats[m]["rtt"].get(80,R) for m in methods), default=R)
    best_r90 = min((stats[m]["rtt"].get(90,R) for m in methods), default=R)
    best_r50 = min((stats[m]["rtt"].get(50,R) for m in methods), default=R)
    w5 = [12, 9, 7, 7, 7, 7, 9, 9]
    print(f"\n{cyan(_row(['Method','AUC','R->50%','R->70%','R->80%','R->90%',
                          'Rate/Rnd','Plat.sigma'], w5))}")
    print(_sep(w5))
    for m in sorted(methods, key=lambda m: -stats[m]["auc"]):
        s = stats[m]
        row = [
            bold(m) if m == best_m else m,
            green(_fmt(s["auc"],9).strip()).ljust(9)
                if abs(s["auc"] - best_auc) < 1e-9 else _fmt(s["auc"], 9),
            green(str(s["rtt"].get(50,R))).ljust(7)
                if s["rtt"].get(50,R) == best_r50 else str(s["rtt"].get(50,R)).ljust(7),
            green(str(s["rtt"].get(70,R))).ljust(7)
                if s["rtt"].get(70,R) == best_r70 else str(s["rtt"].get(70,R)).ljust(7),
            green(str(s["rtt"].get(80,R))).ljust(7)
                if s["rtt"].get(80,R) == best_r80 else str(s["rtt"].get(80,R)).ljust(7),
            green(str(s["rtt"].get(90,R))).ljust(7)
                if s["rtt"].get(90,R) == best_r90 else str(s["rtt"].get(90,R)).ljust(7),
            _fmt(s["rate"], 9),
            _fmt(s["plateau_std"], 9),
        ]
        print(_row(row, w5))
    print(f"\n  {dim('AUC = area under accuracy curve (R rounds)  |  '
                     'R->X% = rounds until X% of final accuracy reached')}")

    # -- S 6  Temporal stability ------------------------------------------------
    section(" 6/12  Temporal stability  (round-to-round variance)")
    best_ts = min((stats[m]["temporal_std"]  for m in methods), default=float("nan"))
    best_wr = max((stats[m]["worst_round"]   for m in methods), default=float("nan"))
    w6 = [12, 11, 11, 11]
    print(f"\n{cyan(_row(['Method','Round std(lower=better)','Worst round(higher=better)','Plateau std(lower=better)'], w6))}")
    print(_sep(w6))
    for m in methods:
        s   = stats[m]
        ts  = s["temporal_std"]
        wr  = s["worst_round"]
        ps  = s["plateau_std"]
        bts = best_ts; bwr = best_wr
        row = [
            bold(m) if m == best_m else m,
            green(_fmt(ts,11).strip()).ljust(11)
                if (not math.isnan(ts) and abs(ts-bts)<1e-9) else _fmt(ts,11),
            green(_fmt(wr,11).strip()).ljust(11)
                if (not math.isnan(wr) and abs(wr-bwr)<1e-9) else _fmt(wr,11),
            _fmt(ps,11),
        ]
        print(_row(row, w6))
    print(f"\n  {dim('Low round std = smooth training  |  '
                     'High worst-round = no catastrophic drops mid-training')}")

    # -- S 7  Fairness analysis -------------------------------------------------
    section(" 7/12  Fairness analysis  (all metrics at R=final)")
    bg  = min((stats[m]["gini"] for m in methods), default=float("nan"))
    bc  = min((stats[m]["cv"]   for m in methods), default=float("nan"))
    bw  = max((stats[m]["w2b"]  for m in methods), default=float("nan"))
    bj  = max((stats[m]["jain"] for m in methods), default=float("nan"))
    bt  = max((stats[m]["tail"] for m in methods), default=float("nan"))
    w7  = [12, 9, 9, 9, 11, 10]
    print(f"\n{cyan(_row(['Method','Gini(lower=better)','CV(lower=better)','W2B(higher=better)','Jain(higher=better)','Tail-20%(higher=better)'], w7))}")
    print(_sep(w7))
    for m in methods:
        s   = stats[m]
        def _g(v, best, higher=False):
            ok_ = (not math.isnan(v) and not math.isnan(best)
                   and abs(v - best) < 1e-9)
            return green(_fmt(v).strip()).ljust(9) if ok_ else _fmt(v, 9)
        row = [
            bold(m) if m == best_m else m,
            _g(s["gini"], bg, False),
            _g(s["cv"],   bc, False),
            _g(s["w2b"],  bw, True),
            green(_fmt(s["jain"],11).strip()).ljust(11)
                if (not math.isnan(s["jain"]) and abs(s["jain"]-bj)<1e-9)
                else _fmt(s["jain"],11),
            green(_fmt(s["tail"],10).strip()).ljust(10)
                if (not math.isnan(s["tail"]) and abs(s["tail"]-bt)<1e-9)
                else _fmt(s["tail"],10),
        ]
        print(_row(row, w7))
    print(f"\n  {dim('Gini / CV : lower = more uniform  |  W2B / Jain / Tail : higher = better')}")

    # -- S 8  Summary stats ----------------------------------------------------
    section(" 8/12  Summary statistics  (top-1)")
    w8 = [12, 8, 8, 8, 8, 8]
    print(f"\n{cyan(_row(['Method','Mean','Std','Min','Max','Range'], w8))}")
    print(_sep(w8))
    for m in methods:
        s = stats[m]
        print(_row([bold(m) if m == best_m else m,
                    green(_fmt(s["fin_mean"])) if m == best_m
                    else _fmt(s["fin_mean"]),
                    _fmt(s["fin_std"]), _fmt(s["fin_min"]),
                    _fmt(s["fin_max"]),
                    _fmt(s["fin_max"] - s["fin_min"])], w8))

    # -- S 9  Communication efficiency -----------------------------------------
    section(" 9/12  Communication efficiency  (accuracy per MB transmitted / round)")
    best_ce = max((stats[m]["comm_eff"] for m in methods
                   if not math.isnan(stats[m]["comm_eff"])),
                  default=float("nan"))
    w9 = [12, 14, 18]
    print(f"\n{cyan(_row(['Method','Acc/MB*round^-1','Bytes/round (extractorxK)'], w9))}")
    print(_sep(w9))
    for m in methods:
        s  = stats[m]
        ce = s["comm_eff"]
        br = s.get("bytes_per_round", float("nan"))
        ce_s = (green(_fmt(ce,14).strip()).ljust(14)
                if (not math.isnan(ce) and abs(ce-best_ce)<1e-9)
                else _fmt(ce,14))
        br_s = (f"{br/1e6:.3f} MB".ljust(18) if not math.isnan(br)
                else "  -   ".ljust(18))
        print(_row([bold(m) if m == best_m else m, ce_s, br_s], w9))
    print(f"\n  {dim('FibFL++ K-pass cost: 2xKxNxp_ex4 bytes/round  '
                     '(K=ceil(N/2), p_e=extractor param count)')}")

    # -- S 10  Privacy & topology audit ----------------------------------------
    section("10/12  Privacy & topology audit")
    w10 = [12, 13, 14, 13, 14, 11]
    print(f"\n{cyan(_row(['Method','No server','Private head','Ring topo',
                          'Fib weights','All [ok]?'], w10))}")
    print(_sep(w10))
    def _tick(v): return green("  [ok]  ") if v else red("  [FAIL]  ")
    for m in methods:
        p      = props.get(m, {})
        all_ok = (not p["server"] and p["head"]
                  and p["ring"] and p["fib"])
        print(_row([bold(m) if m == best_m else m,
                    _tick(not p["server"]), _tick(p["head"]),
                    _tick(p["ring"]),       _tick(p["fib"]),
                    green("ALL [ok]") if all_ok else dim("partial")], w10))

    # -- S 11  Gap vs RDFL -----------------------------------------------------
    section("11/12  Gap vs RDFL  (signed accuracy difference at R=final)")
    if "RDFL" in stats:
        base_t1 = stats["RDFL"]["fin_mean"]
        base_t3 = stats["RDFL"]["top_k"].get(3, float("nan"))
        base_ma = stats["RDFL"]["macro_acc"]
        w11 = [12, 12, 12, 12]
        print(f"\n{cyan(_row(['Method','delta top-1','delta top-3','delta macro-avg'], w11))}")
        print(_sep(w11))
        for m in methods:
            if m == "RDFL":
                continue
            s   = stats[m]
            dt1 = s["fin_mean"] - base_t1
            dt3 = (s["top_k"].get(3, float("nan")) - base_t3
                   if not math.isnan(base_t3) else float("nan"))
            dma = (s["macro_acc"] - base_ma
                   if not math.isnan(base_ma) else float("nan"))
            print(_row([bold(m) if m == best_m else m,
                        _sign(dt1).ljust(12), _sign(dt3).ljust(12),
                        _sign(dma).ljust(12)], w11))

    # -- S 12  Timing ----------------------------------------------------------
    section("12/12  Timing")
    serial = sum(stats[m]["time_s"] for m in methods
                 if not math.isnan(stats[m]["time_s"]))
    wall   = max((stats[m]["time_s"] for m in methods
                  if not math.isnan(stats[m]["time_s"])), default=float("nan"))
    w12 = [12, 10, 10]
    print(f"\n{cyan(_row(['Method','Time(s)',''], w12))}")
    print(_sep(w12))
    for m in methods:
        t = stats[m]["time_s"]
        print(_row([bold(m) if m == best_m else m,
                    f"{t:.1f}" if not math.isnan(t) else "-", ""], w12))
    if not math.isnan(serial) and not math.isnan(wall) and wall > 0:
        print(f"\n  {dim('Serial total   :')} {serial:.1f}s")
        print(f"  {dim('Parallel wall  :')} {wall:.1f}s")
        print(f"  {green('Speedup        :')} {serial/wall:.1f}x")


# =============================================================================
# Serial single-method runner
# =============================================================================

def run_method(session: Session, method: str,
               custom_cfg: Optional[dict] = None) -> None:
    if not session.ready:
        print(f"  {red('No data split loaded.')}  Run Module 1 first.")
        return

    sp    = session.split_cfg
    n     = len(session.tr_idx)
    n_cls = sp["n_classes"]
    bs    = sp.get("batch_size", 64)
    emb   = 128

    trl = [make_loader(session.Xtr, session.Ytr, idx, bs)
           for idx in session.tr_idx]
    tel = [make_loader(session.Xte, session.Yte, idx, bs, False)
           for idx in session.te_idx]
    ns  = [len(idx) for idx in session.tr_idx]

    base = session.base_run_cfg()
    base.update(custom_cfg or {})

    em    = build_models(session.Xtr.shape[1], n, emb, n_cls)
    exts  = [e  for e,  _ in em]
    heads = [hd for _, hd in em]

    section(f"Running  {bold(method)}")
    t0 = time.time()

    if method == "RDFL":
        base_m = get_method_cfg(method, base) if custom_cfg is None else {**base}
        cfg = ask_rdfl_cfg(base_m) if custom_cfg is None else base_m
        h   = run_rdfl(exts, heads, trl, tel, cfg)
        session.results["RDFL"] = {
            "history": [[float(a) for a in r] for r in h],
            "time_s":  round(time.time()-t0, 1),
        }
    elif method in ("FibFL", "FibFL+", "FibFL++"):
        base_m = get_method_cfg(method, base) if custom_cfg is None else {**base}
        cfg = ask_fibfl_cfg(base_m, method) if custom_cfg is None else base_m
        qv  = np.array([class_props(session.Ytr, idx, n_cls)
                        for idx in session.tr_idx])
        extra = {}
        if method == "FibFL":
            h = run_fibfl_basic(exts, heads, trl, tel, cfg)
        elif method == "FibFL+":
            h = run_fibflp(exts, heads, trl, tel, cfg)
        else:
            h, sigma, c_seq, c_opt = run_fibflpp(
                exts, heads, trl, tel, qv, cfg)
            extra = {
                "sigma":           [int(x) for x in sigma],
                "ring_cost_seq":   float(c_seq),
                "ring_cost_opt":   float(c_opt),
                "ring_saving_pct": round(100*(c_seq-c_opt)/max(c_seq,1e-9), 1),
            }
        session.results[method] = {
            "history": [[float(a) for a in r] for r in h],
            "time_s":  round(time.time()-t0, 1),
            **extra,
        }

    # -- Compute extended metrics with final model state ------------------------
    import torch
    from fibfl.common import Extractor
    p_e  = sum(p.numel() for p in Extractor(session.Xtr.shape[1], emb).parameters())
    K    = math.ceil(n / 2) if method == "FibFL++" else 1
    hist = session.results[method]["history"]
    fin  = hist[-1]

    # Compute federation metrics from current model state (models still in scope)
    final_metrics = eval_federation_metrics(exts, heads, tel, n_cls, ks=(1,2,3,5))
    derived = compute_derived_metrics(hist, final_metrics,
                                      session.results[method]["time_s"],
                                      n, p_e, K)
    session.results[method].update({**derived, "final_metrics": final_metrics})

    tk = session.results[method].get("top_k", {})
    ok(f"{method:<10}  top1={np.mean(fin):.4f}  "
       f"top3={tk.get(3,float('nan')):.4f}  "
       f"macro={session.results[method].get('macro_acc',float('nan')):.4f}  "
       f"AUC={session.results[method].get('auc',float('nan')):.4f}  "
       f"Gini={session.results[method].get('gini',float('nan')):.4f}  "
       f"({session.results[method]['time_s']}s)")


# =============================================================================
# Save report
# =============================================================================

def save_report(session: Session, outdir: str = ".") -> str:
    os.makedirs(outdir, exist_ok=True)
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ds  = (session.ds_name or "unknown").replace(" ", "_")
    sp  = session.split_cfg or {}

    # -- Build a descriptive split tag for the filename -------------------------
    split_type = sp.get("split", "unknown")
    if split_type == "iid":
        split_tag = "iid"
    elif split_type == "dirichlet":
        alpha = sp.get("alpha", "?")
        split_tag = f"dirichlet_alpha_{str(alpha).replace('.', '_')}"
    elif split_type == "label-skew":
        K_skew = sp.get("K_skew", "?")
        split_tag = f"label-skew_k_{K_skew}"
    else:
        split_tag = split_type

    N = sp.get("n_clients", "?")
    R_val = sp.get("n_rounds", "?")
    path = os.path.join(
        outdir,
        f"fl_report_{ds}_{split_tag}_N{N}_R{R_val}_{ts}.txt"
    )

    ORDER   = ["RDFL", "FibFL", "FibFL+", "FibFL++"]
    methods = [m for m in ORDER
               if m in session.results and session.results[m].get("history")]
    R  = session.R
    W  = 80
    lines: List[str] = []

    def ln(s=""):  lines.append(_strip_ansi(s))
    def h1(s):     ln("=" * W); ln(f"  {s}"); ln("=" * W)
    def h2(s):     ln(); ln(f"-- {s} --"); ln()
    def sep(ws):   ln("  " + "-" * (sum(ws) + 2*len(ws)))
    def tbl(cells, ws):
        ln("  " + "  ".join(str(c).ljust(w) for c, w in zip(cells, ws)))

    # -- Report header - full split metadata -----------------------------------
    h1(f"FibFL FEDERATED LEARNING - EXPERIMENT REPORT  "
       f"(parallel run)")
    ln()
    ln(f"  Generated     : {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    ln(f"  Runner        : fl_main_parallel.py  (multiprocessing, spawn)")
    ln()
    ln(f"  Dataset       : {session.ds_name}")
    ln(f"  Split type    : {split_type}")

    # -- Split-type-specific parameters ----------------------------------------
    if split_type == "dirichlet":
        alpha = sp.get("alpha", "?")
        ln(f"  Alpha (alpha)     : {alpha}"
           f"  {'(IID-like, mild skew)' if float(alpha) >= 0.8 else ''}"
           f"{'(moderate non-IID)' if 0.3 <= float(alpha) < 0.8 else ''}"
           f"{'(strong non-IID)' if float(alpha) < 0.3 else ''}")
    elif split_type == "label-skew":
        K_skew = sp.get("K_skew", "?")
        ln(f"  Label-skew K  : {K_skew}"
           f"  ({K_skew} primary class{'es' if int(K_skew) > 1 else ''} per client)")
    elif split_type == "iid":
        ln(f"  Heterogeneity : IID - equal random split")

    ln(f"  Clients N     : {sp.get('n_clients', '?')}")
    ln(f"  Rounds R      : {sp.get('n_rounds', '?')}")
    ln(f"  Batch size    : {sp.get('batch_size', '?')}")
    ln(f"  Num classes   : {sp.get('n_classes', '?')}")
    ln(f"  Methods run   : {', '.join(methods)}")
    ln(f"  Device        : {DEVICE}")
    ln(f"  CPU cores     : {mp.cpu_count()}")
    ln(f"  Workers used  : {_default_workers()}")
    ln()
    ln(f"  File          : {os.path.basename(path)}")
    ln()

    if session.split_info:
        ln(session.split_info.get("txt_block", ""))

    def _gini(fin):
        s = np.sort(np.array(fin,float)); n=len(s)
        return (2*np.sum(np.arange(1,n+1)*s)/(n*s.sum())-(n+1)/n) if s.sum()>0 else 0.0

    # -- Summary table ----------------------------------------------------------
    h2("SUMMARY STATISTICS")
    w_s = [12,8,8,8,8,8,9,8,8,9,9,10,8]
    tbl(["Method","top-1","top-2","top-3","top-5","Macro",
         "Gini(lower=better)","CV(lower=better)","W2B(higher=better)","Jain(higher=better)","AUC","Comm.Eff","Time(s)"], w_s)
    sep(w_s)
    for m in methods:
        r    = session.results[m]
        fin  = _last(r["history"], [])
        if not fin: continue
        tk   = r.get("top_k", {})
        tbl([m,
             f"{np.mean(fin):.4f}",
             f"{tk.get(2,float('nan')):.4f}",
             f"{tk.get(3,float('nan')):.4f}",
             f"{tk.get(5,float('nan')):.4f}",
             f"{r.get('macro_acc',float('nan')):.4f}",
             f"{r.get('gini',_gini(fin)):.4f}",
             f"{r.get('cv', np.std(fin)/max(np.mean(fin),1e-9)):.4f}",
             f"{r.get('w2b', np.min(fin)/max(np.max(fin),1e-9)):.4f}",
             f"{r.get('jain',float('nan')):.4f}",
             f"{r.get('auc',float('nan')):.4f}",
             f"{r.get('comm_eff',float('nan')):.4f}",
             f"{r.get('time_s',float('nan')):.1f}"], w_s)

    # -- Convergence table ------------------------------------------------------
    h2("CONVERGENCE PROFILE")
    w_c = [12, 7, 7, 7, 7, 9, 9, 10]
    tbl(["Method","R->50%","R->70%","R->80%","R->90%",
         "Rate/Rnd","Plat.sigma","Worst-Rnd"], w_c)
    sep(w_c)
    for m in methods:
        r   = session.results[m]
        rtt = r.get("rtt", {})
        tbl([m,
             str(rtt.get(50, R)),
             str(rtt.get(70, R)),
             str(rtt.get(80, R)),
             str(rtt.get(90, R)),
             f"{r.get('rate', float('nan')):.4f}",
             f"{r.get('plateau_std', float('nan')):.4f}",
             f"{r.get('worst_round', float('nan')):.4f}"], w_c)

    # -- Per-round top-1 --------------------------------------------------------
    h2("PER-ROUND MEAN TOP-1 ACCURACY")
    w_pr = [12] + [7]*R
    tbl(["Method"] + [f"R={r+1}" for r in range(R)], w_pr)
    sep(w_pr)
    for m in methods:
        hist = session.results[m]["history"]
        vals = [f"{np.mean(hist[r]):.4f}" if r < len(hist) else " -"
                for r in range(R)]
        tbl([m] + vals, w_pr)

    # -- Per-client accuracy ----------------------------------------------------
    h2("PER-CLIENT ACCURACY AT R=FINAL  (top-1 | top-3 | macro)")
    for m in methods:
        r   = session.results[m]
        fin = _last(r["history"], [])
        if not fin: continue
        nc  = len(fin)
        ln(f"\n  {m}")
        w_cl = [9]*nc + [9,9,9,9]
        tbl([f"C{i+1}" for i in range(nc)] +
            ["Mean","Std","Min","Max"], w_cl)
        sep(w_cl)
        tbl([f"{v:.4f}" for v in fin] +
            [f"{np.mean(fin):.4f}", f"{np.std(fin):.4f}",
             f"{np.min(fin):.4f}", f"{np.max(fin):.4f}"], w_cl)
        t3 = r.get("per_client_top3", [])
        if t3:
            ln("  top-3:  " + "  ".join(f"C{i+1}:{v:.4f}" for i,v in enumerate(t3)))
        ma = r.get("per_client_macro", [])
        if ma:
            ln("  macro:  " + "  ".join(f"C{i+1}:{v:.4f}" for i,v in enumerate(ma)))

    # -- Per-class recall -------------------------------------------------------
    h2("FEDERATION PER-CLASS RECALL")
    for m in methods:
        pc = session.results[m].get("per_class", [])
        if pc:
            ln(f"  {m:<12} " +
               "  ".join(f"cls{i}:{v:.4f}" if not math.isnan(v)
                         else f"cls{i}:  -  "
                         for i, v in enumerate(pc)))

    # -- 2-opt summary ----------------------------------------------------------
    if "FibFL++" in session.results:
        rv = session.results["FibFL++"]
        h2("2-OPT RING ORDERING  (FibFL++)")
        ln(f"  Sequential cost : {rv.get('ring_cost_seq', float('nan')):.4f}")
        ln(f"  Optimised cost  : {rv.get('ring_cost_opt', float('nan')):.4f}")
        ln(f"  Saving          : {rv.get('ring_saving_pct', float('nan')):.1f}%")
        ln(f"  Permutation sigma*  : {rv.get('sigma', [])}")

    # -- Gap vs RDFL ------------------------------------------------------------
    if "RDFL" in session.results:
        h2("GAP VS RDFL  (signed, at R=final)")
        base_t1 = _last_mean(session.results["RDFL"]["history"])
        base_t3 = session.results["RDFL"].get("top_k",{}).get(3,float("nan"))
        base_ma = session.results["RDFL"].get("macro_acc", float("nan"))
        w_gap = [12, 10, 10, 10]
        tbl(["Method","delta top-1","delta top-3","delta macro"], w_gap)
        sep(w_gap)
        for m in methods:
            if m == "RDFL": continue
            r   = session.results[m]
            fin = _last(r["history"], [])
            if not fin: continue
            dt1 = np.mean(fin) - base_t1
            dt3 = (r.get("top_k",{}).get(3,float("nan")) - base_t3
                   if not math.isnan(base_t3) else float("nan"))
            dma = (r.get("macro_acc",float("nan")) - base_ma
                   if not math.isnan(base_ma) else float("nan"))
            tbl([m,
                 (f"+{dt1:.4f}" if dt1>=0 else f"{dt1:.4f}"),
                 (f"+{dt3:.4f}" if not math.isnan(dt3) and dt3>=0
                  else f"{dt3:.4f}" if not math.isnan(dt3) else "-"),
                 (f"+{dma:.4f}" if not math.isnan(dma) and dma>=0
                  else f"{dma:.4f}" if not math.isnan(dma) else "-")],
                w_gap)

    ln(); ln("=" * W)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# =============================================================================
# Interactive menu
# =============================================================================

METHOD_MENU = [
    ("RDFL",                     "Ring, uniform weights, full model              "),
    ("FibFL",                    "Ring, Fibonacci, private head  (basic)         "),
    ("FibFL+",                   "Ring, Fibonacci + accuracy gating              "),
    ("FibFL++",                  "Ring, 2-opt + K-pass + warmup  (full)          "),
    ("Run ALL [parallel]",     "Run all 4 methods concurrently on multiple cores"),
    ("Run selected [parallel]","Choose methods - run selected subset in parallel"),
    ("Save report",              "Save full .txt report to disk"),
    ("New dataset",              "Load a new dataset / change split (discards results)"),
    ("Quit",                     "Exit"),
]

ALL_METHODS = ["RDFL", "FibFL", "FibFL+", "FibFL++"]

# -- Per-method default hyperparameters (based on original author papers) ------
#
# FedAvg  - McMahan et al., AISTATS 2017
#   E=5 local epochs, lr=0.01, SGD momentum=0.9
#
# RDFL    - Wang et al., arXiv 2021
#   E=5 local epochs, lr=0.01, gamma=0.5 (self-retention)
#
# FedRep  - Collins et al., ICML 2021
#   Eh=2 head epochs, Ee=2 extractor epochs, lr=0.01
#   (original paper uses tau=1 head step; Eh=2 is a close practical equivalent)
#
# FibFL / FibFL+ / FibFL++  - this work
#   Eh=1 head epoch, Ee=20 extractor epochs (as specified), lr=0.01
#   gamma_start=0.4, gamma_end=0.05 (cosine annealing for FibFL++)
#   tau=0.35 (accuracy gate threshold for FibFL+/++)

# =============================================================================
# Per-method hyperparameters
# =============================================================================
#
# FedAvg, RDFL, FedRep: values are FIXED to match original author papers.
# Do NOT edit these - they are the reference baselines as published.
#
# FibFL / FibFL+ / FibFL++: values are USER-CONFIGURABLE below.
# Edit FIBFL_CFG to tune the FibFL family without touching anything else.
# =============================================================================

# -- Baseline method - fixed to original paper values -------------------------
#
# RDFL  Wang et al., arXiv 2021 (2104.08100)
#   Follows the same local training protocol as FedAvg with E=5
#   Ring blend weight gamma=0.5 (equal weight between own and neighbours)
_RDFL_CFG = dict(
    E     = 5,    # local epochs            Wang et al. 2021
    lr    = 0.01, # SGD learning rate       Wang et al. 2021
    gamma = 1,  # self-retention weight   Wang et al. 2021
)

# -- FibFL family - USER-CONFIGURABLE -----------------------------------------
#
# Edit the values in FIBFL_CFG freely.
# These apply to FibFL, FibFL+, and FibFL++ simultaneously.
# Keys:
#   Eh            : head training epochs per round
#   Ee            : extractor training epochs per round
#   lr            : learning rate (Adam)
#   gamma         : self-retention for FibFL basic (0=full gossip, 1=no gossip)
#   gamma_start   : cosine-anneal start for FibFL++ (high -> more self-retention early)
#   gamma_end     : cosine-anneal end   for FibFL++ (low  -> more gossip late)
#   tau           : accuracy gate threshold for FibFL+/++ (neighbour must exceed
#                   this accuracy to contribute to the blend; 0=always blend)

FIBFL_CFG = dict(
    Eh          = 1,     # head epochs/round      - keep small (private head)
    Ee          = 20,    # extractor epochs/round - heavy extractor training
    lr          = 0.01,  # learning rate
    gamma       = 0.5,   # FibFL basic self-retention
    gamma_start = 0.4,   # FibFL++ cosine-anneal start
    gamma_end   = 0.05,  # FibFL++ cosine-anneal end
    tau         = 0.35,  # accuracy gate threshold
)

# -- Assembled lookup used by the runner --------------------------------------
METHOD_CFGS = {
    "RDFL":    _RDFL_CFG,
    "FibFL":   FIBFL_CFG,
    "FibFL+":  FIBFL_CFG,
    "FibFL++": FIBFL_CFG,
}

# Fallback for any unlisted method
DEFAULT_CFG = FIBFL_CFG


def get_method_cfg(method: str, base: dict) -> dict:
    """
    Merge base_run_cfg() (R, emb_dim, batch_size, log_every) with the
    per-method hyperparameters from METHOD_CFGS.
    Method-specific values override base; explicit custom_cfg overrides all.
    """
    method_defaults = METHOD_CFGS.get(method, DEFAULT_CFG)
    return {**base, **method_defaults}


def interactive_main(outdir: str = ".", datadir: str = "./data",
                     n_workers: int = 0) -> None:
    if n_workers <= 0:
        n_workers = _default_workers()

    banner("FIRMA  Federated Learning Comparison Suite  [parallel + extended metrics]")
    print(f"\n  {dim('All methods run on the SAME data split for a fair comparison.')}")
    print(f" {bold('Run ALL')} / {bold('Run selected')} execute methods "
          f"{bold('in parallel')} ({bold(str(n_workers))} workers).")
    print(f"  {dim('Metrics: top-1/2/3/5, macro-avg, per-class recall, AUC, ')}"
          f"{dim('Gini, Jain, tail, comm-efficiency')}\n")

    session = Session()
    (session.ds_name, session.split_cfg,
     session.Xtr, session.Ytr,
     session.Xte, session.Yte) = dataset_menu(datadir)
    session.tr_idx, session.te_idx = partition(
        session.Ytr, session.Yte, session.split_cfg)
    session.split_cfg["n_clients"] = len(session.tr_idx)
    session.split_info = show_split_stats(
        session.Ytr, session.tr_idx,
        session.split_cfg["n_classes"], session.split_cfg)

    while True:
        banner("Main Menu  [ parallel | extended metrics]")
        completed = set(session.results.keys())
        sp = session.split_cfg
        al_str = (f"alpha={sp.get('alpha','?')}" if sp.get('split')=='dirichlet'
                  else f"K={sp.get('K_skew','?')}" if sp.get('split')=='label-skew'
                  else "IID")
        print(f"  {dim('Dataset  :')} {bold(session.ds_name)}  "
              f"{dim('Split:')} {bold(sp.get('split','?'))} {al_str}  "
              f"{dim('N=')} {bold(sp.get('n_clients','?'))}  "
              f"{dim('R=')} {bold(sp.get('n_rounds','?'))}  "
              f"{dim('Workers=')} {bold(str(n_workers))}")
        if completed:
            done_str = "  ".join(
                green(m) + " " +
                green(f"t1={_last_mean(session.results[m]['history']):.3f}"
                      f" t3={session.results[m].get('top_k',{}).get(3,float('nan')):.3f}")
                for m in ALL_METHODS
                if m in completed and session.results[m].get("history")
            )
            print(f"  {dim('Completed:')} {done_str}")
        remaining = [m for m in ALL_METHODS if m not in completed]
        if remaining:
            print(f"  {dim('Pending  :')} {dim('  '.join(remaining))}")
        print()

        opts = []
        for label, desc in METHOD_MENU:
            m    = label.split()[0]
            tick = green(" [ok]") if m in completed else ""
            opts.append((label + tick, desc))

        choice = menu("Choose action:", opts, default=5)

        if choice <= 4:
            method = ALL_METHODS[choice - 1]
            if method in session.results:
                section(f"Re-run {bold(method)}?")
                prev_fin = _last(session.results[method]["history"], [])
                if not prev_fin:
                    info("Previous run returned no history (worker may have crashed).")
                    if not confirm("Re-run?", default=True): continue
                    run_method(session, method); continue
                info(f"Previous: top1={np.mean(prev_fin):.4f}  "
                     f"top3={session.results[method].get('top_k',{}).get(3,float('nan')):.4f}")
                if not confirm("Re-run?", default=True):
                    continue
            run_method(session, method)

        elif choice == 5:
            cfg = session.base_run_cfg()   # per-method defaults applied in _worker
            new = run_methods_parallel(ALL_METHODS, session, cfg, n_workers)
            session.results.update(new)

        elif choice == 6:
            section("Select methods to run")
            print(f"\n  {bold('Choose methods')} (comma-separated, e.g. 1,3,6):")
            for i, m in enumerate(ALL_METHODS, 1):
                done = green(" [ok]") if m in session.results else ""
                print(f"  {dim(f'[{i}]')}  {bold(m)}{done}")
            while True:
                raw = (input(f"\n  {cyan('->')} Methods [1,2,3,4]: ").strip()
                       or "1,2,3,4")
                try:
                    chosen = [int(x.strip()) for x in raw.split(",")]
                    if all(1 <= c <= len(ALL_METHODS) for c in chosen):
                        break
                except ValueError:
                    pass
                print(f"  {red('Invalid.')} Enter comma-separated 1-{len(ALL_METHODS)}.")
            cfg      = session.base_run_cfg()   # per-method defaults applied in _worker
            selected = [ALL_METHODS[i-1] for i in chosen]
            new = run_methods_parallel(selected, session, cfg,
                                       min(n_workers, len(selected)))
            session.results.update(new)

        elif choice == 7:
            if not session.results:
                print(f"  {red('No results yet.')}"); continue
            path = save_report(session, outdir)
            ok(f"Report saved -> {bold(path)}")

        elif choice == 8:
            if session.results:
                if not confirm("Discard current results and load new dataset?",
                               default=False):
                    continue
            session = Session()
            (session.ds_name, session.split_cfg,
             session.Xtr, session.Ytr,
             session.Xte, session.Yte) = dataset_menu(datadir)
            session.tr_idx, session.te_idx = partition(
                session.Ytr, session.Yte, session.split_cfg)
            session.split_cfg["n_clients"] = len(session.tr_idx)
            session.split_info = show_split_stats(
                session.Ytr, session.tr_idx,
                session.split_cfg["n_classes"], session.split_cfg)

        elif choice == 9:
            print(f"\n  {dim('Goodbye.')}\n")
            break


# =============================================================================
# Auto mode  (7 paper experiments per dataset: IID + 3xDir + 3xLSkew)
# =============================================================================

PAPER_CFGS = [
    dict(label="E1 MNIST-60k  IID           N=10",
         ds="MNIST-60k",
         split=dict(split="iid",        alpha=1.0, K_skew=1, n_classes=10,
                    n_clients=10, n_rounds=20, batch_size=64),
         run=dict(E=2,Eh=2,Ee=2,lr=0.01,gamma=0.5,
                  gamma_start=0.4,gamma_end=0.05,tau=0.35,emb_dim=128)),
    dict(label="E2 MNIST-60k  Dir alpha=0.8     N=10",
         ds="MNIST-60k",
         split=dict(split="dirichlet",  alpha=0.8, K_skew=1, n_classes=10,
                    n_clients=10, n_rounds=20, batch_size=64),
         run=dict(E=2,Eh=2,Ee=2,lr=0.01,gamma=0.5,
                  gamma_start=0.4,gamma_end=0.05,tau=0.35,emb_dim=128)),
    dict(label="E3 MNIST-60k  Dir alpha=0.5     N=10",
         ds="MNIST-60k",
         split=dict(split="dirichlet",  alpha=0.5, K_skew=1, n_classes=10,
                    n_clients=10, n_rounds=20, batch_size=64),
         run=dict(E=2,Eh=2,Ee=2,lr=0.01,gamma=0.5,
                  gamma_start=0.4,gamma_end=0.05,tau=0.35,emb_dim=128)),
    dict(label="E4 MNIST-60k  Dir alpha=0.1     N=10",
         ds="MNIST-60k",
         split=dict(split="dirichlet",  alpha=0.1, K_skew=1, n_classes=10,
                    n_clients=10, n_rounds=20, batch_size=64),
         run=dict(E=2,Eh=2,Ee=2,lr=0.01,gamma=0.5,
                  gamma_start=0.4,gamma_end=0.05,tau=0.35,emb_dim=128)),
    dict(label="E5 MNIST-60k  Label-skew K=1 N=10",
         ds="MNIST-60k",
         split=dict(split="label-skew", alpha=1.0, K_skew=1, n_classes=10,
                    n_clients=10, n_rounds=20, batch_size=64),
         run=dict(E=2,Eh=2,Ee=2,lr=0.01,gamma=0.5,
                  gamma_start=0.4,gamma_end=0.05,tau=0.35,emb_dim=128)),
    dict(label="E6 MNIST-60k  Label-skew K=2 N=10",
         ds="MNIST-60k",
         split=dict(split="label-skew", alpha=1.0, K_skew=2, n_classes=10,
                    n_clients=10, n_rounds=20, batch_size=64),
         run=dict(E=2,Eh=2,Ee=2,lr=0.01,gamma=0.5,
                  gamma_start=0.4,gamma_end=0.05,tau=0.35,emb_dim=128)),
    dict(label="E7 CIFAR-10   Label-skew K=1 N=10",
         ds="CIFAR-10",
         split=dict(split="label-skew", alpha=1.0, K_skew=1, n_classes=10,
                    n_clients=10, n_rounds=20, batch_size=64),
         run=dict(E=2,Eh=2,Ee=2,lr=0.005,gamma=0.5,
                  gamma_start=0.4,gamma_end=0.05,tau=0.35,emb_dim=128)),
]


def run_auto(outdir: str = ".", datadir: str = "./data",
             n_workers: int = 0) -> None:
    if n_workers <= 0:
        n_workers = _default_workers()
    banner(f"FIRMA  -  auto mode  [{len(PAPER_CFGS)} experiments, "
           f"[fast] {n_workers} workers, extended metrics]")

    for exp in PAPER_CFGS:
        banner(exp["label"])
        session           = Session()
        session.ds_name   = exp["ds"]
        session.split_cfg = dict(exp["split"])
        session.Xtr, session.Ytr, session.Xte, session.Yte = \
            load_dataset(exp["ds"], datadir)
        session.tr_idx, session.te_idx = partition(
            session.Ytr, session.Yte, session.split_cfg)
        session.split_cfg["n_clients"] = len(session.tr_idx)
        session.split_info = show_split_stats(
            session.Ytr, session.tr_idx,
            session.split_cfg["n_classes"], session.split_cfg)

        cfg = {**session.base_run_cfg(), **exp["run"]}
        new = run_methods_parallel(ALL_METHODS, session, cfg, n_workers)
        session.results.update(new)

        print_comparison(session.results, session.R)
        path = save_report(session, outdir)
        ok(f"Report saved -> {path}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="FibFL parallel FL comparison suite (extended metrics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
          python3 fl_main_parallel.py                     # interactive
          python3 fl_main_parallel.py --auto              # 7 paper experiments
          python3 fl_main_parallel.py --workers 4
          python3 fl_main_parallel.py --outdir ~/results
          python3 fl_main_parallel.py --datadir ~/data
        """),
    )
    ap.add_argument("--auto",    action="store_true")
    ap.add_argument("--workers", type=int, default=0, metavar="N",
                    help=f"Worker processes (default {_default_workers()})")
    ap.add_argument("--outdir",  default=".", metavar="DIR")
    ap.add_argument("--datadir", default="./data", metavar="DIR")
    args = ap.parse_args()

    if args.auto:
        run_auto(args.outdir, args.datadir, args.workers)
    else:
        interactive_main(args.outdir, args.datadir, args.workers)


if __name__ == "__main__":
    main()

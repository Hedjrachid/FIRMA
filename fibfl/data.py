"""
fl_data.py  -  Module 1
========================
Dataset loading and data partitioning for federated learning.

Public API
----------
    load_dataset(name, data_root)           -> Xtr, Ytr, Xte, Yte
    partition(Ytr, Yte, cfg)                -> tr_idx, te_idx, split_info
    make_loader(X, Y, idx, bs, shuffle)     -> DataLoader
    class_props(Y, idx, n_classes)          -> np.ndarray
    show_split_stats(Ytr, tr_idx, ...)      -> split_info dict

Supported datasets
------------------
    MNIST-1797    sklearn digits (offline, no download needed)
    MNIST-60k     full MNIST via torchvision
    Fashion-MNIST via torchvision
    CIFAR-10      via torchvision
    CIFAR-100     via torchvision

Supported splits
----------------
    IID           equal-size random partition
    Dirichlet     Dir(alpha * 1_N) class-proportion sampling
    Label-skew    K primary + K+1 secondary + 3 % minority classes
"""

import os
import sys
import math
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fibfl.common import (
    banner, section, menu, ask_int, ask_float, confirm,
    ok, info, prog,
    bold, cyan, green, yellow, red, dim,
)

warnings.filterwarnings("ignore")


# =============================================================================
# Dataset registry
# =============================================================================

DATASETS: Dict[str, dict] = {
    "MNIST-1797":    dict(n_classes=10,  input_dim=64,
                          desc="Sklearn digits (1 797 samples, offline)"),
    "MNIST-60k":     dict(n_classes=10,  input_dim=784,
                          desc="Full MNIST (60 000 samples, download)"),
    "Fashion-MNIST": dict(n_classes=10,  input_dim=784,
                          desc="Fashion-MNIST (60 000 samples, download)"),
    "CIFAR-10":      dict(n_classes=10,  input_dim=3072,
                          desc="CIFAR-10 (50 000 samples, download)"),
    "CIFAR-100":     dict(n_classes=100, input_dim=3072,
                          desc="CIFAR-100 (50 000 samples, download)"),
}
DS_NAMES = list(DATASETS.keys())
DS_MENU  = [(name, DATASETS[name]["desc"]) for name in DS_NAMES]


# =============================================================================
# Dataset loading
# =============================================================================

def load_dataset(name: str,
                 data_root: str = "./data") -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    """
    Load a benchmark dataset by name.

    Returns
    -------
    Xtr, Ytr  training split  (X: float32 in [0,1], Y: int64)
    Xte, Yte  test split
    X is flattened to shape (N, D).

    MNIST-1797 uses sklearn (no network).
    All others download via torchvision and cache in data_root.
    """
    os.makedirs(data_root, exist_ok=True)

    if name == "MNIST-1797":
        return _load_sklearn_digits()

    prog(f"Loading {name} ...")
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        print(f"\n  {red('torchvision not installed.')}  Run: pip install torchvision")
        sys.exit(1)

    tf_gray  = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    tf_color = T.Compose([T.ToTensor(),
                           T.Normalize((0.4914, 0.4822, 0.4465),
                                       (0.2023, 0.1994, 0.2010))])
    cls_map = {
        "MNIST-60k":     (torchvision.datasets.MNIST,        tf_gray,  tf_gray),
        "Fashion-MNIST": (torchvision.datasets.FashionMNIST, tf_gray,  tf_gray),
        "CIFAR-10":      (torchvision.datasets.CIFAR10,      tf_color, tf_color),
        "CIFAR-100":     (torchvision.datasets.CIFAR100,     tf_color, tf_color),
    }
    cls, tf_tr, tf_te = cls_map[name]
    try:
        tr = cls(data_root, train=True,  download=True, transform=tf_tr)
        te = cls(data_root, train=False, download=True, transform=tf_te)
    except Exception as exc:
        print(f"\n  {red('Download failed:')} {exc}")
        sys.exit(1)

    def _to_np(ds):
        ld = DataLoader(ds, batch_size=2048, shuffle=False)
        Xs, Ys = [], []
        for x, y in ld:
            Xs.append(x.view(x.size(0), -1).numpy())
            Ys.append(y.numpy() if isinstance(y, torch.Tensor) else np.array(y))
        return (np.concatenate(Xs).astype(np.float32),
                np.concatenate(Ys).astype(np.int64))

    Xtr, Ytr = _to_np(tr)
    Xte, Yte = _to_np(te)
    ok(f"{name}  train={Xtr.shape}  test={Xte.shape}")
    return Xtr, Ytr, Xte, Yte


def _load_sklearn_digits() -> Tuple[np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits
    raw  = load_digits()
    X    = raw.data.astype(np.float32) / 16.0
    Y    = raw.target.astype(np.int64)
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    nt   = int(0.8 * len(X))
    ok(f"MNIST-1797: {len(X)} samples (80/20 split, no download)")
    return X[perm[:nt]], Y[perm[:nt]], X[perm[nt:]], Y[perm[nt:]]


# =============================================================================
# Data partitioning
# =============================================================================

def iid_split(Y: np.ndarray, n: int, seed: int = 42) -> List[np.ndarray]:
    """Equal-size random IID partition.  Returns N index arrays."""
    rng = np.random.default_rng(seed)
    return list(np.array_split(rng.permutation(len(Y)), n))


def dirichlet_split(Y: np.ndarray, n: int, alpha: float,
                    n_classes: int, seed: int = 42) -> List[np.ndarray]:
    """
    Dirichlet(alpha) non-IID partition.

    For each class c, proportions p ~ Dir(alpha*1_N) determine how many
    samples of class c each client receives.
    Lower alpha -> stronger label skew.  alpha -> inf approaches IID.
    """
    rng    = np.random.default_rng(seed)
    by_cls = [np.where(Y == c)[0] for c in range(n_classes)]
    clients: List[List[int]] = [[] for _ in range(n)]
    for c in range(n_classes):
        idx    = rng.permutation(by_cls[c])
        props  = rng.dirichlet(np.ones(n) * alpha)
        splits = (props * len(idx)).astype(int)
        splits[-1] = len(idx) - splits[:-1].sum()   # fix rounding
        s = 0
        for i, size in enumerate(splits):
            clients[i].extend(idx[s:s+size].tolist())
            s += size
    return [np.array(c) for c in clients]


def label_skew_split(Y: np.ndarray, n: int, K: int,
                     n_classes: int, seed: int = 42) -> List[np.ndarray]:
    """
    Label-skew(K) structured non-IID partition.

    Each client receives:
        K   primary   classes -> 55 % of its samples
        K+1 secondary classes -> 20 % of its samples
        all remaining classes ->  3 % minority share
    Primary classes are assigned round-robin for balanced coverage.
    """
    rng    = np.random.default_rng(seed)
    by_cls = [list(np.where(Y == c)[0]) for c in range(n_classes)]
    for lst in by_cls:
        rng.shuffle(lst)

    cyc  = (list(range(n_classes)) * (n * (K + K + 1) // n_classes + 2))
    p_of = {i: cyc[i*K            : i*K + K]         for i in range(n)}
    s_of = {i: cyc[i*K + K        : i*K + K + (K+1)] for i in range(n)}

    clients: List[List[int]] = [[] for _ in range(n)]

    def _take(pool, frac, budget):
        want = max(1, int(frac * budget))
        out  = []
        for c in pool:
            batch = by_cls[c][:want]
            by_cls[c] = by_cls[c][want:]
            out.extend(batch)
        return out

    for i in range(n):
        budget = max(20, len(Y) // n)
        clients[i] += _take(p_of[i], 0.55, budget)
        clients[i] += _take(s_of[i], 0.20, budget)
        minority = [c for c in range(n_classes)
                    if c not in p_of[i] and c not in s_of[i]]
        clients[i] += _take(minority, 0.03, budget)

    return [np.array(c) for c in clients]


def class_props(Y: np.ndarray, idx: np.ndarray,
                n_classes: int) -> np.ndarray:
    """Normalised class-proportion vector q_i (used by 2-opt ordering)."""
    counts = np.bincount(Y[idx], minlength=n_classes).astype(float)
    s = counts.sum()
    return counts / s if s > 0 else counts


def make_loader(X: np.ndarray, Y: np.ndarray, idx: np.ndarray,
                bs: int = 64, shuffle: bool = True) -> DataLoader:
    """Wrap an index array into a PyTorch DataLoader."""
    xs = torch.tensor(X[idx], dtype=torch.float32)
    ys = torch.tensor(Y[idx], dtype=torch.long)
    return DataLoader(TensorDataset(xs, ys), batch_size=bs, shuffle=shuffle)


# =============================================================================
# Partition orchestrator
# =============================================================================

def partition(Ytr: np.ndarray, Yte: np.ndarray,
              cfg: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply the partitioning strategy from cfg to Ytr and Yte.

    cfg keys  : split ('iid'|'dirichlet'|'label-skew'),
                n_clients, alpha, K_skew, n_classes, batch_size
    Returns   : tr_idx, te_idx  (lists of index arrays, one per client)
                Empty clients (< batch_size samples) are dropped.
    """
    n         = cfg["n_clients"]
    n_classes = cfg["n_classes"]
    bs        = cfg.get("batch_size", 64)
    split     = cfg["split"]

    if split == "iid":
        tr_idx = iid_split(Ytr, n)
        te_idx = iid_split(Yte, n)
    elif split == "dirichlet":
        tr_idx = dirichlet_split(Ytr, n, cfg["alpha"], n_classes)
        te_idx = dirichlet_split(Yte, n, cfg["alpha"], n_classes)
    else:   # label-skew
        tr_idx = label_skew_split(Ytr, n, cfg["K_skew"], n_classes)
        te_idx = label_skew_split(Yte, n, cfg["K_skew"], n_classes)

    # Drop clients with too few samples
    tr_idx = [idx for idx in tr_idx if len(idx) >= bs]
    te_idx = te_idx[:len(tr_idx)]
    n_act  = len(tr_idx)
    if n_act < n:
        info(f"Dropped {n - n_act} empty clients - actual N = {n_act}")

    return tr_idx, te_idx


# =============================================================================
# Split statistics display (shown before running any method)
# =============================================================================

def show_split_stats(Ytr: np.ndarray,
                     tr_idx: List[np.ndarray],
                     n_classes: int,
                     split_cfg: dict) -> dict:
    """
    Print a rich per-client class-distribution table and return a
    plain-text block for inclusion in the .txt report.

    Columns  : Client | Samples | Share% | #Classes | Gini | Top-3
    Bar chart: one character per class, height-coded by fraction.
    Federation summary: totals, global Gini, class coverage.
    """
    n             = len(tr_idx)
    total_samples = sum(len(idx) for idx in tr_idx)
    W             = 72

    # Per-client statistics
    client_stats = []
    for i, idx in enumerate(tr_idx):
        counts    = np.bincount(Ytr[idx], minlength=n_classes)
        props_v   = counts.astype(float) / max(counts.sum(), 1)
        top3      = np.argsort(counts)[-3:][::-1]
        n_nonzero = int((counts > 0).sum())
        s         = np.sort(counts.astype(float))
        gini_c    = (2*np.sum(np.arange(1, n_classes+1)*s)
                     / (n_classes * s.sum()) - (n_classes+1)/n_classes
                     ) if s.sum() > 0 else 0.0
        client_stats.append(dict(
            n_samples=len(idx),
            share_pct=100.0 * len(idx) / max(total_samples, 1),
            counts=counts, props=props_v,
            top3=top3.tolist(),
            n_classes_present=n_nonzero,
            gini=gini_c,
        ))

    # Federation-level
    fed_counts = sum(st["counts"] for st in client_stats)
    ns_arr     = np.sort([st["n_samples"] for st in client_stats]).astype(float)
    fed_gini   = (2*np.sum(np.arange(1, n+1)*ns_arr)/(n*ns_arr.sum())-(n+1)/n
                  ) if ns_arr.sum() > 0 else 0.0

    # -- Screen display --------------------------------------------------------
    section("Data split statistics  (before training)")
    BAR_W = min(n_classes, 30)
    print(f"\n  {cyan('Client')}  {cyan('Samples')}  {cyan('Share')}  {cyan('Cls')}"
          f"  {cyan('Top-3 classes')}  {cyan('Class distribution (bar)')}")
    print(f"  {dim('-' * W)}")

    for i, st in enumerate(client_stats):
        bar = "".join(
            green("#") if st["props"][c] > 0.20 else
            yellow("+") if st["props"][c] > 0.08 else
            dim(".")   if st["props"][c] > 0.02 else dim("*")
            for c in range(min(n_classes, BAR_W))
        )
        top_str = ", ".join(f"{c}({st['counts'][c]})" for c in st["top3"])
        gini_s  = f'G={st["gini"]:.2f}'
        print(f"  {bold(f'C{i+1:<5}')}  {st['n_samples']:>6}   "
              f"{st['share_pct']:>5.1f}%  {st['n_classes_present']:>3}   "
              f"{top_str:<20}  {bar}  {dim(gini_s)}")

    print(f"\n  {dim('-' * W)}")
    print(f"  {bold('Federation totals')}:")
    print(f"    Total samples : {total_samples}")
    print(f"    Sample Gini   : {fed_gini:.4f}  "
          + dim("(0=all clients equal size)"))
    print(f"    Classes held  : {int((fed_counts > 0).sum())} / {n_classes}")

    print(f"\n  {cyan('Per-class totals')}  (summed over all clients):")
    max_c = max(fed_counts) if fed_counts.max() > 0 else 1
    cls_per_row = min(n_classes, 20)
    for start in range(0, n_classes, cls_per_row):
        end  = min(start + cls_per_row, n_classes)
        head = "  " + "".join(f"{c:>5}" for c in range(start, end))
        vals = "  " + "".join(
            (green if fed_counts[c] == max_c else
             red   if fed_counts[c] == 0 else dim)(f"{fed_counts[c]:>5}")
            for c in range(start, end)
        )
        print(head)
        print(vals)

    split_type = split_cfg.get("split", "?")
    if split_type == "dirichlet":
        alpha = split_cfg.get("alpha", "?")
        degree_note = (f"Dirichlet alpha={alpha}  "
                       + ("(near-IID)" if float(alpha) >= 2
                          else "(moderate non-IID)" if float(alpha) >= 0.3
                          else "(strong non-IID)"))
    elif split_type == "label-skew":
        degree_note = f"Label-skew K={split_cfg.get('K_skew','?')} primary classes"
    else:
        degree_note = "IID - equal random split"

    print(f"\n  {dim(degree_note)}")

    # -- Plain-text block for the .txt report ----------------------------------
    txt = []
    txt.append(f"  {'Client':<8} {'Samples':>8} {'Share%':>7} "
               f"{'#Classes':>9} {'Gini':>7}  Top-3 dominant classes")
    txt.append("  " + "-" * 70)
    for i, st in enumerate(client_stats):
        top_str = ", ".join(f"cls{c}({st['counts'][c]})" for c in st["top3"])
        txt.append(f"  C{i+1:<7} {st['n_samples']:>8} {st['share_pct']:>7.1f} "
                   f"{st['n_classes_present']:>9} {st['gini']:>7.4f}  {top_str}")
    txt.append("  " + "-" * 70)
    txt.append(f"  Total samples : {total_samples}")
    txt.append(f"  Sample Gini   : {fed_gini:.4f}")
    txt.append(f"  Classes held  : {int((fed_counts > 0).sum())} / {n_classes}")
    txt.append(f"  Heterogeneity : {degree_note}")
    txt.append("")
    txt.append("  Per-class sample counts (federation total):")
    txt.append("  " + "-" * 50)
    for start in range(0, n_classes, 10):
        end  = min(start + 10, n_classes)
        head = "  " + "".join(f"  cls{c:<4}" for c in range(start, end))
        vals = "  " + "".join(f"{fed_counts[c]:>8}" for c in range(start, end))
        txt.append(head)
        txt.append(vals)

    return {
        "client_stats":  client_stats,
        "fed_counts":    fed_counts.tolist(),
        "fed_gini":      float(fed_gini),
        "total_samples": total_samples,
        "n_classes":     n_classes,
        "split_type":    split_type,
        "degree_note":   degree_note,
        "txt_block":     "\n".join(txt),
    }


# =============================================================================
# Interactive dataset + split configuration menu
# =============================================================================

def subsample_dataset(
    Xtr: np.ndarray, Ytr: np.ndarray,
    Xte: np.ndarray, Yte: np.ndarray,
    n_train: int,
    n_test:  int = 0,
    seed:    int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified subsample of the training set (and optionally the test set).

    Stratified means each class keeps the same proportion as in the full set,
    so the class distribution is preserved exactly.

    Parameters
    ----------
    n_train : desired training set size  (<= len(Ytr))
    n_test  : desired test set size      (0 = keep full test set)
    seed    : random seed for reproducibility

    Returns
    -------
    Xtr_sub, Ytr_sub, Xte_sub, Yte_sub
    """
    rng = np.random.default_rng(seed)

    def _stratified(X, Y, n_keep):
        if n_keep >= len(Y):
            return X, Y
        classes, counts = np.unique(Y, return_counts=True)
        # Proportional allocation - at least 1 sample per class
        proportions = counts / counts.sum()
        alloc = np.maximum(1, np.round(proportions * n_keep).astype(int))
        # Adjust total to exactly n_keep
        diff = n_keep - alloc.sum()
        if diff > 0:
            alloc[np.argmax(proportions)] += diff
        elif diff < 0:
            for _ in range(-diff):
                cand = np.where(alloc > 1)[0]
                alloc[cand[np.argmin(alloc[cand])]] -= 1
        idx_keep = []
        for cls, n_cls in zip(classes, alloc):
            cls_idx = np.where(Y == cls)[0]
            chosen  = rng.choice(cls_idx, size=int(n_cls), replace=False)
            idx_keep.append(chosen)
        idx_keep = np.concatenate(idx_keep)
        rng.shuffle(idx_keep)
        return X[idx_keep], Y[idx_keep]

    Xtr_s, Ytr_s = _stratified(Xtr, Ytr, n_train)
    if n_test > 0:
        Xte_s, Yte_s = _stratified(Xte, Yte, n_test)
    else:
        Xte_s, Yte_s = Xte, Yte

    ok(f"Subsample  train: {len(Ytr):,} -> {len(Ytr_s):,}  "
       f"test: {len(Yte):,} -> {len(Yte_s):,}  (stratified, seed={seed})")
    return Xtr_s, Ytr_s, Xte_s, Yte_s


def dataset_menu(data_root: str = "./data") -> Tuple[str, dict,
                                                       np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    """
    Interactive Steps 1-4: choose dataset, partitioning strategy,
    federation size, and optional stratified subsample.

    Returns
    -------
    ds_name   : str
    split_cfg : dict  (split, alpha, K_skew, n_classes, n_clients, batch_size)
    Xtr, Ytr, Xte, Yte : numpy arrays
    """
    banner("FibFL  -  Module 1: Dataset & Partitioning")

    # Step 1: Dataset
    section("Step 1 / 4  -  Dataset")
    ds_choice = menu("Select dataset:", DS_MENU, default=1)
    ds_name   = DS_NAMES[ds_choice - 1]
    ds_info   = DATASETS[ds_name]
    n_classes = ds_info["n_classes"]
    ok(f"{bold(ds_name)}  ({n_classes} classes, D={ds_info['input_dim']})")

    # Step 2: Partitioning
    section("Step 2 / 4  -  Data partitioning")
    sp_choice = menu("Partitioning strategy:", [
        ("IID",        "Equal random split - homogeneous baseline"),
        ("Dirichlet",  "Dir(alpha) class proportions - standard non-IID benchmark"),
        ("Label-skew", "K primary + K+1 secondary + 3% minority classes"),
    ], default=2)
    split = {1: "iid", 2: "dirichlet", 3: "label-skew"}[sp_choice]

    alpha  = 0.5
    K_skew = 2
    if split == "dirichlet":
        print(f"  {dim('alpha=1 ~= IID,  alpha=0.5 moderate,  alpha=0.1 strong skew')}")
        alpha = ask_float("Dirichlet alpha", 0.5, 0.01, 10.0)
    elif split == "label-skew":
        K_skew = ask_int("Primary classes per client K",
                         2, 1, max(1, n_classes // 3))

    # Step 3: Federation size
    section("Step 3 / 4  -  Federation size")
    n_clients = ask_int("Number of clients N", 10, 2, 100)
    n_rounds  = ask_int("Number of FL rounds R", 20, 1, 300)

    split_cfg = dict(
        split=split, alpha=alpha, K_skew=K_skew,
        n_classes=n_classes, n_clients=n_clients,
        n_rounds=n_rounds, batch_size=64,
    )

    # Load full dataset
    banner(f"Loading  {ds_name}")
    Xtr, Ytr, Xte, Yte = load_dataset(ds_name, data_root)
    full_train = len(Ytr)
    full_test  = len(Yte)

    # Step 4: Optional subsample
    section("Step 4 / 4  -  Dataset subsample  (optional)")
    print(f"  Full training set : {bold(str(full_train)):>8} samples")
    print(f"  Full test set     : {bold(str(full_test)):>8} samples")
    print(f"  {dim('Subsampling is stratified - class proportions are preserved.')}")
    print(f"  {dim('Useful for quick experiments or memory-constrained machines.')}")
    print()

    do_sub = confirm("Use a subsample of the training set?", default=False)
    if do_sub:
        # Suggest sensible presets
        presets = []
        for frac in (0.1, 0.25, 0.5, 0.75):
            n = int(full_train * frac)
            if n >= n_clients * n_classes:   # at least one sample per class per client
                presets.append(n)
        preset_str = "  |  ".join(f"{p:,}" for p in presets) if presets else ""
        if preset_str:
            print(f"  {dim(f'Suggested sizes: {preset_str}')}")

        n_train_max = full_train
        n_train_min = max(n_clients * n_classes, 100)
        n_train_sub = ask_int(
            f"Training samples to keep [{n_train_min:,} - {n_train_max:,}]",
            default=min(int(full_train * 0.5), full_train),
            lo=n_train_min,
            hi=n_train_max,
        )

        do_test_sub = confirm("Also subsample the test set?", default=False)
        n_test_sub  = 0
        if do_test_sub:
            n_test_sub = ask_int(
                f"Test samples to keep [100 - {full_test:,}]",
                default=min(int(full_test * 0.5), full_test),
                lo=100,
                hi=full_test,
            )

        Xtr, Ytr, Xte, Yte = subsample_dataset(
            Xtr, Ytr, Xte, Yte,
            n_train=n_train_sub,
            n_test=n_test_sub,
        )
        # Record subsample sizes in split_cfg for reporting
        split_cfg["subsample_train"] = len(Ytr)
        split_cfg["subsample_test"]  = len(Yte) if do_test_sub else full_test
    else:
        ok("Using full dataset.")

    return ds_name, split_cfg, Xtr, Ytr, Xte, Yte

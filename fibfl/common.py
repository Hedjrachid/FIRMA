"""
fl_common.py
============
Shared utilities used by every module:
  - Terminal colour / UI helpers
  - Neural network architecture (Extractor, Head, FullModel)
  - Training primitives (train_epoch, eval_acc, gp, sp, wavg, lerp, _log)
  - Fibonacci / golden-ratio constants used by fl_fibfl.py

Import pattern (all other modules):
    from fl_common import *
"""

import math
import os
import sys
import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# -- Device --------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

# -- Fibonacci / golden-ratio constants ----------------------------------------
# ALPHA + BETA = 1  (Fibonacci normalisation identity)
PHI   = (1 + math.sqrt(5)) / 2
ALPHA = 1.0 / PHI        # ~= 0.618  left-neighbour weight
BETA  = 1.0 / PHI ** 2   # ~= 0.382  right-neighbour weight
assert abs(ALPHA + BETA - 1.0) < 1e-12, "Fibonacci identity violated"


# =============================================================================
# Terminal UI
# =============================================================================

def _col(text, code):
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

def bold(t):   return _col(t, 1)
def cyan(t):   return _col(t, 96)
def green(t):  return _col(t, 92)
def yellow(t): return _col(t, 93)
def red(t):    return _col(t, 91)
def dim(t):    return _col(t, 2)


def banner(title: str) -> None:
    w = 65
    print()
    print(cyan("+" + "="*(w-2) + "+"))
    print(cyan("|") + bold(f"  {title}".center(w-2)) + cyan("|"))
    print(cyan("+" + "="*(w-2) + "+"))


def section(title: str) -> None:
    print(f"\n{cyan('-'*4)} {bold(title)} {cyan('-'*max(0,56-len(title)))}")


def menu(title: str, options: List[Tuple[str, str]], default: int = 1) -> int:
    """Numbered selection menu.  Returns 1-based index."""
    print(f"\n  {bold(title)}")
    for i, (label, desc) in enumerate(options, 1):
        mk = green(f"  [{i}]") if i == default else dim(f"  [{i}]")
        print(f"{mk}  {bold(label):<22} {dim(desc)}")
    while True:
        raw = input(f"\n  {cyan('->')} Choice [{default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw)
        print(f"  {red('Invalid.')} Enter 1-{len(options)}.")


def ask_int(prompt: str, default: int, lo: int, hi: int) -> int:
    while True:
        raw = input(f"  {cyan('->')} {prompt} [{default}]: ").strip()
        if raw == "":
            return default
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"  {red('Invalid.')} Enter an integer {lo}-{hi}.")


def ask_float(prompt: str, default: float, lo: float, hi: float) -> float:
    while True:
        raw = input(f"  {cyan('->')} {prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  {red('Invalid.')} Enter a float {lo}-{hi}.")


def confirm(prompt: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    raw  = input(f"  {cyan('->')} {prompt} {hint}: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


def ok(msg: str)   -> None: print(f"  {green('[ok]')}  {msg}")
def info(msg: str) -> None: print(f"  {dim('[i]')}  {dim(msg)}")
def prog(msg: str) -> None: print(f"  {yellow('~')}  {msg}", flush=True)


# =============================================================================
# Neural network architecture
# =============================================================================

class Extractor(nn.Module):
    """
    Three-layer MLP feature extractor.

        Linear(d, h) -> LayerNorm -> ReLU
        Linear(h, h) -> LayerNorm -> ReLU
        Linear(h, e) -> ReLU

    LayerNorm avoids BatchNorm's batch-size > 1 requirement, which fails
    when a client has very few samples under strong non-IID partitioning.
    """
    def __init__(self, d: int, h: int = 256, e: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
            nn.Linear(h, e), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Head(nn.Module):
    """Linear classification head.  Permanently private in FibFL variants."""
    def __init__(self, e: int = 128, n_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(e, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class FullModel(nn.Module):
    """Combined extractor + head for joint forward passes."""
    def __init__(self, ext: Extractor, head: Head):
        super().__init__()
        self.ext  = ext
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.ext(x))


def build_models(in_dim: int, n: int, emb: int,
                 n_classes: int) -> List[Tuple[Extractor, Head]]:
    """Create fresh (Extractor, Head) pairs for N clients."""
    return [(Extractor(in_dim, e=emb).to(DEVICE),
             Head(emb, n_classes).to(DEVICE))
            for _ in range(n)]


# =============================================================================
# Training primitives  (shared by all FL modules)
# =============================================================================

def train_epoch(model: nn.Module, loader: DataLoader,
                opt: torch.optim.Optimizer) -> None:
    """One epoch of supervised training with cross-entropy loss."""
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()


def eval_acc(ext: Extractor, head: Head, loader: DataLoader) -> float:
    """Test accuracy of (ext, head) on loader."""
    ext.eval(); head.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (head(ext(x)).argmax(1) == y).sum().item()
            total   += len(y)
    return correct / max(total, 1)


def gp(m: nn.Module) -> List[torch.Tensor]:
    """Clone all parameters of a module."""
    return [p.data.clone() for p in m.parameters()]


def sp(m: nn.Module, ps: List[torch.Tensor]) -> None:
    """Copy parameter list into module in-place."""
    for p, v in zip(m.parameters(), ps):
        p.data.copy_(v)


def wavg(pls: List[List[torch.Tensor]],
         ws: List[float]) -> List[torch.Tensor]:
    """Weighted average: sum w_k * params_k."""
    out = [torch.zeros_like(p) for p in pls[0]]
    for w, ps in zip(ws, pls):
        for o, p in zip(out, ps):
            o.add_(p, alpha=w)
    return out


def lerp(a: List[torch.Tensor], b: List[torch.Tensor],
         t: float) -> List[torch.Tensor]:
    """Linear interpolation: t*a + (1-t)*b."""
    return [t*pa + (1-t)*pb for pa, pb in zip(a, b)]


def _log(method: str, r: int, accs: List[float], log_every: int) -> None:
    """Print a per-round progress line if round is a logging checkpoint."""
    if r % log_every == 0:
        print(f"  {dim(f'{method:<14}')}  R={r:3d}"
              f"  mean={green(f'{np.mean(accs):.4f}')}"
              f"  std={dim(f'{np.std(accs):.4f}')}"
              f"  [{np.min(accs):.3f}, {np.max(accs):.3f}]")


# =============================================================================
# Results serialisation helper
# =============================================================================

def ser(h: List[List[float]]) -> List[List[float]]:
    """Convert nested list to plain Python floats (for JSON / pickle)."""
    return [[float(a) for a in row] for row in h]

"""
FibFL: Fibonacci-Weighted Federated Learning Protocol Family
=============================================================
A peer-to-peer ring federated learning framework with no central server.

Modules
-------
    fibfl.common   -- shared utilities, neural network architecture, training primitives
    fibfl.data     -- dataset loading, partitioning (IID / Dirichlet / label-skew)
    fibfl.rdfl     -- RDFL baseline (Wang et al., 2021)
    fibfl.fibfl    -- FibFL, FibFL+, FibFL++ protocol implementations
    fibfl.runner   -- parallel experiment runner (FIRMA)
"""

from fibfl.common import (
    DEVICE, PHI, ALPHA, BETA,
    Extractor, Head, FullModel, build_models,
    train_epoch, eval_acc,
    gp, sp, wavg, lerp, _log, ser,
    banner, section, menu, ok, info, prog,
    bold, cyan, green, yellow, red, dim,
)

from fibfl.fibfl import (
    run_fibfl_basic, run_fibflp, run_fibflpp,
    ask_fibfl_cfg, fibfl_on_split,
    two_opt, fib_blend,
)

from fibfl.rdfl import (
    run_rdfl, ask_rdfl_cfg, rdfl_on_split,
)

__version__ = "1.0.0"
__author__  = "FibFL Authors"

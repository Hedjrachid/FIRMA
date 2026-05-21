"""
tests/test_core.py
==================
Basic unit tests for FibFL core components.
Run with: python -m pytest tests/
"""

import math
import numpy as np
import pytest


# =============================================================================
# Fibonacci constants
# =============================================================================

def test_fibonacci_identity():
    """ALPHA + BETA must equal 1 (Fibonacci normalisation identity)."""
    from fibfl.common import ALPHA, BETA
    assert abs(ALPHA + BETA - 1.0) < 1e-12


def test_fibonacci_values():
    """Verify golden-ratio-derived weights are correct."""
    from fibfl.common import PHI, ALPHA, BETA
    assert abs(PHI - (1 + math.sqrt(5)) / 2) < 1e-12
    assert abs(ALPHA - 1.0 / PHI) < 1e-12
    assert abs(BETA  - 1.0 / PHI**2) < 1e-12
    assert ALPHA > BETA   # left neighbour always gets more weight


# =============================================================================
# Data partitioning
# =============================================================================

def test_iid_split_sizes():
    """IID split should give equal-ish partition sizes."""
    from fibfl.data import iid_split
    Y = np.array([i % 10 for i in range(1000)])
    idxs = iid_split(Y, n=10)
    assert len(idxs) == 10
    for idx in idxs:
        assert 90 <= len(idx) <= 110   # roughly equal


def test_iid_split_no_overlap():
    """IID split indices must be disjoint and cover the full dataset."""
    from fibfl.data import iid_split
    Y = np.zeros(500, dtype=int)
    idxs = iid_split(Y, n=5)
    all_idx = np.concatenate(idxs)
    assert len(all_idx) == 500
    assert len(set(all_idx)) == 500   # no duplicates


def test_dirichlet_split_sizes():
    """Dirichlet split should produce N non-empty partitions."""
    from fibfl.data import dirichlet_split
    Y = np.array([i % 10 for i in range(1000)])
    idxs = dirichlet_split(Y, n=5, alpha=0.5, n_classes=10)
    assert len(idxs) == 5
    for idx in idxs:
        assert len(idx) > 0


def test_class_props_sum_to_one():
    """class_props must return a probability vector summing to 1."""
    from fibfl.data import class_props
    Y   = np.array([0, 0, 1, 1, 2, 3])
    idx = np.arange(6)
    props = class_props(Y, idx, n_classes=4)
    assert props.shape == (4,)
    assert abs(props.sum() - 1.0) < 1e-9


# =============================================================================
# Model construction
# =============================================================================

def test_build_models_count():
    """build_models should return exactly N (extractor, head) pairs."""
    from fibfl.common import build_models
    models = build_models(in_dim=64, n=5, emb=32, n_classes=10)
    assert len(models) == 5
    for ext, head in models:
        assert ext is not None
        assert head is not None


def test_extractor_output_shape():
    """Extractor should output (batch, emb_dim) tensors."""
    import torch
    from fibfl.common import Extractor, DEVICE
    ext = Extractor(d=64, e=32).to(DEVICE)
    x   = torch.randn(8, 64).to(DEVICE)
    out = ext(x)
    assert out.shape == (8, 32)


def test_head_output_shape():
    """Head should output (batch, n_classes) tensors."""
    import torch
    from fibfl.common import Head, DEVICE
    head = Head(e=32, n_classes=10).to(DEVICE)
    x    = torch.randn(8, 32).to(DEVICE)
    out  = head(x)
    assert out.shape == (8, 10)


# =============================================================================
# Training primitives
# =============================================================================

def test_wavg_sum_to_params():
    """Weighted average of identical params should return the same params."""
    import torch
    from fibfl.common import gp, sp, wavg, Extractor, DEVICE
    ext = Extractor(d=16, e=8).to(DEVICE)
    p   = gp(ext)
    result = wavg([p, p], [0.5, 0.5])
    for r, orig in zip(result, p):
        assert torch.allclose(r, orig, atol=1e-6)


def test_lerp_endpoints():
    """lerp at t=1 should return a, at t=0 should return b."""
    import torch
    from fibfl.common import lerp
    a = [torch.tensor([1.0, 2.0])]
    b = [torch.tensor([3.0, 4.0])]
    assert torch.allclose(lerp(a, b, 1.0)[0], a[0])
    assert torch.allclose(lerp(a, b, 0.0)[0], b[0])


# =============================================================================
# 2-opt ring ordering
# =============================================================================

def test_two_opt_permutation():
    """two_opt must return a valid permutation of N clients."""
    from fibfl.fibfl import two_opt
    N = 8
    C = 10
    rng = np.random.default_rng(0)
    q   = rng.dirichlet(np.ones(C), size=N)  # (N, C) normalised
    sigma, c_seq, c_opt = two_opt(q)
    assert len(sigma) == N
    assert set(sigma) == set(range(N))
    assert c_opt <= c_seq + 1e-9   # 2-opt must not increase cost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

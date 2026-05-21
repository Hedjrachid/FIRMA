# FibFL

Fibonacci-weighted federated learning on a ring topology — no central server.

## Install

```bash
git clone https://github.com/YOUR_USERNAME/fibfl.git
cd fibfl
pip install -r requirements.txt
```

## Run

```bash
# Interactive
python run_firma.py

# Auto (reproduces paper experiments)
python run_firma.py --auto

# Options
python run_firma.py --workers 4 --outdir ./results --datadir ./data
```

## Structure

```
fibfl/
|-- fibfl/
|   |-- common.py   # architecture, training primitives
|   |-- data.py     # dataset loading, IID/Dirichlet/label-skew splits
|   |-- rdfl.py     # RDFL baseline (Wang et al., 2021)
|   |-- fibfl.py    # FibFL, FibFL+, FibFL++
|   |-- runner.py   # parallel experiment runner
|-- tests/
|-- run_firma.py
|-- requirements.txt
```

## Methods

| Method | Weights | Head private | Ring ordering |
|---|---|---|---|
| RDFL | Uniform (0.5, 0.5) | No | Sequential |
| FibFL | Fibonacci (0.618, 0.382) | Yes | Sequential |
| FibFL+ | Fibonacci + accuracy gate | Yes | Sequential |
| FibFL++ | Fibonacci + accuracy gate | Yes | 2-opt optimised |

## Reference

```bibtex
@article{fibfl2025,
  title  = {FibFL: A Fibonacci-Weighted Federated Learning Protocol Family
            for Personalised Learning on a Ring Topology Without a Central Server},
  author = {YOUR NAME},
  year   = {2025},
}
```

Baseline: Wang et al., arXiv 2104.08100, 2021.

## License

MIT

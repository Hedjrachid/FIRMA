# Methods

## Fibonacci weights

```
phi   = (1 + sqrt(5)) / 2
alpha = 1 / phi    # ~= 0.618  left neighbour
beta  = 1 / phi^2  # ~= 0.382  right neighbour
alpha + beta = 1
```

## Variants

**RDFL** -- uniform weights (0.5, 0.5), full model shared, no head privacy.

**FibFL** -- Fibonacci weights, head permanently private, constant gamma.

**FibFL+** -- FibFL + accuracy gate: neighbour only contributes if accuracy > tau.

**FibFL++** -- FibFL+ + 2-opt ring ordering + K=ceil(N/2) gossip passes + cosine-annealed gamma.

## Hyperparameters

| Param | Default | Description |
|---|---|---|
| E | 5 | Local epochs (RDFL) |
| Eh | 1 | Head epochs per round |
| Ee | 20 | Extractor epochs per round |
| lr | 0.01 | Learning rate |
| gamma | 0.5 | Self-retention (0=full gossip, 1=no gossip) |
| gamma_start | 0.4 | Cosine anneal start (FibFL++) |
| gamma_end | 0.05 | Cosine anneal end (FibFL++) |
| tau | 0.35 | Accuracy gate threshold (FibFL+/++) |
| emb_dim | 128 | Embedding dimension |

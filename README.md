# pathsig

pathsig is a GPU-accelerated library for differentiable signature and log-signature computations. It offers greater flexibility through projections and windows, and achieves **~4–10×** speedups for training (forward + backward) and **~10–30×** speedups for forward-only evaluation in our benchmarks.

Documentation can be found at https://pathsig.readthedocs.io.

## Installation

```bash
pip install pathsig
```

`pathsig` is currently built from source (prebuilt wheels are not yet published).
If you already have build requirements available locally, you can often speed up installation by disabling build isolation:
```bash
pip install scikit-build-core
pip install pathsig --no-build-isolation
```

## Quickstart
The example below computes truncated, windowed, and projected signatures. Log-signatures are computed analogously, typically with a suitable projection (see the [documentation](https://pathsig.readthedocs.io/en/latest/api/logsignature.html)).
```python
import torch
import pathsig

x = torch.randn(32, 128, 8, device="cuda", dtype=torch.float32)

# Truncated signature
sig = pathsig.Signature(depth=4)
y = sig(x)  # (B, D)

# Windowed signature
windows = torch.tensor([[0, 32], [32, 64]], device="cuda")  # (W, 2)
sig = pathsig.Signature(depth=4, windows=windows)
y = sig(x)  # (B, W, D)

# Signature with a word projection
proj = pathsig.projections.words(
    words=[(0, 1), (2, 2, 3)],
    depth=4,
    path_dim=8,
    full_levels=(1,),
)
sig = pathsig.Signature(depth=4, projection=proj)
y = sig(x)  # (B, proj.sig_size)
```


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


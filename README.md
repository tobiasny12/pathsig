# pathsig
A high-performance, GPU-accelerated library for differentiable signature computations, with PyTorch integration. Built on a decomposition approach that enables efficient, scalable signature computations with minimal memory usage.

## Installation
```bash
pip install pathsig
```

**Requirements:**
- PyTorch >= 2.0
- CUDA-capable GPU (compute capability >= 7.0)
- Python >= 3.8

## Quick Start
`pathsig` provides both a functional API and a PyTorch module for computing path signatures. The library allows for straightforward usage of the signature as part of machine learning models, acting as a differentiable feature extractor for time series or path data.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import pathsig

# Compute signature of a path
path = torch.randn(1, 100, 3, device='cuda') # (batch_size, sequence_length, path_dim)
sig = pathsig.signature(path, truncation_level=4)
print(sig.shape) # (1, 120)

# Use as a PyTorch module
signature_layer = pathsig.Signature(truncation_level=4).to('cuda')
sig = signature_layer(path)

# As part of a neural network 
class SignatureNet(nn.Module):
    def __init__(self, input_channels, sig_level, num_classes):
        super().__init__()
        self.signature = pathsig.Signature(truncation_level=sig_level)
        sig_dim = pathsig.sig_size(input_channels, sig_level)
        self.classifier = nn.Linear(sig_dim, num_classes)
 
    def forward(self, x):
        sig = self.signature(x)
        return self.classifier(sig)
```


## Limitations
- GPU-only (no CPU support currently).
- Maximum truncation level: 12.
- Maximum path dimension: 1000.

## License
MIT License, see [LICENSE](LICENSE) file for details.
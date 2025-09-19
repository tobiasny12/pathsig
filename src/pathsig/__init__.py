# pathsig/__init__.py
import torch
from .Signature import Signature, signature
from ._impl import sig_size

__all__ = ["Signature", "signature", "sig_size"]
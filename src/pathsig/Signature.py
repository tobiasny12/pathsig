import torch
import torch.nn as nn
from ._impl import SigDecomposition
from ._autograd import SignatureFunction

class Signature(nn.Module):
    def __init__(self, truncation_level: int, extended_precision: bool = False):
        """
        Args:
            truncation_level: Maximum degree of signature terms
            extended_precision: If True and dtype is double, use higher precision during computation
        """
        super(Signature, self).__init__()
        self.truncation_level = truncation_level
        self.extended_precision = extended_precision
        self.sig_decomp = None
        self._cached_path_dim = None

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        """
        Computes the signatures of the input paths.
        Args:
            path: Input tensor of shape (batch_size, sequence_length, path_dim)
        Returns:
            Tensor containing signatures of shape (batch_size, signature_size)
        """
        batch_size, sequence_length, path_dim = path.shape
        if (self.sig_decomp is None or
                self._cached_path_dim != path_dim):
            self.sig_decomp = SigDecomposition(path_dim, self.truncation_level)
            self._cached_path_dim = path_dim
        return SignatureFunction.apply(path, self.truncation_level, self.extended_precision, self.sig_decomp)

def signature(path: torch.Tensor, truncation_level: int, extended_precision: bool = False, sig_decomp=None) -> torch.Tensor:
    """
    Computes the signature/signatures of a path/paths up to a given truncation level.

    Args:
        path: Input tensor of shape (batch_size, num_time_steps, path_dim)
        truncation_level: Maximum degree of signature terms
        extended_precision: If True and path is of dtype FP64, use higher precision in signature computations
    Returns:
        Tensor containing the truncated signature of the input path
    """
    return SignatureFunction.apply(path, truncation_level, extended_precision, sig_decomp)
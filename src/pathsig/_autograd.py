import torch
from ._impl import compute_sig, compute_sig_gradients, SigDecomposition
from torch.autograd.function import once_differentiable
from typing import Optional, Tuple, Union

class SignatureFunction(torch.autograd.Function):
    """
    Custom autograd function for signature computation and differentiation.
    """
    @staticmethod
    def forward(path: torch.Tensor, truncation_level: int,
                extended_precision: bool = False, sig_decomp: Optional[SigDecomposition] = None) -> torch.Tensor:
        """
        Args:
            path: Input tensor of shape (batch_size, num_time_steps, path_dim)
            truncation_level: Maximum degree of signature terms
            extended_precision: If True and path is of dtype FP64, use higher precision in signature computation
        Returns:
            Signatures torch tensor of shape (batch_size, signature_size)
        """

        # Compute signatures
        sig = compute_sig(path, truncation_level, extended_precision, sig_decomp)
        return sig

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Args:
            ctx: Context object
            inputs: Tuple of inputs to forward
            output: Output of forward
        """
        path, truncation_level, extended_precision, sig_decomp = inputs
        sig = output
        if path.requires_grad:
            ctx.save_for_backward(path, sig)
            ctx.truncation_level = truncation_level
            ctx.sig_decomp = sig_decomp

    @staticmethod
    def backward(ctx, incoming_gradients: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        Args:
            ctx: Context object with saved tensors
            incoming_gradients: Gradient of loss w.r.t. signature
        Returns:
            gradient of signature w.r.t. input path, and None for other inputs
        """
        path, sig = ctx.saved_tensors
        path_grad = compute_sig_gradients(
            path, sig, incoming_gradients, ctx.truncation_level, ctx.sig_decomp
        )
        return path_grad, None, None, None
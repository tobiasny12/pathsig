# pathsig/ops.py
from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "compute_signature",
    "sig_to_logsig",
    "signature_backward",
    "logsig_backward",
]


# Common validation helpers
def _check_path(path: Tensor) -> tuple[int, int, int]:
    """Validate the input path tensor (used by compute_signature and signature_backward)."""
    torch._check(path.device.type in ("cuda", "meta"), "path must be a CUDA tensor")
    torch._check(path.dtype in (torch.float32, torch.float64), "path must be float32 or float64")
    torch._check(path.dim() == 3, "path must have shape (B, L, d)")
    B, T, d = path.shape
    torch._check(T >= 2, "path length must be at least 2")
    return B, T, d


def _check_windows(windows: Tensor) -> None:
    """Validate the windows tensor."""
    torch._check(windows.device.type in ("cuda", "meta"), "windows must be a CUDA tensor")
    torch._check(windows.dtype in (torch.int32, torch.int64), "windows must be int32 or int64")
    torch._check(windows.dim() == 2, "windows must have shape (num_windows, 2)")
    torch._check(windows.size(1) == 2, "windows must have shape (num_windows, 2)")


def _check_projection(encoded_words: Tensor, level_sizes: list[int], depth: int) -> None:
    """Validate projection data when alternative_projection=True."""
    torch._check(encoded_words.device.type in ("cuda", "meta"), "encoded_words must be a CUDA tensor")
    torch._check(encoded_words.dtype == torch.int64, "encoded_words must be int64")
    torch._check(encoded_words.dim() == 1, "encoded_words must be 1D")
    torch._check(isinstance(level_sizes, (list, tuple)), "level_sizes must be a list/tuple")
    torch._check(len(level_sizes) >= depth + 1, "level_sizes must cover at least depth + 1 entries")


# Forward ops
def compute_signature(
        path: Tensor,
        depth: int,
        alternative_projection: bool,
        encoded_words: Tensor,
        level_sizes: list[int],
        use_windows: bool,
        windows: Tensor,
) -> Tensor:
    """Compute the (possibly windowed/projected) signature of a path."""
    return torch.ops.pathsig.compute_signature.default(
        path, depth, alternative_projection, encoded_words, level_sizes, use_windows, windows
    )


@torch.library.register_fake("pathsig::compute_signature")
def _fake_compute_signature(
        path,
        depth,
        alternative_projection,
        encoded_words,
        level_sizes,
        use_windows,
        windows,
):
    B, _, d = _check_path(path)
    depth_i = int(depth)
    torch._check(1 <= depth_i <= 12, "depth must satisfy 1 <= depth <= 12")

    if use_windows:
        _check_windows(windows)

    if alternative_projection:
        _check_projection(encoded_words, level_sizes, depth_i)
        sig_size = int(sum(level_sizes[1 : depth_i + 1]))
    else:
        sig_size = int(sum(d**k for k in range(1, depth_i + 1)))

    if use_windows:
        return torch.empty((B, windows.size(0), sig_size), device=path.device, dtype=path.dtype)
    return torch.empty((B, sig_size), device=path.device, dtype=path.dtype)


def sig_to_logsig(
        signature: Tensor,
        depth: int,
        d: int,
        alternative_projection: bool,
        encoded_words: Tensor,
        level_sizes: list[int],
) -> tuple[Tensor, Tensor]:
    """Convert signature → log-signature (returns (logsig, P_arr_cache))."""
    return torch.ops.pathsig.sig_to_logsig.default(
        signature, depth, d, alternative_projection, encoded_words, level_sizes
    )


@torch.library.register_fake("pathsig::sig_to_logsig")
def _fake_sig_to_logsig(signature, depth, d, alternative_projection, encoded_words, level_sizes):
    torch._check(signature.device.type in ("cuda", "meta"), "signature must be a CUDA tensor")
    torch._check(signature.dtype in (torch.float32, torch.float64), "signature must be float32 or float64")

    depth_i = int(depth)
    d_i = int(d)
    torch._check(1 <= depth_i <= 12, "depth must satisfy 1 <= depth <= 12")
    torch._check(d_i >= 1, "path dimension d must be >= 1")

    if signature.dim() == 3:
        B, W, S = signature.shape
        P_arr = torch.empty((B, W, S * depth_i), device=signature.device, dtype=signature.dtype)
    else:
        torch._check(signature.dim() == 2, "signature must be 2D or 3D")
        B, S = signature.shape
        P_arr = torch.empty((B, S * depth_i), device=signature.device, dtype=signature.dtype)

    if alternative_projection:
        _check_projection(encoded_words, level_sizes, depth_i)

    logsig = torch.empty_like(signature)
    return logsig, P_arr


# Backward ops
def signature_backward(
        path: Tensor,
        signature: Tensor,
        grad_signature: Tensor,
        depth: int,
        alternative_projection: bool,
        encoded_words: Tensor,
        level_sizes: list[int],
        use_windows: bool,
        windows: Tensor,
) -> Tensor:
    """Compute dL/d(path) from dL/d(signature)."""
    return torch.ops.pathsig.signature_backward.default(
        path,
        signature,
        grad_signature,
        depth,
        alternative_projection,
        encoded_words,
        level_sizes,
        use_windows,
        windows,
    )


@torch.library.register_fake("pathsig::signature_backward")
def _fake_signature_backward(
        path, signature, grad_signature, depth, alternative_projection, encoded_words, level_sizes, use_windows, windows
):
    _check_path(path)
    torch._check(signature.device.type in ("cuda", "meta"))
    torch._check(grad_signature.device.type in ("cuda", "meta"))

    torch._check(signature.dtype == path.dtype, "signature must have same dtype as path")
    torch._check(grad_signature.dtype == path.dtype, "grad_signature must have same dtype as path")
    torch._check(grad_signature.shape == signature.shape)

    depth_i = int(depth)
    torch._check(1 <= depth_i <= 12)

    if use_windows:
        _check_windows(windows)

    if use_windows:
        torch._check(signature.dim() == 3)
        torch._check(windows.size(0) == signature.size(1))
    else:
        torch._check(signature.dim() == 2)

    if alternative_projection:
        _check_projection(encoded_words, level_sizes, depth_i)

    return torch.empty_like(path)


def logsig_backward(
        signature: Tensor,
        P_arr: Tensor,
        grad_logsig: Tensor,
        depth: int,
        alternative_projection: bool,
        encoded_words: Tensor,
        level_sizes: list[int],
) -> Tensor:
    """Compute dL/d(signature) from dL/d(logsig) using the forward P_arr cache."""
    return torch.ops.pathsig.logsig_backward.default(
        signature, P_arr, grad_logsig, depth, alternative_projection, encoded_words, level_sizes
    )


@torch.library.register_fake("pathsig::logsig_backward")
def _fake_logsig_backward(signature, P_arr, grad_logsig, depth, alternative_projection, encoded_words, level_sizes):
    torch._check(signature.device.type in ("cuda", "meta"))
    torch._check(P_arr.device.type in ("cuda", "meta"))
    torch._check(grad_logsig.device.type in ("cuda", "meta"))

    torch._check(signature.dtype in (torch.float32, torch.float64), "signature must be float32 or float64")
    torch._check(signature.dtype == grad_logsig.dtype, "signature must have same dtype as grad_logsig")
    torch._check(grad_logsig.dtype == P_arr.dtype, "grad_logsig must have same dtype as P_arr")

    depth_i = int(depth)
    torch._check(1 <= depth_i <= 12)
    torch._check(grad_logsig.shape == signature.shape)

    if signature.dim() == 3:
        B, W, S = signature.shape
        torch._check(P_arr.dim() == 3)
        torch._check(P_arr.size(0) == B)
        torch._check(P_arr.size(1) == W)
        torch._check(P_arr.size(2) == S * depth_i)
    else:
        torch._check(signature.dim() == 2)
        B, S = signature.shape
        torch._check(P_arr.dim() == 2)
        torch._check(P_arr.size(0) == B)
        torch._check(P_arr.size(1) == S * depth_i)

    if alternative_projection:
        _check_projection(encoded_words, level_sizes, depth_i)

    return torch.empty_like(signature)


# Autograd registrations
def _setup_ctx_compute_signature(ctx, inputs, output):
    path, depth, alt, encoded_words, level_sizes, use_windows, windows = inputs
    ctx.save_for_backward(path, output, windows, encoded_words)
    ctx.depth = int(depth)
    ctx.alt = bool(alt)
    ctx.level_sizes = level_sizes
    ctx.use_windows = bool(use_windows)


def _backward_compute_signature(ctx, grad_out):
    path, signature, windows, encoded_words = ctx.saved_tensors
    grad_path = None
    if ctx.needs_input_grad[0]:
        grad_path = torch.ops.pathsig.signature_backward.default(
            path,
            signature,
            grad_out,
            ctx.depth,
            ctx.alt,
            encoded_words,
            ctx.level_sizes,
            ctx.use_windows,
            windows,
        )
    # grads for: path, depth, alt, encoded_words, level_sizes, use_windows, windows
    return grad_path, None, None, None, None, None, None


torch.library.register_autograd(
    "pathsig::compute_signature",
    _backward_compute_signature,
    setup_context=_setup_ctx_compute_signature,
)


def _setup_ctx_sig_to_logsig(ctx, inputs, output):
    signature, depth, d, alt, encoded_words, level_sizes = inputs
    logsig, P_arr = output
    ctx.save_for_backward(signature, P_arr, encoded_words)
    ctx.depth = int(depth)
    ctx.d = int(d)
    ctx.alt = bool(alt)
    ctx.level_sizes = level_sizes


def _backward_sig_to_logsig(ctx, grad_logsig, grad_P_arr):
    signature, P_arr, encoded_words = ctx.saved_tensors
    grad_sig = None
    if ctx.needs_input_grad[0]:
        grad_sig = torch.ops.pathsig.logsig_backward.default(
            signature,
            P_arr,
            grad_logsig,
            ctx.depth,
            ctx.alt,
            encoded_words,
            ctx.level_sizes,
        )
    # grads for: signature, depth, d, alt, encoded_words, level_sizes
    return grad_sig, None, None, None, None, None


torch.library.register_autograd(
    "pathsig::sig_to_logsig",
    _backward_sig_to_logsig,
    setup_context=_setup_ctx_sig_to_logsig,
)
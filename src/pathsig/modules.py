# pathsig/modules.py
from __future__ import annotations

import torch
from torch import nn

from .ops import compute_signature, sig_to_logsig


def _validate_depth(depth: int) -> int:
    if not (1 <= depth <= 12):
        raise ValueError("depth must satisfy 1 <= depth <= 12.")
    return depth


def _process_windows(windows: torch.Tensor | None) -> tuple[bool, torch.Tensor, bool]:
    if windows is None:
        return False, torch.empty((0, 2), dtype=torch.int32, device="cuda"), False

    if windows.device.type != "cuda":
        raise TypeError("windows must be a CUDA tensor.")
    if windows.dim() != 2 or windows.size(1) != 2:
        raise TypeError("windows must have shape [num_windows, 2].")
    if windows.dtype not in (torch.int32, torch.int64):
        raise TypeError("windows must have dtype int32 or int64.")

    return True, windows.contiguous().to(dtype=torch.int32), True


def _process_projection(projection) -> tuple[bool, torch.Tensor | None]:
    if projection is None:
        return False, None

    ew = projection.encoded_words
    if ew.device.type != "cuda" or ew.dtype != torch.int64 or ew.dim() != 1:
        raise TypeError("projection.encoded_words must be a 1D CUDA int64 tensor.")

    return True, ew.contiguous()


def _empty_cuda(dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(0, dtype=dtype, device="cuda")


class Signature(nn.Module):
    """Computes the signature of a path."""
    def __init__(self, *, depth: int, projection=None, windows=None):
        super().__init__()
        self.depth = _validate_depth(depth)
        self.projection = projection
        self.alternative_projection, encoded_words = _process_projection(projection)
        self.use_windows, win_buffer, win_persistent = _process_windows(windows)

        self.register_buffer("_windows", win_buffer, persistent=win_persistent)
        self.register_buffer(
            "_encoded_words",
            encoded_words if self.alternative_projection else _empty_cuda(torch.int64),
            persistent=self.alternative_projection,
        )
        self._level_sizes = projection.level_sizes if self.alternative_projection else []

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        return compute_signature(
            path,
            self.depth,
            self.alternative_projection,
            self._encoded_words,
            self._level_sizes,
            self.use_windows,
            self._windows,
        )


class LogSignature(nn.Module):
    """Computes the log-signature of a path."""

    def __init__(self, *, depth: int, projection=None, windows=None):
        super().__init__()
        self.depth = _validate_depth(depth)
        self.projection = projection
        self.alternative_projection, encoded_words = _process_projection(projection)
        self.use_windows, win_buffer, win_persistent = _process_windows(windows)

        self.register_buffer("_windows", win_buffer, persistent=win_persistent)

        if not self.alternative_projection:
            for name in ("_encoded_words", "_top_words", "_indices_from_sig"):
                self.register_buffer(name, _empty_cuda(torch.int64), persistent=False)
            self._level_sizes: list[int] = []
            self._ext_level_sizes: list[int] = []
            return

        self.register_buffer("_encoded_words", encoded_words, persistent=True)
        self._level_sizes = projection.level_sizes

        path_dim = self._path_dim = int(projection.path_dim)

        # Count how many entries come from levels 1...depth-1 that are projected (non-dense)
        lower_size, d_pow = 0, 1
        for k in range(1, self.depth):
            d_pow *= path_dim
            if d_pow != self._level_sizes[k]:
                lower_size += self._level_sizes[k]

        self.register_buffer("_top_words", encoded_words[lower_size:], persistent=True)

        # Levels 1..depth-1 are dense; level depth uses the projection size
        self._ext_level_sizes = [0] + [path_dim**k for k in range(1, self.depth)] + [int(self._level_sizes[self.depth])]
        idx = projection.logsig_indices()
        self.register_buffer("_indices_from_sig", idx.to(device="cuda", dtype=torch.int64).contiguous(), persistent=True)

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        if self.alternative_projection:
            path_dim = self._path_dim
            top_words = self._top_words
            ext_level_sizes = self._ext_level_sizes
        else:
            path_dim = int(path.shape[-1])
            top_words = self._top_words  # empty buffer
            ext_level_sizes = [0] + [path_dim**k for k in range(1, self.depth + 1)]

        sig = compute_signature(
            path, self.depth, self.alternative_projection,
            top_words, ext_level_sizes, self.use_windows, self._windows,
        )
        logsig, _ = sig_to_logsig(
            sig, self.depth, path_dim, self.alternative_projection,
            top_words, ext_level_sizes,
        )

        if not self.alternative_projection:
            return logsig

        idx = self._indices_from_sig
        return logsig.index_select(dim=-1, index=idx.to(logsig.device))
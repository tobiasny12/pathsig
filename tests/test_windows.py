# tests/test_windows.py
"""Windowed (log-)signature correctness, autograd, and shape tests."""
from __future__ import annotations

import pytest
import torch

import pathsig


# Helpers

def _make_path(B: int, M: int, d: int, dtype=torch.float64) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(B, M, d, device="cuda", dtype=dtype)


def _make_windows(seq_len: int, win_len: int, stride: int) -> torch.Tensor:
    """Return a (num_windows, 2) int32 tensor of [start, end) pairs."""
    starts = list(range(0, seq_len - win_len + 1, stride))
    return torch.tensor([[s, s + win_len] for s in starts], dtype=torch.int32, device="cuda")


# Windowed signature

class TestWindowedSignature:
    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3, 4])
    @pytest.mark.parametrize("B", [1, 4])
    def test_matches_per_window(self, depth, path_dim, B):
        seq_len, win_len, stride = 50, 15, 5
        path = _make_path(B, seq_len, path_dim)
        windows = _make_windows(seq_len, win_len, stride)

        windowed = pathsig.Signature(depth=depth, windows=windows)(path)
        ref = pathsig.Signature(depth=depth)

        assert windowed.dim() == 3
        assert windowed.size(1) == windows.size(0)

        for w in range(windows.size(0)):
            start, end = windows[w].tolist()
            expected = ref(path[:, start:end, :])
            torch.testing.assert_close(windowed[:, w, :], expected, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_single_window_equals_full(self, depth, path_dim):
        B, seq_len = 2, 30
        path = _make_path(B, seq_len, path_dim)
        windows = torch.tensor([[0, seq_len]], dtype=torch.int32, device="cuda")

        windowed = pathsig.Signature(depth=depth, windows=windows)(path)
        full = pathsig.Signature(depth=depth)(path)

        assert windowed.shape == (B, 1, full.size(1))
        torch.testing.assert_close(windowed[:, 0, :], full, atol=1e-10, rtol=1e-10)


# Windowed log-signature

class TestWindowedLogSignature:
    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3, 4])
    @pytest.mark.parametrize("B", [1, 4])
    def test_matches_per_window(self, depth, path_dim, B):
        seq_len, win_len, stride = 50, 15, 5
        path = _make_path(B, seq_len, path_dim)
        windows = _make_windows(seq_len, win_len, stride)

        windowed = pathsig.LogSignature(depth=depth, windows=windows)(path)
        ref = pathsig.LogSignature(depth=depth)

        assert windowed.dim() == 3
        assert windowed.size(1) == windows.size(0)

        for w in range(windows.size(0)):
            start, end = windows[w].tolist()
            expected = ref(path[:, start:end, :])
            torch.testing.assert_close(windowed[:, w, :], expected, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_single_window_equals_full(self, depth, path_dim):
        B, seq_len = 2, 30
        path = _make_path(B, seq_len, path_dim)
        windows = torch.tensor([[0, seq_len]], dtype=torch.int32, device="cuda")

        windowed = pathsig.LogSignature(depth=depth, windows=windows)(path)
        full = pathsig.LogSignature(depth=depth)(path)

        assert windowed.shape == (B, 1, full.size(1))
        torch.testing.assert_close(windowed[:, 0, :], full, atol=1e-10, rtol=1e-10)


# Window patterns

class TestWindowPatterns:
    def test_non_overlapping(self):
        depth, path_dim, B, seq_len = 3, 3, 2, 40
        win_len = 10
        path = _make_path(B, seq_len, path_dim)
        windows = _make_windows(seq_len, win_len, stride=win_len)

        windowed = pathsig.Signature(depth=depth, windows=windows)(path)
        ref = pathsig.Signature(depth=depth)

        for w in range(windows.size(0)):
            start, end = windows[w].tolist()
            torch.testing.assert_close(windowed[:, w, :], ref(path[:, start:end, :]), atol=1e-10, rtol=1e-10)

    def test_heavily_overlapping(self):
        depth, path_dim, B, seq_len = 3, 3, 2, 30
        win_len, stride = 20, 2
        path = _make_path(B, seq_len, path_dim)
        windows = _make_windows(seq_len, win_len, stride)

        windowed = pathsig.Signature(depth=depth, windows=windows)(path)
        ref = pathsig.Signature(depth=depth)

        for w in range(windows.size(0)):
            start, end = windows[w].tolist()
            torch.testing.assert_close(windowed[:, w, :], ref(path[:, start:end, :]), atol=1e-10, rtol=1e-10)

    def test_non_uniform(self):
        depth, path_dim, B = 3, 3, 2
        path = _make_path(B, 50, path_dim)
        windows = torch.tensor(
            [[0, 10], [5, 30], [10, 50], [0, 50], [20, 25]],
            dtype=torch.int32, device="cuda",
        )

        windowed = pathsig.Signature(depth=depth, windows=windows)(path)
        ref = pathsig.Signature(depth=depth)

        for w in range(windows.size(0)):
            start, end = windows[w].tolist()
            torch.testing.assert_close(windowed[:, w, :], ref(path[:, start:end, :]), atol=1e-10, rtol=1e-10)

    def test_minimum_window_length(self):
        depth, path_dim, B = 3, 3, 2
        path = _make_path(B, 20, path_dim)
        windows = torch.tensor([[0, 2], [5, 7], [18, 20]], dtype=torch.int32, device="cuda")

        windowed = pathsig.Signature(depth=depth, windows=windows)(path)
        ref = pathsig.Signature(depth=depth)

        for w in range(windows.size(0)):
            start, end = windows[w].tolist()
            torch.testing.assert_close(windowed[:, w, :], ref(path[:, start:end, :]), atol=1e-10, rtol=1e-10)


# Windowed + projected

class TestWindowedWithProjection:
    @pytest.mark.parametrize("depth", [3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_lyndon_signature(self, depth, path_dim):
        B, seq_len, win_len, stride = 2, 40, 12, 4
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        path = _make_path(B, seq_len, path_dim)
        windows = _make_windows(seq_len, win_len, stride)

        windowed = pathsig.Signature(depth=depth, projection=proj, windows=windows)(path)
        ref = pathsig.Signature(depth=depth, projection=proj)

        for w in range(windows.size(0)):
            start, end = windows[w].tolist()
            torch.testing.assert_close(windowed[:, w, :], ref(path[:, start:end, :]), atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_lyndon_logsignature(self, depth, path_dim):
        B, seq_len, win_len, stride = 2, 40, 12, 4
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        path = _make_path(B, seq_len, path_dim)
        windows = _make_windows(seq_len, win_len, stride)

        windowed = pathsig.LogSignature(depth=depth, projection=proj, windows=windows)(path)
        ref = pathsig.LogSignature(depth=depth, projection=proj)

        for w in range(windows.size(0)):
            start, end = windows[w].tolist()
            torch.testing.assert_close(windowed[:, w, :], ref(path[:, start:end, :]), atol=1e-10, rtol=1e-10)


# Autograd

class TestWindowedAutograd:
    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_signature_gradient_flow(self, depth, path_dim):
        B, seq_len, win_len, stride = 1, 30, 10, 5
        path = _make_path(B, seq_len, path_dim).requires_grad_(True)
        windows = _make_windows(seq_len, win_len, stride)

        pathsig.Signature(depth=depth, windows=windows)(path).sum().backward()

        assert path.grad is not None
        assert path.grad.shape == path.shape
        assert torch.isfinite(path.grad).all()

    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_logsignature_gradient_flow(self, depth, path_dim):
        B, seq_len, win_len, stride = 1, 30, 10, 5
        path = _make_path(B, seq_len, path_dim).requires_grad_(True)
        windows = _make_windows(seq_len, win_len, stride)

        pathsig.LogSignature(depth=depth, windows=windows)(path).sum().backward()

        assert path.grad is not None
        assert path.grad.shape == path.shape
        assert torch.isfinite(path.grad).all()

    def test_gradient_accumulation_overlapping(self):
        """With overlapping windows, grad at a shared timestep must equal the
        sum of the independent per-window gradients at that timestep."""
        depth, path_dim, B = 3, 3, 1
        seq_len, win_len, stride = 20, 10, 3
        path = _make_path(B, seq_len, path_dim).requires_grad_(True)
        windows = _make_windows(seq_len, win_len, stride)

        pathsig.Signature(depth=depth, windows=windows)(path).sum().backward()

        ref_grad = torch.zeros_like(path)
        ref = pathsig.Signature(depth=depth)
        for w in range(windows.size(0)):
            start, end = windows[w].tolist()
            sub = path[:, start:end, :].detach().requires_grad_(True)
            ref(sub).sum().backward()
            ref_grad[:, start:end, :] += sub.grad

        torch.testing.assert_close(path.grad, ref_grad, atol=1e-10, rtol=1e-10)

    def test_gradient_with_projection(self):
        depth, path_dim, B = 3, 3, 1
        seq_len, win_len, stride = 30, 10, 5
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        path = _make_path(B, seq_len, path_dim).requires_grad_(True)
        windows = _make_windows(seq_len, win_len, stride)

        pathsig.Signature(depth=depth, projection=proj, windows=windows)(path).sum().backward()

        assert path.grad is not None
        assert path.grad.shape == path.shape
        assert torch.isfinite(path.grad).all()

    @pytest.mark.parametrize("depth", [2, 3])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_signature_gradcheck(self, depth, path_dim):
        B, seq_len, win_len, stride = 1, 12, 5, 3
        path = _make_path(B, seq_len, path_dim).requires_grad_(True)
        windows = _make_windows(seq_len, win_len, stride)

        def func(p):
            return pathsig.Signature(depth=depth, windows=windows)(p)

        torch.autograd.gradcheck(func, (path,), atol=1e-5, rtol=1e-3)

    @pytest.mark.parametrize("depth", [2, 3])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_logsignature_gradcheck(self, depth, path_dim):
        B, seq_len, win_len, stride = 1, 12, 5, 3
        path = _make_path(B, seq_len, path_dim).requires_grad_(True)
        windows = _make_windows(seq_len, win_len, stride)

        def func(p):
            return pathsig.LogSignature(depth=depth, windows=windows)(p)

        torch.autograd.gradcheck(func, (path,), atol=1e-5, rtol=1e-3)

    @pytest.mark.parametrize("depth", [2, 3])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_projected_gradcheck(self, depth, path_dim):
        B, seq_len, win_len, stride = 1, 12, 5, 3
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        path = _make_path(B, seq_len, path_dim).requires_grad_(True)
        windows = _make_windows(seq_len, win_len, stride)

        def func(p):
            return pathsig.Signature(depth=depth, projection=proj, windows=windows)(p)

        torch.autograd.gradcheck(func, (path,), atol=1e-5, rtol=1e-3)


# Shapes

class TestWindowedShapes:
    def test_signature_shape(self):
        depth, path_dim, B, seq_len = 3, 4, 3, 50
        win_len, stride = 15, 5
        path = _make_path(B, seq_len, path_dim)
        windows = _make_windows(seq_len, win_len, stride)

        sig = pathsig.Signature(depth=depth, windows=windows)(path)
        expected_dim = sum(path_dim ** k for k in range(1, depth + 1))
        assert sig.shape == (B, windows.size(0), expected_dim)

    def test_logsignature_shape(self):
        depth, path_dim, B, seq_len = 3, 4, 3, 50
        win_len, stride = 15, 5
        path = _make_path(B, seq_len, path_dim)
        windows = _make_windows(seq_len, win_len, stride)

        logsig = pathsig.LogSignature(depth=depth, windows=windows)(path)
        expected_dim = sum(path_dim ** k for k in range(1, depth + 1))
        assert logsig.shape == (B, windows.size(0), expected_dim)

    def test_projected_shape(self):
        depth, path_dim, B, seq_len = 4, 3, 2, 40
        win_len, stride = 10, 5
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        path = _make_path(B, seq_len, path_dim)
        windows = _make_windows(seq_len, win_len, stride)

        sig = pathsig.Signature(depth=depth, projection=proj, windows=windows)(path)
        assert sig.shape == (B, windows.size(0), proj.sig_size)

    def test_num_windows(self):
        seq_len, win_len, stride = 100, 20, 7
        windows = _make_windows(seq_len, win_len, stride)
        expected = len(range(0, seq_len - win_len + 1, stride))
        assert windows.size(0) == expected

    def test_single_window_shape(self):
        depth, path_dim, B, seq_len = 3, 3, 2, 20
        path = _make_path(B, seq_len, path_dim)
        windows = torch.tensor([[5, 15]], dtype=torch.int32, device="cuda")

        sig = pathsig.Signature(depth=depth, windows=windows)(path)
        expected_dim = sum(path_dim ** k for k in range(1, depth + 1))
        assert sig.shape == (B, 1, expected_dim)
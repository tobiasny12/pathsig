# tests/test_projections.py
from __future__ import annotations

import itertools
import pytest
import torch

import pathsig


# Helpers

def _full_sig_indices(projection) -> torch.Tensor:
    """Indices into the full signature that correspond to the projected terms."""
    d = projection.path_dim
    depth = projection.depth
    level_sizes = projection.level_sizes
    encoded_words = projection.encoded_words

    full_level_off = 0
    encoded_off = 0
    indices: list[int] = []

    for degree in range(1, depth + 1):
        d_pow = d ** degree
        proj_size = level_sizes[degree]

        if proj_size == d_pow:
            indices.extend(range(full_level_off, full_level_off + d_pow))
        elif proj_size > 0:
            codes = encoded_words[encoded_off : encoded_off + proj_size]
            indices.extend((codes + full_level_off).tolist())
            encoded_off += proj_size

        full_level_off += d_pow

    return torch.tensor(indices, dtype=torch.int64, device="cuda")


def _all_words_up_to(d: int, depth: int):
    return [
        w
        for length in range(1, depth + 1)
        for w in itertools.product(range(d), repeat=length)
    ]


def _original_words(d_orig: int, depth: int):
    return [
        w
        for length in range(1, depth + 1)
        for w in itertools.product(range(d_orig), repeat=length)
    ]


def _make_path(B: int, M: int, d: int, dtype=torch.float64) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(B, M, d, device="cuda", dtype=dtype)


def _extend_path(path: torch.Tensor, d_extra: int) -> torch.Tensor:
    """Append d_extra random channels to the path."""
    B, M, _ = path.shape
    torch.manual_seed(999)
    extra = torch.randn(B, M, d_extra, device=path.device, dtype=path.dtype)
    return torch.cat([path, extra], dim=-1)


DEPTHS = [2, 3, 4, 5]
PATH_DIMS = [2, 3, 4, 5]
BATCH_SIZES = [1, 4]
D_EXTRA = 2


class TestLyndonProjection:
    @pytest.mark.parametrize("depth", DEPTHS)
    @pytest.mark.parametrize("path_dim", PATH_DIMS)
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_signature(self, depth, path_dim, B):
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        path = _make_path(B, 20, path_dim)

        full = pathsig.Signature(depth=depth)(path)
        projected = pathsig.Signature(depth=depth, projection=proj)(path)
        expected = full[:, _full_sig_indices(proj)]

        assert projected.shape == expected.shape
        torch.testing.assert_close(projected, expected, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", DEPTHS)
    @pytest.mark.parametrize("path_dim", PATH_DIMS)
    @pytest.mark.parametrize("B", BATCH_SIZES)
    def test_logsignature(self, depth, path_dim, B):
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        path = _make_path(B, 20, path_dim)

        full = pathsig.LogSignature(depth=depth)(path)
        projected = pathsig.LogSignature(depth=depth, projection=proj)(path)
        expected = full[:, _full_sig_indices(proj)]

        assert projected.shape == expected.shape
        torch.testing.assert_close(projected, expected, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3, 4])
    def test_signature_gradient(self, depth, path_dim):
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        indices = _full_sig_indices(proj)

        path_proj = _make_path(2, 15, path_dim).requires_grad_(True)
        pathsig.Signature(depth=depth, projection=proj)(path_proj).sum().backward()

        path_full = _make_path(2, 15, path_dim).requires_grad_(True)
        pathsig.Signature(depth=depth)(path_full)[:, indices].sum().backward()

        torch.testing.assert_close(path_proj.grad, path_full.grad, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3, 4])
    def test_logsignature_gradient(self, depth, path_dim):
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        indices = _full_sig_indices(proj)

        path_proj = _make_path(2, 15, path_dim).requires_grad_(True)
        pathsig.LogSignature(depth=depth, projection=proj)(path_proj).sum().backward()

        path_full = _make_path(2, 15, path_dim).requires_grad_(True)
        pathsig.LogSignature(depth=depth)(path_full)[:, indices].sum().backward()

        torch.testing.assert_close(path_proj.grad, path_full.grad, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [2, 3])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_signature_gradcheck(self, depth, path_dim):
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        path = _make_path(1, 8, path_dim).requires_grad_(True)

        def func(p):
            return pathsig.Signature(depth=depth, projection=proj)(p)

        torch.autograd.gradcheck(func, (path,), atol=1e-5, rtol=1e-3)

    @pytest.mark.parametrize("depth", [2, 3])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_logsignature_gradcheck(self, depth, path_dim):
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        path = _make_path(1, 8, path_dim).requires_grad_(True)

        def func(p):
            return pathsig.LogSignature(depth=depth, projection=proj)(p)

        torch.autograd.gradcheck(func, (path,), atol=1e-5, rtol=1e-3)


class TestWordsProjection:
    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_signature(self, depth, path_dim):
        d_total = path_dim + D_EXTRA
        words = list(_original_words(path_dim, depth))
        proj = pathsig.projections.words(words=words, depth=depth, path_dim=d_total)

        path_orig = _make_path(2, 20, path_dim)
        path_ext = _extend_path(path_orig, D_EXTRA)

        full = pathsig.Signature(depth=depth)(path_orig)
        projected = pathsig.Signature(depth=depth, projection=proj)(path_ext)

        assert projected.shape == full.shape
        torch.testing.assert_close(projected, full, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_logsignature(self, depth, path_dim):
        d_total = path_dim + D_EXTRA
        words = list(_original_words(path_dim, depth))
        proj = pathsig.projections.words(words=words, depth=depth, path_dim=d_total)

        path_orig = _make_path(2, 20, path_dim)
        path_ext = _extend_path(path_orig, D_EXTRA)

        full = pathsig.LogSignature(depth=depth)(path_orig)
        projected = pathsig.LogSignature(depth=depth, projection=proj)(path_ext)

        assert projected.shape == full.shape
        torch.testing.assert_close(projected, full, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_all_words_recovers_full(self, depth, path_dim):
        words = list(_all_words_up_to(path_dim, depth))
        proj = pathsig.projections.words(words=words, depth=depth, path_dim=path_dim)
        path = _make_path(2, 20, path_dim)

        full = pathsig.Signature(depth=depth)(path)
        projected = pathsig.Signature(depth=depth, projection=proj)(path)

        torch.testing.assert_close(projected, full, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_signature_gradient(self, depth, path_dim):
        d_total = path_dim + D_EXTRA
        words = list(_original_words(path_dim, depth))
        proj = pathsig.projections.words(words=words, depth=depth, path_dim=d_total)

        path_ext = _extend_path(_make_path(2, 15, path_dim), D_EXTRA).requires_grad_(True)
        pathsig.Signature(depth=depth, projection=proj)(path_ext).sum().backward()

        path_ref = _make_path(2, 15, path_dim).requires_grad_(True)
        pathsig.Signature(depth=depth)(path_ref).sum().backward()

        torch.testing.assert_close(
            path_ext.grad[:, :, :path_dim], path_ref.grad, atol=1e-10, rtol=1e-10,
        )


class TestAnisotropicProjection:
    @pytest.mark.parametrize("depth", [2, 3, 4, 5])
    @pytest.mark.parametrize("path_dim", [2, 3, 4])
    def test_signature(self, depth, path_dim):
        d_total = path_dim + D_EXTRA
        threshold = float(depth)
        weights = [1.0] * path_dim + [threshold + 1.0] * D_EXTRA

        proj = pathsig.projections.anisotropic(
            weights=weights, weight_threshold=threshold,
            depth=depth, path_dim=d_total,
        )

        path_orig = _make_path(2, 20, path_dim)
        path_ext = _extend_path(path_orig, D_EXTRA)

        full = pathsig.Signature(depth=depth)(path_orig)
        projected = pathsig.Signature(depth=depth, projection=proj)(path_ext)

        assert projected.shape == full.shape
        torch.testing.assert_close(projected, full, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [2, 3, 4, 5])
    @pytest.mark.parametrize("path_dim", [2, 3, 4])
    def test_logsignature(self, depth, path_dim):
        d_total = path_dim + D_EXTRA
        threshold = float(depth)
        weights = [1.0] * path_dim + [threshold + 1.0] * D_EXTRA

        proj = pathsig.projections.anisotropic(
            weights=weights, weight_threshold=threshold,
            depth=depth, path_dim=d_total,
        )

        path_orig = _make_path(2, 20, path_dim)
        path_ext = _extend_path(path_orig, D_EXTRA)

        full = pathsig.LogSignature(depth=depth)(path_orig)
        projected = pathsig.LogSignature(depth=depth, projection=proj)(path_ext)

        assert projected.shape == full.shape
        torch.testing.assert_close(projected, full, atol=1e-10, rtol=1e-10)

    def test_large_threshold_keeps_all(self):
        depth, path_dim = 3, 3
        weights = [0.1] * path_dim
        proj = pathsig.projections.anisotropic(
            weights=weights, weight_threshold=100.0,
            depth=depth, path_dim=path_dim,
        )
        path = _make_path(2, 20, path_dim)

        full = pathsig.Signature(depth=depth)(path)
        projected = pathsig.Signature(depth=depth, projection=proj)(path)

        torch.testing.assert_close(projected, full, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [2, 3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_signature_gradient(self, depth, path_dim):
        d_total = path_dim + D_EXTRA
        threshold = float(depth)
        weights = [1.0] * path_dim + [threshold + 1.0] * D_EXTRA
        proj = pathsig.projections.anisotropic(
            weights=weights, weight_threshold=threshold,
            depth=depth, path_dim=d_total,
        )

        path_ext = _extend_path(_make_path(2, 15, path_dim), D_EXTRA).requires_grad_(True)
        pathsig.Signature(depth=depth, projection=proj)(path_ext).sum().backward()

        path_ref = _make_path(2, 15, path_dim).requires_grad_(True)
        pathsig.Signature(depth=depth)(path_ref).sum().backward()

        torch.testing.assert_close(
            path_ext.grad[:, :, :path_dim], path_ref.grad, atol=1e-10, rtol=1e-10,
        )


class TestFullLevels:
    @pytest.mark.parametrize("depth", [3, 4, 5])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_lyndon_with_full_levels(self, depth, path_dim):
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim, full_levels=[2])
        path = _make_path(2, 20, path_dim)

        full = pathsig.Signature(depth=depth)(path)
        projected = pathsig.Signature(depth=depth, projection=proj)(path)
        expected = full[:, _full_sig_indices(proj)]

        torch.testing.assert_close(projected, expected, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("depth", [3, 4])
    @pytest.mark.parametrize("path_dim", [2, 3])
    def test_anisotropic_with_full_levels(self, depth, path_dim):
        weights = [0.5 + 0.3 * i for i in range(path_dim)]
        proj = pathsig.projections.anisotropic(
            weights=weights, weight_threshold=depth * 0.6,
            depth=depth, path_dim=path_dim, full_levels=[1],
        )
        path = _make_path(2, 20, path_dim)

        full = pathsig.Signature(depth=depth)(path)
        projected = pathsig.Signature(depth=depth, projection=proj)(path)
        expected = full[:, _full_sig_indices(proj)]

        torch.testing.assert_close(projected, expected, atol=1e-10, rtol=1e-10)


class TestProjectionProperties:
    def test_sig_size_matches_output(self):
        proj = pathsig.projections.lyndon(depth=4, path_dim=3)
        path = _make_path(2, 20, 3)
        sig = pathsig.Signature(depth=4, projection=proj)(path)
        assert sig.shape[1] == proj.sig_size

    def test_logsig_indices_shape(self):
        proj = pathsig.projections.lyndon(depth=4, path_dim=3)
        indices = proj.logsig_indices()
        logsig = pathsig.LogSignature(depth=4, projection=proj)(_make_path(2, 20, 3))
        assert indices.shape[0] == logsig.shape[1]

    @pytest.mark.parametrize("path_dim", [2, 3, 4])
    def test_lyndon_level_sizes_sum(self, path_dim):
        depth = 4
        proj = pathsig.projections.lyndon(depth=depth, path_dim=path_dim)
        assert proj.sig_size == sum(proj.level_sizes[1 : depth + 1])

    def test_words_full_level_conflict_raises(self):
        with pytest.raises(ValueError, match="full"):
            pathsig.projections.words(
                words=[(0,), (1,)],
                depth=2,
                path_dim=2,
                full_levels=[1],
            )
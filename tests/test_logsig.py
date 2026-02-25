# tests/test_logsig.py
import numpy as np
import torch
import pytest

import pathsig

try:
    import pysiglib.torch_api as pysig_torch
except ImportError:
    raise ImportError("pysiglib must be installed for tests.")

DEVICE = "cuda"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class TestConfig:
    DTYPES = [torch.float64, torch.float32]

    @staticmethod
    def get_tolerances(dtype, trunc_lvl=4, d=3):
        if dtype == torch.float64:
            tol_fwd = 1e-12 if trunc_lvl < 10 else 1e-11
            tol_rel = 1e-10 if trunc_lvl < 10 else 1e-9
        else:
            tol_fwd = 1e-5 if trunc_lvl < 10 else 1e-4
            tol_rel = 1e-3 if trunc_lvl < 10 else 1e-2
        return tol_fwd, tol_rel


_PREPARED = set()


def _pysig_prepare_logsig(d, depth):
    key = (d, depth)
    if key not in _PREPARED:
        pysig_torch.prepare_log_sig(d, depth, lead_lag=False, method=1)
        _PREPARED.add(key)


def compute_pathsig_forward_only(X, trunc_lvl, dtype):
    X = X.to(device=DEVICE, dtype=dtype).contiguous()
    d = X.size(-1)
    proj = pathsig.projections.lyndon(depth=trunc_lvl, path_dim=d)
    mod = pathsig.LogSignature(depth=trunc_lvl, projection=proj).to(DEVICE)
    mod.eval()
    with torch.inference_mode():
        return mod(X)


def compute_pathsig_logsig_and_grad(X, trunc_lvl, G, dtype):
    X = X.to(device=DEVICE, dtype=dtype).contiguous().detach().requires_grad_(True)
    d = X.size(-1)
    proj = pathsig.projections.lyndon(depth=trunc_lvl, path_dim=d)
    mod = pathsig.LogSignature(depth=trunc_lvl, projection=proj).to(DEVICE)
    Y = mod(X)
    G = G.to(device=Y.device, dtype=Y.dtype)
    (Y * G).sum().backward()
    return Y.detach(), X.grad.detach()


def compute_pysig_logsig_and_grad(X_np, trunc_lvl, G_np, dtype):
    d = X_np.shape[-1]
    _pysig_prepare_logsig(d, trunc_lvl)
    X_torch = torch.tensor(X_np, dtype=dtype, device=DEVICE, requires_grad=True)
    with torch.enable_grad():
        Y = pysig_torch.log_sig(X_torch, trunc_lvl, lead_lag=False, method=1, n_jobs=-1)
        G_torch = torch.tensor(G_np, dtype=dtype, device=Y.device)
        (Y * G_torch).sum().backward()
    return Y.detach().cpu().numpy(), X_torch.grad.detach().cpu().numpy()


def compare_implementations(B, T, d, trunc_lvl, n_tests=5, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    X = 200 * torch.rand((B, T, d), device=DEVICE, dtype=dtype) - 100

    Y_pathsig = compute_pathsig_forward_only(X, trunc_lvl, dtype)
    G = torch.randn_like(Y_pathsig)
    _, grad_pathsig = compute_pathsig_logsig_and_grad(X, trunc_lvl, G, dtype)

    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    X_np = X.detach().cpu().numpy().astype(np_dtype)
    G_np = G.detach().cpu().numpy().astype(np_dtype)
    Y_pysig_np, grad_pysig_np = compute_pysig_logsig_and_grad(X_np, trunc_lvl, G_np, dtype)

    results = {}

    Y_pathsig_np = Y_pathsig.detach().cpu().numpy()
    diff = np.abs(Y_pathsig_np - Y_pysig_np)
    results["fwd_max_abs_diff"] = float(diff.max())
    results["fwd_rel_error"] = results["fwd_max_abs_diff"] / max(1.0, float(np.abs(Y_pysig_np).max()))

    grad_pysig_torch = torch.from_numpy(grad_pysig_np).to(device=DEVICE, dtype=dtype)

    dot_errors = []
    for _ in range(n_tests):
        H = torch.randn_like(X, dtype=dtype)
        dot_pathsig = (grad_pathsig * H).sum().item()
        dot_pysig = (grad_pysig_torch * H).sum().item()
        dot_errors.append(abs(dot_pathsig - dot_pysig) / max(1.0, abs(dot_pysig)))
    results["grad_dot_max_rel_error"] = max(dot_errors)

    timestep_errors = []
    for t in [0, T // 2, T - 1] if T > 2 else [0]:
        g_pathsig_t = grad_pathsig[:, t, :].to(torch.float64)
        g_pysig_t = grad_pysig_torch[:, t, :].to(torch.float64)
        max_abs = float((g_pathsig_t - g_pysig_t).abs().max().item())
        denom = max(1.0, float(g_pysig_t.abs().max().item()))
        timestep_errors.append(max_abs / denom)
    results["grad_timestep_max_rel_error"] = max(timestep_errors)

    return results


@pytest.mark.parametrize("dtype", TestConfig.DTYPES)
@pytest.mark.parametrize("B", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
def test_batch_sizes(dtype, B):
    results = compare_implementations(B=B, T=50, d=3, trunc_lvl=4, dtype=dtype)
    tol_fwd, tol_rel = TestConfig.get_tolerances(dtype)
    assert results["fwd_rel_error"] < tol_fwd
    assert results["grad_dot_max_rel_error"] < tol_rel
    assert results["grad_timestep_max_rel_error"] < tol_rel


@pytest.mark.parametrize("dtype", TestConfig.DTYPES)
@pytest.mark.parametrize("T", [10, 20, 50, 100, 200, 500, 1000])
def test_sequence_lengths(dtype, T):
    results = compare_implementations(B=4, T=T, d=3, trunc_lvl=4, dtype=dtype)
    tol_fwd, tol_rel = TestConfig.get_tolerances(dtype)
    assert results["fwd_rel_error"] < tol_fwd
    assert results["grad_dot_max_rel_error"] < tol_rel
    assert results["grad_timestep_max_rel_error"] < tol_rel


@pytest.mark.parametrize("dtype", TestConfig.DTYPES)
@pytest.mark.parametrize("trunc_lvl", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
def test_truncation_levels(dtype, trunc_lvl):
    results = compare_implementations(B=4, T=50, d=3, trunc_lvl=trunc_lvl, dtype=dtype)
    tol_fwd, tol_rel = TestConfig.get_tolerances(dtype, trunc_lvl=trunc_lvl)
    assert results["fwd_rel_error"] < tol_fwd
    assert results["grad_dot_max_rel_error"] < tol_rel
    assert results["grad_timestep_max_rel_error"] < tol_rel


@pytest.mark.parametrize("dtype", TestConfig.DTYPES)
@pytest.mark.parametrize("d", [2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 22, 26, 30, 35, 40, 45, 50, 75, 100])
def test_path_dimensions(dtype, d):
    results = compare_implementations(B=2, T=10, d=d, trunc_lvl=3, dtype=dtype)
    tol_fwd, tol_rel = TestConfig.get_tolerances(dtype, d=d)
    assert results["fwd_rel_error"] < tol_fwd
    assert results["grad_dot_max_rel_error"] < tol_rel
    assert results["grad_timestep_max_rel_error"] < tol_rel
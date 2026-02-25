# test_sig.py
import os
import numpy as np
import torch
import pytest

os.environ["KERAS_BACKEND"] = "torch"
import keras
from pathsig import Signature

try:
    import keras_sig
except ImportError:
    raise ImportError("keras_sig must be installed for tests.")

DEVICE = "cuda"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class TestConfig:
    DTYPES = [torch.float64, torch.float32]

    @staticmethod
    def get_tolerances(dtype, trunc_lvl=4, d=3):
        if dtype == torch.float64:
            tol_fwd = 1e-14 if trunc_lvl < 10 else 1e-13
            tol_rel = 1e-12 if trunc_lvl < 10 else 1e-11
        else:
            tol_fwd = 1e-6 if trunc_lvl < 10 else 1e-5
            tol_rel = 1e-4 if trunc_lvl < 10 else 1e-3
        return tol_fwd, tol_rel


def set_keras_dtype(dtype):
    dtype_str = "float64" if dtype == torch.float64 else "float32"
    if hasattr(keras.backend, "set_floatx"):
        keras.backend.set_floatx(dtype_str)
    if hasattr(keras.config, "set_floatx"):
        keras.config.set_floatx(dtype_str)


def compute_pathsig_forward_only(X, trunc_lvl, dtype):
    X = X.to(device=DEVICE, dtype=dtype).contiguous()
    sig = Signature(depth=trunc_lvl).to(X.device)
    with torch.inference_mode():
        return sig(X)


def compute_pathsig_signature_and_grad(X, trunc_lvl, G, dtype):
    X = X.to(device=DEVICE, dtype=dtype).contiguous().detach().requires_grad_(True)
    sig = Signature(depth=trunc_lvl).to(X.device)
    S = sig(X)
    G = G.to(device=S.device, dtype=S.dtype)
    (S * G).sum().backward()
    return S.detach(), X.grad.detach()


def compute_keras_signature_and_grad(X_np, trunc_lvl, G_np, dtype):
    set_keras_dtype(dtype)
    layer = keras_sig.SigLayer(depth=trunc_lvl, stream=False, gpu_optimized=(dtype == torch.float32))
    model = keras.Sequential([layer])
    X_torch = torch.tensor(X_np, dtype=dtype, device=DEVICE, requires_grad=True)
    with torch.enable_grad():
        S = model(X_torch)
        G_torch = torch.tensor(G_np, dtype=dtype, device=S.device)
        (S * G_torch).sum().backward()
    return S.detach().cpu().numpy(), X_torch.grad.detach().cpu().numpy()


def compare_implementations(B, T, d, trunc_lvl, n_tests=5, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    X = 200 * torch.rand((B, T, d), device=DEVICE, dtype=dtype) - 100

    S_pathsig = compute_pathsig_forward_only(X, trunc_lvl, dtype)
    G = torch.randn_like(S_pathsig)
    _, grad_pathsig = compute_pathsig_signature_and_grad(X, trunc_lvl, G, dtype)

    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    X_np = X.detach().cpu().numpy().astype(np_dtype)
    G_np = G.detach().cpu().numpy().astype(np_dtype)
    S_keras_np, grad_keras_np = compute_keras_signature_and_grad(X_np, trunc_lvl, G_np, dtype)

    results = {}

    S_pathsig_np = S_pathsig.detach().cpu().numpy()
    diff = np.abs(S_pathsig_np - S_keras_np)
    results["fwd_max_abs_diff"] = float(diff.max())
    results["fwd_rel_error"] = results["fwd_max_abs_diff"] / max(1.0, float(np.abs(S_keras_np).max()))

    grad_keras_torch = torch.from_numpy(grad_keras_np).to(device=DEVICE, dtype=dtype)

    dot_errors = []
    for _ in range(n_tests):
        H = torch.randn_like(X, dtype=dtype)
        dot_pathsig = (grad_pathsig * H).sum().item()
        dot_keras = (grad_keras_torch * H).sum().item()
        dot_errors.append(abs(dot_pathsig - dot_keras) / max(1.0, abs(dot_keras)))
    results["grad_dot_max_rel_error"] = max(dot_errors)

    timestep_errors = []
    for t in [0, T // 2, T - 1] if T > 2 else [0]:
        g_pathsig_t = grad_pathsig[:, t, :].to(torch.float64)
        g_keras_t = grad_keras_torch[:, t, :].to(torch.float64)
        max_abs = float((g_pathsig_t - g_keras_t).abs().max().item())
        denom = max(1.0, float(g_keras_t.abs().max().item()))
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
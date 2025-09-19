import os
import numpy as np
import torch
import pytest

# Set up environment
os.environ["KERAS_BACKEND"] = "torch"
import keras
from pathsig import signature
try:
    import keras_sig
except ImportError:
    raise ImportError("keras_sig must be installed for tests.")

# ---- Global Configuration ----
torch.set_default_dtype(torch.float64)
DEVICE = "cuda"

# Disable TF32 for more consistent precision
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class TestConfig:
    """Test configuration and tolerance settings."""

    DTYPES_PRECISIONS = [
        (torch.float64, "kahan"),
        (torch.float64, "extended"),
        (torch.float32, "kahan"),
    ]

    @staticmethod
    def get_tolerances(dtype, trunc_lvl=4, d=3):
        """Get test tolerances based on dtype and problem size."""
        if dtype == torch.float64:
            tol_fwd = 1e-14 if trunc_lvl < 10 else 1e-13
            tol_rel = 1e-12 if trunc_lvl < 10 else 1e-11
        else:  # float32
            tol_fwd = 1e-6 if trunc_lvl < 10 else 1e-5
            tol_rel = 1e-4 if trunc_lvl < 10 else 1e-3
        return tol_fwd, tol_rel


def set_keras_dtype(dtype: torch.dtype):
    """Configure Keras to use the specified float type."""
    dtype_str = "float64" if dtype == torch.float64 else "float32"
    try:
        if hasattr(keras.backend, "set_floatx"):
            keras.backend.set_floatx(dtype_str)
        if hasattr(keras.config, "set_floatx"):
            keras.config.set_floatx(dtype_str)
    except Exception:
        pass


def compute_pathsig_signature_and_grad(X, trunc_lvl, G):
    """
    Compute signature and gradients using pathsig.

    Args:
        X: Input path tensor (B, T, d)
        trunc_lvl: Truncation level
        G: Gradient tensor for backward pass

    Returns:
        Tuple of (signature, gradient w.r.t. X)
    """
    # Forward pass using autograd - NO extended precision for backward pass
    X.requires_grad_(True)
    S = signature(X, trunc_lvl)

    # Backward pass
    loss = torch.sum(S * G)
    loss.backward()

    return S.detach(), X.grad.detach()


def compute_pathsig_forward_only(X, trunc_lvl, extended_precision=False):
    """
    Compute signature using pathsig (forward only, no gradients).

    Args:
        X: Input path tensor (B, T, d)
        trunc_lvl: Truncation level
        extended_precision: Use extended precision if True

    Returns:
        Signature tensor
    """
    with torch.inference_mode():
        return signature(X, trunc_lvl, extended_precision=extended_precision)


def compute_keras_signature_and_grad(X_np, trunc_lvl, G_np, dtype):
    """
    Compute signature and gradients using keras_sig.

    Args:
        X_np: Input path numpy array
        trunc_lvl: Truncation level
        G_np: Gradient numpy array
        dtype: Target torch dtype

    Returns:
        Tuple of (signature, gradient) as numpy arrays
    """

    set_keras_dtype(dtype)

    # Build model
    dtype_str = "float64" if dtype == torch.float64 else "float32"
    layer = keras_sig.SigLayer(depth=trunc_lvl, stream=False, gpu_optimized=True)
    model = keras.Sequential([layer])

    # Compute with gradient tracking
    X_torch = torch.tensor(X_np, dtype=dtype, device=DEVICE, requires_grad=True)

    with torch.enable_grad():
        S = model(X_torch)
        G_torch = torch.tensor(G_np, dtype=dtype, device=S.device)
        loss = torch.sum(S * G_torch)
        loss.backward()

    return S.detach().cpu().numpy(), X_torch.grad.detach().cpu().numpy()


def compare_implementations(B, T, d, trunc_lvl, n_tests=5, seed=0, dtype=torch.float64, precision="kahan"):
    """
    Compare pathsig and keras_sig implementations.

    Args:
        B: Batch size
        T: Sequence length
        d: Path dimension
        trunc_lvl: Truncation level
        n_tests: Number of gradient tests
        seed: Random seed
        dtype: Data type
        precision: Precision mode ("kahan" or "extended")

    Returns:
        Dictionary with comparison results
    """
    if dtype == torch.float32 and precision == "extended":
        raise ValueError("Extended precision not supported for float32")

    use_extended_precision = (precision == "extended")

    # Generate test data
    torch.manual_seed(seed)
    X = 200 * torch.rand((B, T, d), device=DEVICE, dtype=dtype) - 100

    # Compute forward pass with pathsig (can use extended precision here)
    S_pathsig = compute_pathsig_forward_only(X, trunc_lvl, extended_precision=use_extended_precision)

    G = torch.randn_like(S_pathsig, dtype=dtype, device=S_pathsig.device)

    # Compute gradients with pathsig
    X_grad = X.clone().detach().requires_grad_(True)
    S_grad, grad_pathsig = compute_pathsig_signature_and_grad(X_grad, trunc_lvl, G)

    # Compute with keras_sig
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    X_np = X.cpu().numpy().astype(np_dtype)
    G_np = G.cpu().numpy().astype(np_dtype)

    S_keras_np, grad_keras_np = compute_keras_signature_and_grad(X_np, trunc_lvl, G_np, dtype)

    # Compare results
    results = {"device": DEVICE, "dtype": str(dtype), "precision": precision}

    # Forward pass comparison
    S_pathsig_np = S_pathsig.cpu().numpy()
    diff = np.abs(S_pathsig_np - S_keras_np)
    results["fwd_max_abs_diff"] = float(diff.max())
    results["fwd_rel_error"] = results["fwd_max_abs_diff"] / max(1.0, float(np.abs(S_keras_np).max()))

    # Gradient comparison - dot product tests
    grad_keras_torch = torch.from_numpy(grad_keras_np).to(X.device, dtype=dtype)
    dot_errors = []

    for _ in range(n_tests):
        H = torch.randn_like(X, dtype=dtype)
        dot_pathsig = (grad_pathsig * H).sum().item()
        dot_keras = (grad_keras_torch * H).sum().item()
        rel_err = abs(dot_pathsig - dot_keras) / max(1.0, abs(dot_keras))
        dot_errors.append(rel_err)

    results["grad_dot_max_rel_error"] = max(dot_errors) if dot_errors else 0

    # Pointwise gradient comparison at selected timesteps
    timestep_errors = []
    test_timesteps = [0, T//2, T-1] if T > 2 else [0]

    for t in test_timesteps:
        g_pathsig_t = grad_pathsig[:, t, :].to(torch.float64)
        g_keras_t = grad_keras_torch[:, t, :].to(torch.float64)
        diff = (g_pathsig_t - g_keras_t).abs()
        max_abs = float(diff.max().item())
        denom = max(1.0, float(g_keras_t.abs().max().item()))
        timestep_errors.append(max_abs / denom)

    results["grad_timestep_max_rel_error"] = max(timestep_errors) if timestep_errors else 0

    return results


# Pytest Test Cases

@pytest.mark.parametrize("dtype, precision", TestConfig.DTYPES_PRECISIONS)
@pytest.mark.parametrize("B", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
def test_batch_sizes(dtype, precision, B):
    """Test various batch sizes."""
    results = compare_implementations(
        B=B, T=50, d=3, trunc_lvl=4,
        dtype=dtype, precision=precision
    )

    tol_fwd, tol_rel = TestConfig.get_tolerances(dtype)
    assert results["fwd_rel_error"] < tol_fwd
    assert results["grad_dot_max_rel_error"] < tol_rel
    assert results["grad_timestep_max_rel_error"] < tol_rel


@pytest.mark.parametrize("dtype, precision", TestConfig.DTYPES_PRECISIONS)
@pytest.mark.parametrize("T", [10, 20, 50, 100, 200, 500, 1000])
def test_sequence_lengths(dtype, precision, T):
    """Test various sequence lengths."""
    results = compare_implementations(
        B=4, T=T, d=3, trunc_lvl=4,
        dtype=dtype, precision=precision
    )

    tol_fwd, tol_rel = TestConfig.get_tolerances(dtype)
    assert results["fwd_rel_error"] < tol_fwd
    assert results["grad_dot_max_rel_error"] < tol_rel
    assert results["grad_timestep_max_rel_error"] < tol_rel


@pytest.mark.parametrize("dtype, precision", TestConfig.DTYPES_PRECISIONS)
@pytest.mark.parametrize("trunc_lvl", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
def test_truncation_levels(dtype, precision, trunc_lvl):
    """Test various truncation levels."""
    results = compare_implementations(
        B=4, T=50, d=3, trunc_lvl=trunc_lvl,
        dtype=dtype, precision=precision
    )

    tol_fwd, tol_rel = TestConfig.get_tolerances(dtype, trunc_lvl=trunc_lvl)
    assert results["fwd_rel_error"] < tol_fwd
    assert results["grad_dot_max_rel_error"] < tol_rel
    assert results["grad_timestep_max_rel_error"] < tol_rel


@pytest.mark.parametrize("dtype, precision", TestConfig.DTYPES_PRECISIONS)
@pytest.mark.parametrize("d", [2, 3, 4, 5, 7, 10, 12, 15, 18, 20, 22, 26, 30, 35, 40, 45, 50, 75, 100])
def test_path_dimensions(dtype, precision, d):
    """Test various path dimensions."""
    results = compare_implementations(
        B=2, T=10, d=d, trunc_lvl=3,
        dtype=dtype, precision=precision
    )

    tol_fwd, tol_rel = TestConfig.get_tolerances(dtype, d=d)
    assert results["fwd_rel_error"] < tol_fwd
    assert results["grad_dot_max_rel_error"] < tol_rel
    assert results["grad_timestep_max_rel_error"] < tol_rel
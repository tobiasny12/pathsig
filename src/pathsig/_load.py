# pathsig/_load.py
from __future__ import annotations

import threading
from importlib.resources import as_file, files

import torch

_loaded = False
_lock = threading.Lock()

def ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return

    if torch.version.cuda is None:
        raise RuntimeError(
            "pathsig is CUDA-only, but the installed PyTorch is CPU-only "
        )

    with _lock:
        if _loaded:
            return

        pkg = files("pathsig")

        candidates = [
            pkg / "libpathsig_ops.so",     # Linux
            pkg / "libpathsig_ops.dylib",  # macOS
            pkg / "pathsig_ops.dll",       # Windows
            pkg / "libpathsig_ops.dll",    # Windows if PREFIX "lib"
        ]

        for p in candidates:
            if p.is_file():
                with as_file(p) as fp:
                    torch.ops.load_library(str(fp))
                _loaded = True
                return

        raise RuntimeError(
            f"Could not find native ops library inside {pkg}. "
            "Expected one of: " + ", ".join(c.name for c in candidates)
        )

# pathsig/__init__.py
from ._load import ensure_loaded as _ensure_loaded
_ensure_loaded()

from .modules import Signature, LogSignature
from .projections import projections

__all__ = ["Signature", "LogSignature", "projections"]


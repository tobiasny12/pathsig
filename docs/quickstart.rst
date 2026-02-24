Quickstart
==========

``pathsig`` provides differentiable :doc:`signature <api/signature>` and
:doc:`log-signature <api/logsignature>` computations for discrete paths.

Input tensor shape
------------------

Paths are provided as a tensor ``x`` of shape ``(B, T, d)``:

- ``B``: batch size
- ``T``: sequence length
- ``d``: path dimension

``x`` must be a CUDA tensor with dtype ``float32`` or ``float64``.

Basic usage
-----------

.. code-block:: python

   import torch
   import pathsig

   x = torch.randn(32, 128, 8, device="cuda", dtype=torch.float32)
   sig = pathsig.Signature(depth=4)
   y = sig(x)  # (B, D)

Windowed signatures
-------------------

Multiple signatures can be computed on subintervals of the same path by passing ``windows`` as an integer tensor
of shape ``(W, 2)`` with rows ``(start, end)``.
Windows are interpreted as half-open intervals ``[start, end)`` that may overlap and may be provided in any order.

.. code-block:: python

   windows = torch.tensor(
    [[0, 64], [32, 96], [64, 128]],
    device="cuda",
   )

   sig = pathsig.Signature(depth=4, windows=windows)
   y = sig(x)  # (B, W, D)

Projections
-----------

By default, :class:`pathsig.Signature` returns all signature coordinates up to the specified `depth`.
Projections let you compute only selected coordinates, which can give a more compact or more expressive feature set.

The example below uses an explicit word-based projection. For other projection types, see :doc:`api/projections`.

.. code-block:: python

   proj = pathsig.projections.words(
       words=[(0,), (1,), (2, 3), (3, 3, 0)],
       depth=4,
       path_dim=4,
   )

   x = torch.randn(32, 128, 4, device="cuda", dtype=torch.float32)
   sig = pathsig.Signature(depth=4, projection=proj)
   y = sig(x) # (B, proj.sig_size)
pathsig.LogSignature
====================

.. py:class:: pathsig.LogSignature(depth, projection=None, windows=None)

   Computes the log-signature of a path.

   :param int depth: Truncation level.
   :param projection: *(optional)* Projection controlling which log-signature coordinates are computed and returned.
                      See :doc:`projections` for supported types.
   :type projection: object | None
   :param torch.Tensor windows: *(optional)* Integer tensor of shape :math:`(W,2)` specifying windows :math:`[start,end)`.
   :type windows: torch.Tensor | None

   .. py:method:: forward(x)

      :param torch.Tensor x: CUDA tensor of shape :math:`(B,T,d)` with dtype ``float32`` or ``float64``.
      :returns: Tensor of shape :math:`(B,D)` if ``windows`` is ``None``, and :math:`(B,W,D)` otherwise.
      :rtype: torch.Tensor

      Here :math:`B` is the batch size, :math:`T` the sequence length, :math:`d` the path dimension, :math:`W` the number of windows,
      and :math:`D` the output dimension. If no projection is provided then
      :math:`D=\sum_{k=1}^{\mathrm{depth}} d^k`. With a projection, :math:`D=\texttt{projection.sig\_size}`.

   .. rubric:: Notes
      :class: pathsig-small-rubric

   For a compact coordinate representation of the log-signature, a projection should be specified. A canonical choice is the
   :doc:`Lyndon projection <projections_lyndon>`.

   Autograd is supported for ``float32`` and ``float64`` in both windowed and non-windowed modes, with or without a projection.

   **Examples**

   .. code-block:: python

      import torch
      import pathsig

      x = torch.randn(32, 128, 8, device="cuda", dtype=torch.float32) # (B, T, d)

      # Log-signature with a Lyndon projection
      proj = pathsig.projections.lyndon(depth=4, path_dim=8)
      logsig = pathsig.LogSignature(depth=4, projection=proj)
      y = logsig(x)  # (B, proj.sig_size)

      # Windowed log-signature with a Lyndon projection
      windows = torch.tensor([[0, 32], [32, 64]], device="cuda") # (W, 2)
      logsig = pathsig.LogSignature(depth=4, projection=proj, windows=windows)
      y = logsig(x)  # (B, W, proj.sig_size)

      # Full truncated log-signature
      logsig = pathsig.LogSignature(depth=4)
      y = logsig(x)  # (B, D)
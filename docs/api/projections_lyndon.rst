pathsig.projections.lyndon
==========================

.. py:function:: pathsig.projections.lyndon(depth, path_dim, full_levels=())

   Projection selecting Lyndon-word coordinates up to the specified depth.

   :param int depth: Maximum word length (signature level) to include.
   :param int path_dim: Path dimension (alphabet size).
   :param full_levels: *(optional)* Levels in :math:`\{1,2,\ldots,\mathrm{depth}\}` that should be included in full in addition to
                       the Lyndon coordinates.
   :type full_levels: iterable of int

   :returns:  Projection object to be used with :class:`pathsig.Signature` and :class:`pathsig.LogSignature`.
   :rtype: Projection

   .. rubric:: Notes
      :class: pathsig-small-rubric

   Lyndon words form a canonical generating set for the truncated tensor algebra up to level ``depth``, so every coordinate of the truncated
   signature can be expressed as a polynomial in the Lyndon coordinates.
   Moreover, Lyndon words index a basis of the free Lie algebra, so the truncated log-signature is linear in these coordinates.
   For this reason, the Lyndon projection is typically the default choice for log-signature computations.

   .. rubric:: Examples
      :class: pathsig-small-rubric

   .. code-block:: python

      import pathsig

      proj = pathsig.projections.lyndon(depth=5, path_dim=8)
      sig_size = proj.sig_size  # output feature dimension

      sig = pathsig.Signature(depth=5, projection=proj)
      logsig = pathsig.LogSignature(depth=5, projection=proj)
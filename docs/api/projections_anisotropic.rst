pathsig.projections.anisotropic
===============================

.. py:function:: pathsig.projections.anisotropic(weights, weight_threshold, depth, path_dim, full_levels=())

   Projection selecting coordinates via an anisotropic truncation rule. It selects words :math:`(i_1,\ldots,i_n)` satisfying

   .. math::

      \texttt{weights}[i_1] + \cdots + \texttt{weights}[i_n] \le \texttt{weight\_threshold},

   together with all words at levels listed in ``full_levels``.

   :param weights: Per-letter weights of length ``path_dim``.
   :type weights: iterable of float or int
   :param weight_threshold: Non-strict threshold used by the selection rule.
   :type weight_threshold: float or int
   :param int depth: Maximum word length (signature level) to include.
   :param int path_dim: Path dimension (alphabet size).
   :param full_levels: *(optional)* Levels in :math:`\{1,2,\ldots,\mathrm{depth}\}` that should be included in full in addition to
                       the anisotropically selected set.
   :type full_levels: iterable of int

   :returns:  Projection object to be used with :class:`pathsig.Signature` and :class:`pathsig.LogSignature`.
   :rtype: Projection

   .. rubric:: Notes
      :class: pathsig-small-rubric

   It is preferable to specify ``weights`` and ``weight_threshold`` as integers. Floating-point inputs can make the cutoff
   ``weights[i_1] + ... + weights[i_n] <= weight_threshold`` unstable near the boundary due to inexact representation and accumulated rounding
   error. Integer arithmetic, by contrast, yields a deterministic, platform-independent truncation rule.

   .. rubric:: Examples
      :class: pathsig-small-rubric

   .. code-block:: python

      import pathsig

      proj = pathsig.projections.anisotropic(
          weights=[1, 2, 3, 4],
          weight_threshold=6,
          depth=5,
          path_dim=4,
      )
      sig_size = proj.sig_size  # output feature dimension
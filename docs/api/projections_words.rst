pathsig.projections.words
========================

.. py:function:: pathsig.projections.words(words, depth, path_dim, full_levels=())

   Projection selecting an explicit word set.

   :param words: Nested iterable of words.
   :type words: iterable of iterable of int
   :param int depth: Maximum word length (signature level) to include.
   :param int path_dim: Path dimension (alphabet size).
   :param full_levels: *(optional)* Levels in :math:`\{1,2,\ldots,\mathrm{depth}\}` to include in full in addition to
                       the explicitly provided words.
   :type full_levels: iterable of int

   :returns: Projection object to be used with :class:`pathsig.Signature` and :class:`pathsig.LogSignature`.
   :rtype: Projection

   .. rubric:: Notes
      :class: pathsig-small-rubric

   Words are specified as integer sequences over the 0-based alphabet :math:`\{0,\ldots,\texttt{path\_dim}-1\}`.
   Only words of length at most ``depth`` are included. The input iterable must be ordered by
   non-decreasing word length.

   .. rubric:: Examples
      :class: pathsig-small-rubric

   .. code-block:: python

      import pathsig

      proj = pathsig.projections.words(
          words=[(2, 3), (3, 2), (2, 3, 3), (3, 2, 3), (2, 3, 2)],
          depth=3,
          path_dim=4,
          full_levels=(1,),
      )
      sig_size = proj.sig_size  # output feature dimension
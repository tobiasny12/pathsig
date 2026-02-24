pathsig.projections
===================

.. py:module:: pathsig.projections

``pathsig.projections`` is a namespace for constructing projection objects.
A projection restricts which signature coordinates are computed and returned.

Conventions
---------------------------

- For a path in :math:`\mathbb{R}^d`, valid letters are integers in ``{0, 1, ..., d-1}``.
- A *word* is a finite sequence of letters, e.g. ``(0,)``, ``(2, 1)``, ``(3, 3, 0)``.
- Projections use level-major ordering, with lexicographic ordering within each level. The only exception is explicit word projections, where non-full levels follow the word order provided.

Projections
-----------

.. toctree::
   :maxdepth: 1

   projections_words
   projections_anisotropic
   projections_lyndon

# pathsig/projections.py
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch


class Projection:
    """Base class for (log) signature projections."""

    @property
    def encoded_words(self) -> torch.Tensor:
        """Base-d encoded word indices (only non-full levels)."""
        return self._encoded_words_cuda

    @property
    def level_sizes(self) -> List[int]:
        """Number of selected words per degree."""
        return self._level_sizes

    @property
    def sig_size(self) -> int:
        """Dimension of the projected signature."""
        return sum(self._level_sizes[1 : self.depth + 1])

    def logsig_indices(self) -> torch.Tensor:
        """Indices to select from the signature (truncated signature up to depth-1 and word projection at depth) for the log-signature projection."""
        level_off = 0
        encoded_off = 0
        d_pow = 1
        indices: List[int] = []

        for degree in range(1, self.depth):
            d_pow *= self.path_dim
            level_size = self.level_sizes[degree]

            if level_size == d_pow:  # full level
                indices.extend(range(level_off, level_off + d_pow))
            else:
                codes = self.encoded_words[encoded_off : encoded_off + level_size]
                indices.extend((codes + level_off).tolist())
                encoded_off += level_size

            level_off += d_pow

        # top level
        top_size = self.level_sizes[self.depth]
        indices.extend(range(level_off, level_off + top_size))

        return torch.tensor(indices, device=self.encoded_words.device, dtype=torch.int64)

    def _validate_inputs(self, d, depth, full_levels, max_depth: int = 12, word_bits: int = 64) -> None:
        if d <= 1:
            raise ValueError(f"path_dim must be >= 2 (got {d}).")
        if not (1 <= depth <= max_depth):
            raise ValueError(f"depth must satisfy 1 <= depth <= {max_depth} (got {depth}).")

        if full_levels and (min(full_levels) < 1 or max(full_levels) > depth):
            raise ValueError("full_levels entries must satisfy 1 <= level <= depth.")

        n_eff = 2 if depth == 1 else depth  # use length-2 constraint for depth=1

        # packing constraint
        b_min = (d - 1).bit_length()
        b_max = word_bits // n_eff
        d_max_pack = 1 << b_max
        if b_min > b_max:
            raise ValueError(f"Maximum path_dim for depth={depth} is {d_max_pack}, but got path_dim={d}.")

        # base-d int64 overflow constraints that supersedes packing constraints
        if (n_eff & (n_eff - 1)) == 0:
            d_max_base_int64 = {2: 3037000499, 4: 55108, 8: 234}[n_eff]
            if d > d_max_base_int64:
                raise ValueError(f"Maximum path_dim for depth={depth} is {d_max_base_int64}, but got path_dim={d}.")

class WordsProjection(Projection):
    """Projection onto explicit list of words."""
    def __init__(
            self,
            words: Iterable[Iterable[int]],
            depth: int,
            path_dim: int,
            full_levels: Iterable[int] = (),
    ) -> None:
        self.path_dim = int(path_dim)
        self.depth = int(depth)
        self._full_levels = {int(n) for n in full_levels}
        self._validate_inputs(self.path_dim, self.depth, self._full_levels)

        self._encoded_words_cuda, self._level_sizes = self._encode_words(words)

    def _encode_words(self, words: Iterable[Iterable[int]]) -> Tuple[torch.Tensor, List[int]]:
        d = self.path_dim
        encoded_words = []
        level_sizes = [0] * (self.depth + 1) # index 0 unused

        prev_len = 0
        for word in words:
            code = 0
            n = 0
            for letter in word:
                n += 1
                if not (0 <= letter < d):
                    raise ValueError("Invalid letter encountered in a word, ensure all letters satisfy 0 <= letter < d.")
                code = code * d + letter
            if (n < prev_len):
                raise ValueError("Words must be non-decreasing in length.")
            if n > self.depth:
                break
            prev_len = n
            encoded_words.append(code)
            level_sizes[n] += 1

        # Drop encoded words at levels that are fully enumerated.
        # Full levels are handled as dense 0..d**n-1, so we do not store per-word codes for them.
        level_off = 0
        for n in range(1, self.depth+1):
            lvl_sz = level_sizes[n]
            if lvl_sz == d**n:
                del encoded_words[level_off : level_off + lvl_sz]
            else:
                level_off += lvl_sz

        for n in self._full_levels:
            if level_sizes[n] > 0:
                raise ValueError(f"Degree {n} is marked as full, but explicit words were also provided for that level.")
            level_sizes[n] = d**n

        encoded_words = torch.tensor(encoded_words, dtype=torch.int64, device="cuda")
        return encoded_words, level_sizes

class AnisotropicProjection(Projection):
    """Anisotropic truncation."""
    def __init__(
            self,
            weights: Union[Sequence[float], torch.Tensor],
            weight_threshold: float,
            depth: int,
            path_dim: int,
            full_levels: Iterable[int] = (),
    ) -> None:
        self.path_dim = int(path_dim)
        self.depth = int(depth)

        self._full_levels = {int(n) for n in full_levels}
        self._validate_inputs(self.path_dim, self.depth, self._full_levels)

        self.weight_threshold = float(weight_threshold)
        if self.weight_threshold <= 0.0:
            raise ValueError("weight_threshold must be > 0.")

        # Materialise weights on CPU as a Python list of floats
        if isinstance(weights, torch.Tensor):
            w_list = weights.detach().to(device="cpu").flatten().tolist()
        else:
            w_list = list(weights)

        if len(w_list) != self.path_dim:
            raise ValueError(
                f"weights must have length == path_dim ({self.path_dim}), "
                f"got {len(w_list)}."
            )

        self.weights = [float(x) for x in w_list]
        if any(w < 0.0 for w in self.weights):
            raise ValueError("weights must be non-negative.")

        self._encoded_words_cuda, self._level_sizes = self._encode_words()

    def _encode_words(self) -> Tuple[torch.Tensor, List[int]]:
        weights = self.weights
        d = self.path_dim
        thr = self.weight_threshold

        level_sizes = [0]*(self.depth+1)
        level_offsets = [0]*(self.depth+1)
        word_weights = []
        encoded_words = []

        for letter in range(d):
            wt = weights[letter]
            if wt <= thr:
                encoded_words.append(letter)
                word_weights.append(wt)
        level_sizes[1] = len(encoded_words)

        if level_sizes[1] == 0:
            raise ValueError("At least one weight must be <= weight_threshold.")

        for degree in range(2, self.depth+1):
            prev_sz = level_sizes[degree - 1]
            prev_off = level_offsets[degree-1]
            current_sz = 0

            for i in range(prev_sz):
                idx = prev_off + i
                prev_wt = word_weights[idx]
                prev_code = encoded_words[idx]
                for letter in range(d):
                    new_wt = prev_wt + weights[letter]
                    if new_wt <= thr:
                        encoded_words.append(prev_code * d + letter)
                        word_weights.append(new_wt)
                        current_sz += 1

            level_sizes[degree] = current_sz
            level_offsets[degree] = prev_off + prev_sz
            if current_sz == 0:
                break

        level_off = 0
        for degree in range(1, self.depth+1):
            lvl_sz = level_sizes[degree]
            if lvl_sz == 0 and degree in self._full_levels:
                level_sizes[degree] = d**degree
            elif degree in self._full_levels or lvl_sz == d**degree:
                del encoded_words[level_off : level_off + lvl_sz]
                level_sizes[degree] = d**degree
            else:
                level_off += lvl_sz

        encoded_words = torch.tensor(encoded_words, dtype=torch.int64, device="cuda")
        return encoded_words, level_sizes

class LyndonProjection(Projection):
    """Lyndon-word projection."""

    def __init__(self, depth: int, path_dim: int, full_levels: Iterable[int] = ()) -> None:
        self.depth = int(depth)
        self.path_dim = int(path_dim)

        self._full_levels = {int(n) for n in full_levels}
        self._validate_inputs(self.path_dim, self.depth, self._full_levels)

        self._encoded_words_cuda, self._level_sizes = self._encode_words()

    def _encode_words(self) -> Tuple[torch.Tensor, List[int]]:
        d = self.path_dim
        level_sizes = [0]*(self.depth+1)
        level_sizes[1] = d # only full level

        all_encoded_words = []
        for degree in range(2, self.depth+1):
            if degree in self._full_levels:
                level_sizes[degree] = d**degree
            else:
                encoded_words = self._lyndon_words_at_len(d, degree)
                level_sizes[degree] = len(encoded_words)
                all_encoded_words.extend(encoded_words)
        all_encoded_words = torch.tensor(all_encoded_words, dtype=torch.int64, device="cuda")
        return all_encoded_words, level_sizes

    def _lyndon_words_at_len(self, d: int, n: int) -> list[int]:
        """Base-d integer encodings of Lyndon words of length n (Duval's algorithm)."""
        encoded_words = []
        w = [-1]
        while w:
            w[-1] += 1
            m = len(w)

            if m == n:
                code = 0
                for letter in w:
                    code = code * d + letter
                encoded_words.append(code)

            while len(w) < n:
                w.append(w[len(w) - m])
            while w and w[-1] == d - 1:
                w.pop()

        return encoded_words

class projections:
    @staticmethod
    def words(
            words: Iterable[Iterable[int]],
            depth: int,
            path_dim: int,
            full_levels: Iterable[int] = (),
    ) -> WordsProjection:
        return WordsProjection(words, depth, path_dim, full_levels)

    @staticmethod
    def anisotropic(
            weights: Union[Sequence[float], torch.Tensor],
            weight_threshold: float,
            depth: int,
            path_dim: int,
            full_levels: Iterable[int] = (),
    ) -> AnisotropicProjection:
        return AnisotropicProjection(weights, weight_threshold, depth, path_dim, full_levels)

    @staticmethod
    def lyndon(
            depth: int, path_dim: int, full_levels: Iterable[int] = ()
    ) -> LyndonProjection:
        return LyndonProjection(depth, path_dim, full_levels)
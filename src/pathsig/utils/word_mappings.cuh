// word_mappings.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>


namespace pathsig {
    namespace word_mappings {
        /// Unpack a letter from the packed uint64
        template<unsigned bits>
        static __forceinline__ __device__ unsigned unpack(uint64_t packed, unsigned index) {
            constexpr uint64_t mask = (1ULL << bits) - 1;
            return ((packed >> (bits * index)) & mask);
        }

        /// Pack letters of a word
        template<unsigned bits>
        __device__ __noinline__ uint64_t packLetters(uint64_t word, unsigned degree, unsigned d) {
            constexpr uint64_t mask = (1ULL << bits) - 1;
            uint64_t packed = 0;
            uint64_t current_word = word;
            for (int pos = degree - 1; pos >= 0; --pos) {
                unsigned letter = current_word % d;
                packed |= (letter & mask) << (bits * pos);
                current_word /= d;
            }
            return packed;
        }

        /// Pack first occurrence map
        template<unsigned bits>
        __device__ __noinline__ uint64_t packFirstOcc(uint64_t letter_map, unsigned degree) {
            constexpr uint64_t mask = (1ULL << bits) - 1;
            uint64_t first_occ_map = 0;

            for (unsigned pos = 0; pos < degree; ++pos) {
                unsigned current_letter = unpack<bits>(letter_map, pos);
                unsigned first_occ = pos;

                for (unsigned prior_pos = 0; prior_pos < pos; ++prior_pos) {
                    if (current_letter == unpack<bits>(letter_map, prior_pos)) {
                        first_occ = prior_pos;
                        break;
                    }
                }
                first_occ_map |= (first_occ & mask) << (bits * pos);
            }
            return first_occ_map;
        }

        // Pack letters from the back of a word
        template<unsigned bits>
        __device__ uint64_t packLettersFromBack(uint64_t word, unsigned degree, int d) {
            constexpr uint64_t mask = (1ULL << bits) - 1;
            uint64_t packed = 0;
            uint64_t current_word = word;
            for (unsigned i = 0; i < degree; ++i) {
                uint64_t letter = current_word % d;
                packed |= (letter & mask) << (bits * i);
                current_word /= d;
            }
            return packed;
        }

        template<unsigned bits>
        __device__ __noinline__ uint64_t packSortedPos(uint64_t letter_map, unsigned degree, unsigned d) {
            constexpr uint64_t mask = (1ULL << bits) - 1;
            uint64_t sorted_pos_map = 0;
            unsigned packed_idx = 0;

            for (unsigned letter = 0; letter < d; ++letter) {
                for (unsigned pos = 0; pos < degree; ++pos) {
                    unsigned current_letter = unpack<bits>(letter_map, pos);

                    // Since position start at 0, first match is the first occurrence
                    if (current_letter == letter) {
                        sorted_pos_map |= (pos & mask) << (bits * packed_idx);
                        ++packed_idx;
                        break;
                    }
                }
            }
            return sorted_pos_map;
        }


        /// Pack rightmost occurrence mask
        template<unsigned bits>
        __device__ __noinline__ unsigned packRightmost(
            uint64_t letters_from_back, unsigned degree) {
            unsigned rightmost_mask = 0;
            for (unsigned pos = 0; pos < degree; ++pos) {
                unsigned current_letter = unpack<bits>(letters_from_back, pos);
                bool is_rightmost = true;
                for (unsigned later_pos = 0; later_pos < pos; ++later_pos) {
                    unsigned later_letter = unpack<bits>(letters_from_back, later_pos);
                    if (later_letter == current_letter) {
                        is_rightmost = false;
                        break;
                    }
                }
                if (is_rightmost) {
                    rightmost_mask |= (1U << pos);
                }
            }
            return rightmost_mask;
        }
    } // namespace word_mappings
} // namespace pathsig
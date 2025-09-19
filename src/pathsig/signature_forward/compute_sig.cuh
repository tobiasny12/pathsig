// compute_sig.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>


namespace pathsig {
namespace compute_sig {
    /**
     * @brief CUDA kernel to compute the truncated signature over the interval of the path [0, T].
     *
     * Each thread computes the signature value for one word in the tensor algebra.
     * Blocks handle groups of words with the same fixed prefix of length fixed_len.
     * The signature is updated incrementally using Chen's relation.
     *
     * @tparam bits Number of bits used for packing letters (5, 8, or 10)
     * @param path_increments Path increment data (batch_size x num_time_steps x path_dim)
     * @param signature Output signature tensor (batch_size x total_sig_size)
     * @param d Path dimension
     * @param trunc_lvl Truncation level
     * @param fixed_len Fixed prefix length
     * @param num_time_steps Number of time steps to compute signature over
     * @param total_sig_size Total size of the signature
     * @param d_powers Array of precomputed powers: d_powers[j] = d^j
     * @param level_offsets Array of precomputed level offsets: [0, 0, d, d^2 + d, ...]
     * @param degrees Array with the degree of each word in the decomposition
     * @param varying_words Array with varying words that extend the fixed word
     * @param prefix_maps Array with the mappings of each word's prefixes
     */
    template<int bits>
    __global__ void computeSignature(
        const double* path_increments,
        double* signature,
        int d,
        int trunc_lvl,
        int fixed_len,
        int num_time_steps,
        uint64_t total_sig_size,
        const uint64_t* d_powers,
        const uint64_t* level_offsets,
        const unsigned* degrees,
        const unsigned* varying_words,
        const uint64_t* prefix_maps);


    /// Single precision version of computeSignature
    template<int bits>
    __global__ void computeSignature(
        const float* path_increments,
        float* signature,
        int d,
        int trunc_lvl,
        int fixed_len,
        int num_time_steps,
        uint64_t total_sig_size,
        const uint64_t* d_powers,
        const uint64_t* level_offsets,
        const unsigned* degrees,
        const unsigned* varying_words,
        const uint64_t* prefix_maps);


    /**
     * @brief CUDA kernel to compute the truncated signature with extended precision
     *
     * Similar to computeSignature but uses double-double arithmetic for higher precision.
     * Input increments are expected in double-double format (2 doubles per value).
     *
     * @tparam bits Number of bits used for packing letters (5, 8, or 10)
     * @param path_increments Path increment data in double-double format
     * @param signature Output signature tensor (batch_size x total_sig_size)
     * @param d Path dimension
     * @param trunc_lvl Truncation level
     * @param fixed_len Fixed prefix length
     * @param num_time_steps Number of time steps to compute signature over
     * @param total_sig_size Total size of the signature
     * @param d_powers Array of precomputed powers: d_powers[j] = d^j
     * @param level_offsets Array of precomputed level offsets: [0, 0, d, d^2 + d, ...]
     * @param degrees Array with the degree of each word in the decomposition
     * @param varying_words Array with varying words that extend the fixed word
     * @param prefix_maps Array with the mappings of each word's prefixes
     */
    template<int bits>
    __global__ void computePreciseSig(
        const double* path_increments,
        double* signature,
        int d,
        int trunc_lvl,
        int fixed_len,
        int num_time_steps,
        uint64_t total_sig_size,
        const uint64_t* d_powers,
        const uint64_t* level_offsets,
        const unsigned* degrees,
        const unsigned* varying_words,
        const uint64_t* prefix_maps);
} // namespace compute_sig
} // namespace pathsig
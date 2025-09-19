// sig_backprop.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdint>


namespace pathsig {
namespace sig_backprop {
    /**
     * @brief Main CUDA kernel for signature backpropagation (double precision)
     *
     * Computes gradients of loss L=L(f(S_{0,T}(X))) with respect to the path increments.
     *
     * @tparam bits Number of bits used for packing letters and maps (5, 8, or 10)
     * @tparam TRUNC_LVL Truncation level (compile-time constant for optimization)
     * @param path_increments Input path increment data
     * @param signature Forward pass signature values: S_{0,T}(X)
     * @param incoming_gradients Gradients of loss w.r.t. the signature: dL/dS_{0,T}(X)
     * @param increment_grads Output gradients: dL/dΔX
     * @param d_powers Array of precomputed powers: d_powers[j] = d^j
     * @param level_offsets Array of precomputed level offsets: [0, 0, d, d^2 + d, ...]
     * @param d Path dimension
     * @param fixed_len Number of fixed letters in word decomposition
     * @param num_time_steps Sequence length of increments tensor
     * @param total_sig_size Total signature size
     * @param degrees Array with the degree of each word in the decomposition
     * @param varying_words Array with varying words that extend the fixed word
     * @param prefix_maps Array with the mappings of each word's prefixes
     * @param num_lower Number of words in decomposition with degree < truncation_level
     */
    template<int bits, int TRUNC_LVL>
    __global__ void signatureBackwardPass(
        const double* __restrict__ path_increments,
        const double* __restrict__ signature,
        const double* __restrict__ incoming_gradients,
        double* __restrict__ increment_grads,
        const uint64_t* __restrict__ d_powers,
        const uint64_t* __restrict__ level_offsets,
        int d,
        int fixed_len,
        int num_time_steps,
        uint64_t total_sig_size,
        const unsigned* __restrict__ degrees,
        const unsigned* __restrict__ varying_words,
        const uint64_t* __restrict__ prefix_maps,
        int num_lower);


    /// Single precision version of signatureBackwardPass
    template<int bits, int TRUNC_LVL>
    __global__ void signatureBackwardPass(
        const float* __restrict__ path_increments,
        const float* __restrict__ signature,
        const float* __restrict__ incoming_gradients,
        float* __restrict__ increment_grads,
        const uint64_t* __restrict__ d_powers,
        const uint64_t* __restrict__ level_offsets,
        int d,
        int fixed_len,
        int num_time_steps,
        uint64_t total_sig_size,
        const unsigned* __restrict__ degrees,
        const unsigned* __restrict__ varying_words,
        const uint64_t* __restrict__ prefix_maps,
        int num_lower);


    /**
     * @brief Compute increment gradients from path gradients
     *
     * Computes the gradients of the path dL/dX_t from the gradients
     * with respect to increments dL/d(ΔX_t).
     *
     * @tparam Scalar Floating point type (float or double)
     * @param increment_gradients Input gradients of shape (batch_size, num_time_steps-1, path_dim)
     * @param path_gradients Output gradients of shape (batch_size, num_time_steps, path_dim)
     * @param batch_size Number of batch elements (i.e. the number of signatures)
     * @param num_time_steps Sequence length of the input path
     * @param path_dim Path dimension
     */
    template<typename Scalar>
    __global__ void incrementGradToPathGrad(
        const Scalar* __restrict__ increment_gradients,
        Scalar* __restrict__ path_gradients,
        int batch_size, int num_time_steps, int path_dim);
} //namespace sig_backprop
} //namespace pathsig
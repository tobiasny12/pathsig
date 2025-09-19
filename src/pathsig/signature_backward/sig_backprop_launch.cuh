// sig_backprop_launch.cuh
#pragma once
#include <torch/types.h>
#include "SigDecomposition.h"


namespace pathsig {
    /// Templated dispatcher on truncation level
    template <int TRUNC_LVL>
    torch::Tensor dispatchByTruncLvl(
        const torch::Tensor& path,
        const torch::Tensor& signature,
        const torch::Tensor& incoming_grads,
        int truncation_level,
        const SigDecomposition* decomp_ptr);


    /**
     * @brief Computes gradients of L(f(S_{0,T}(X))) with respect to the input path X
     *
     * @param path Input path tensor of shape (batch_size, time_steps, path_dim)
     * @param signature Forward pass signature tensor of shape (batch_size, signature_size)
     * @param incoming_grads Gradients with respect to the signature from the next layer of shape (batch_size, signature_size)
     * @param truncation_level Maximum depth of the signature (must match forward pass)
     * @return Tensor containing gradients with respect to the path of shape (batch_size, time_steps, path_dim)
     */
    torch::Tensor computeSigGradients(const torch::Tensor& path,
                                     const torch::Tensor& signature,
                                     const torch::Tensor& incoming_grads,
                                     int truncation_level,
                                     const SigDecomposition* decomp_ptr = nullptr);


    /**
    * @brief Templated implementation for computing gradients of the signature with respect to the input path
    *
    * @tparam Scalar Floating point type (float or double)
    * @tparam Bits Bit width for decomposition (5, 8, or 10)
    * @tparam TRUNC_LVL Truncation level (2 to 12)
    */
    template <typename Scalar, int Bits, int TRUNC_LVL>
    torch::Tensor computeSigGradientsImpl(const torch::Tensor& path,
                                         const torch::Tensor& signature,
                                         const torch::Tensor& incoming_grads,
                                         int truncation_level,
                                         const SigDecomposition* decomp_ptr);
} // namespace pathsig
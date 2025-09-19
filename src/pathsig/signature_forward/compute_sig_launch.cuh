// compute_sig_launch.cuh
#pragma once
#include <torch/types.h>
#include "SigDecomposition.h"


namespace pathsig {
    /**
     * @brief Dispatch function for computing signatures of input paths
     *
     * @param path Input path tensor of shape [batch_size, time_steps, path_dim]
     * @param truncation_level Maximum depth of the signature (2 to 12)
     * @param extended_precision If true and type is double, uses extended precision for signature computation
     * @param decomp Optional pre-computed decomposition
     *
     * @return torch::Tensor Signature tensor of shape [batch_size, signature_size]
     */
    torch::Tensor computeSignature(
        const torch::Tensor& path,
        int truncation_level,
        bool extended_precision = false,
        const SigDecomposition* decomp = nullptr
    );


    /**
     * @brief Templated function for launching the correct kernel for the signature computation
     *
     * @tparam T Floating point type (float or double)
     * @tparam Bits Bit width for decomposition (5, 8, or 10)
     */
    template<typename T, int Bits>
    torch::Tensor computeSignatureImpl(
        const torch::Tensor& path,
        int truncation_level,
        bool extended_precision,
        const SigDecomposition* decomp_ptr
    );
} // namespace pathsig
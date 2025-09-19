// sig_setup.cuh - Header for shared setup utilities and increment computation
#pragma once
#include <torch/types.h>
#include <cstdint>

namespace pathsig {
    namespace sig_setup {

        // Constraints
        constexpr int MIN_PATH_DIM = 1;
        constexpr int MAX_PATH_DIM = 1023;
        constexpr int MIN_TRUNC_LVL = 2;
        constexpr int MAX_TRUNC_LVL = 12;


        /// @brief Validates the input path tensor is given correctly and that minimal requirements are satisfied.
        ///
        /// @param path Input path tensor (batch_size, time_steps, path_dim)
        /// @param truncation_level Truncation level for signature computation
        /// @throws std::runtime_error if validation fails
        ///
        void validateInputs(const torch::Tensor& path, int truncation_level);


        /// @brief Computes the total size of the signature
        ///
        /// @param path_dim Dimension of the path
        /// @param truncation_level Truncation level
        /// @return Total number of signature components
        ///
        uint64_t computeTotalSignatureSize(int path_dim, int truncation_level);


        /// @brief Computes path increments used in the signature forward and backward pass
        ///
        /// @param path Input path tensor (batch_size, time_steps, path_dim)
        /// @param extended_precision Whether to compute increments in double-double (available for FP64)
        /// @return Tensor containing path increments
        ///
        torch::Tensor computePathIncrements(const torch::Tensor& path, bool extended_precision);


        // Kernel for computing increments for FP64
        __global__ void computePreciseIncrements(
            const double* path,
            double* increments_out,
            int batch_size,
            int num_time_steps,
            int path_dim,
            int num_increments,
            bool extended_precision);
    } // namespace sig_setup
} // namespace pathsig
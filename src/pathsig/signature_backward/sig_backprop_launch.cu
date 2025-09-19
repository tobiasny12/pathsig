// sig_backprop_launch.cu
#include "sig_backprop_launch.cuh"
#include "sig_backprop.cuh"
#include "sig_setup.cuh"
#include "SigDecomposition.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Exceptions.h>
#include <stdexcept>
#include <memory>


namespace pathsig {

/// Templated dispatcher on truncation level
template <int TRUNC_LVL>
torch::Tensor dispatchByTruncLvl(
    const torch::Tensor& path,
    const torch::Tensor& signature,
    const torch::Tensor& incoming_grads,
    int truncation_level,
    const SigDecomposition* decomp_ptr)
{
    constexpr int Bits = (TRUNC_LVL <= 6 ? 10 : (TRUNC_LVL <= 8 ? 8 : 5));
    auto dtype = path.scalar_type();

    if (dtype == torch::kFloat64) {
        return computeSigGradientsImpl<double, Bits, TRUNC_LVL>(
            path, signature, incoming_grads, truncation_level, decomp_ptr);
    } else if (dtype == torch::kFloat32) {
        return computeSigGradientsImpl<float, Bits, TRUNC_LVL>(
            path, signature, incoming_grads, truncation_level, decomp_ptr);
    } else {
        throw std::runtime_error("Unsupported data type. Only FP32 and FP64 are supported.");
    }
}


/// Dispatch function for backward pass of signature
torch::Tensor computeSigGradients(
    const torch::Tensor& path,
    const torch::Tensor& signature,
    const torch::Tensor& incoming_grads,
    int truncation_level,
    const SigDecomposition* decomp_ptr)
{
    // Validate inputs
    sig_setup::validateInputs(path, truncation_level);

    // Make tensors contiguous and on same device
    auto path_c = path.contiguous();
    auto sig_c = signature.to(path.device(), /*non_blocking=*/true).contiguous();
    auto grads_c = incoming_grads.to(path.device(), /*non_blocking=*/true).contiguous();

    // Dispatch based on truncation_level
    switch (truncation_level) {
        case 2:  return dispatchByTruncLvl<2>(path_c, sig_c, grads_c, truncation_level, decomp_ptr);
        case 3:  return dispatchByTruncLvl<3>(path_c, sig_c, grads_c, truncation_level, decomp_ptr);
        case 4:  return dispatchByTruncLvl<4>(path_c, sig_c, grads_c, truncation_level, decomp_ptr);
        case 5:  return dispatchByTruncLvl<5>(path_c, sig_c, grads_c, truncation_level, decomp_ptr);
        case 6:  return dispatchByTruncLvl<6>(path_c, sig_c, grads_c, truncation_level, decomp_ptr);
        case 7:  return dispatchByTruncLvl<7>(path_c, sig_c, grads_c, truncation_level, decomp_ptr);
        case 8:  return dispatchByTruncLvl<8>(path_c, sig_c, grads_c, truncation_level, decomp_ptr);
        case 9:  return dispatchByTruncLvl<9>(path_c, sig_c, grads_c, truncation_level, decomp_ptr);
        case 10: return dispatchByTruncLvl<10>(path_c, sig_c, grads_c, truncation_level, decomp_ptr);
        case 11: return dispatchByTruncLvl<11>(path_c, sig_c, grads_c, truncation_level, decomp_ptr);
        case 12: return dispatchByTruncLvl<12>(path_c, sig_c, grads_c, truncation_level, decomp_ptr);
        default:
            throw std::runtime_error(
                "Unsupported truncation_level: " + std::to_string(truncation_level) +
                ". Must be between 2 and 12.");
    }
}

// Templated implementation for different scalar types, bits, and truncation levels
template <typename Scalar, int Bits, int TRUNC_LVL>
torch::Tensor computeSigGradientsImpl(
    const torch::Tensor& path,
    const torch::Tensor& signature,
    const torch::Tensor& incoming_grads,
    int truncation_level,
    const SigDecomposition* decomp_ptr)
{
    // Manual device management to guard the device before any CUDA calls
    int prev_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&prev_device));

    const int dev = path.get_device();
    if (prev_device != dev) {
        C10_CUDA_CHECK(cudaSetDevice(dev));
    }

    // Setting stream
    auto torch_stream = c10::cuda::getCurrentCUDAStream(dev);
    cudaStream_t stream = torch_stream.stream();

    // Check for template match
    TORCH_CHECK(truncation_level == TRUNC_LVL,
                "Truncation level mismatch between call and template");

    // Get dimensions from path
    int batch_size = static_cast<int>(path.size(0));
    int num_time_steps = static_cast<int>(path.size(1));
    int path_dim = static_cast<int>(path.size(2));
    int num_increment_steps = num_time_steps - 1;

    // Compute path increments
    torch::Tensor path_increments;
    if constexpr (std::is_same_v<Scalar, float>) {
        path_increments = (path.slice(/*dim=*/1, /*start=*/1, /*end=*/num_time_steps) -
                          path.slice(/*dim=*/1, /*start=*/0, /*end=*/num_time_steps - 1))
                          .contiguous()
                          .view({batch_size * num_increment_steps, path_dim});
    } else {
        path_increments = sig_setup::computePathIncrements(path, /*extended_precision*/false);
    }

    // Allocate output gradient tensor for increments
    auto increment_grads = torch::zeros(
        {static_cast<int64_t>(batch_size), static_cast<int64_t>(num_increment_steps), static_cast<int64_t>(path_dim)},
        path.options()
    );

    // Use decomposition if provided or create new
    std::unique_ptr<SigDecomposition> decomp_owner;
    const SigDecomposition* decomp;

    if (decomp_ptr) {
        decomp = decomp_ptr;
    } else {
        decomp_owner = std::make_unique<SigDecomposition>(
            path_dim, truncation_level);
        decomp = decomp_owner.get();
    }

    // Get parameters from decomposition
    int partial_sig_sz = decomp->getPartialSigSize();
    int num_partial_sigs = decomp->getNumPartialSigs();
    int num_lower = decomp->getNumLower();
    int fixed_len = truncation_level - decomp->getNumFreeLetters();
    int runtime_bits = decomp->getNumBits(truncation_level);

    TORCH_CHECK(runtime_bits == Bits,
                "Bit width mismatch between template and decomposition");

    // Move setup data from decomp to device asynchronously
    auto d_powers_device = decomp->getDPowers().to(
        path.device(), /*non_blocking=*/true);
    auto level_offsets_device = decomp->getLevelOffsets().to(
        path.device(), /*non_blocking=*/true);
    auto degrees_device = decomp->getDegrees().to(
        path.device(), /*non_blocking=*/true);
    auto varying_words_device = decomp->getVaryingWords().to(
        path.device(), /*non_blocking=*/true);
    auto prefix_maps_device = decomp->getPrefixMaps().to(
        path.device(), /*non_blocking=*/true);

    uint64_t total_sig_size = decomp->getTotalSigSize();

    // Calculate shared memory size
    const int warp_size = 32;
    int num_warps = (partial_sig_sz + warp_size - 1) / warp_size;

    size_t shmem_bytes = sizeof(Scalar) * (
        path_dim +                     // SHARED_PATH_INCREMENTS
        num_lower +                    // SHARED_PARTIAL_SIG
        num_warps * num_lower +        // SHARED_COMMON_TERMS
        path_dim * num_warps           // SHARED_LETTER_GRADS
    );

    // Configure kernel launch parameters
    dim3 block(partial_sig_sz);
    dim3 grid(num_partial_sigs, batch_size);

    // Launch kernel
    sig_backprop::signatureBackwardPass<Bits, TRUNC_LVL><<<grid, block, shmem_bytes, stream>>>(
        path_increments.data_ptr<Scalar>(),
        signature.data_ptr<Scalar>(),
        incoming_grads.data_ptr<Scalar>(),
        increment_grads.data_ptr<Scalar>(),
        reinterpret_cast<const uint64_t*>(d_powers_device.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(level_offsets_device.data_ptr<int64_t>()),
        path_dim, fixed_len, num_increment_steps, total_sig_size,
        reinterpret_cast<const unsigned*>(degrees_device.data_ptr<int32_t>()),
        reinterpret_cast<const unsigned*>(varying_words_device.data_ptr<int32_t>()),
        reinterpret_cast<const uint64_t*>(prefix_maps_device.data_ptr<int64_t>()),
        num_lower);
    C10_CUDA_CHECK(cudaGetLastError());

    // Computing gradients w.r.t. path from gradients w.r.t. increments
    auto path_grads = torch::zeros(
        {static_cast<int64_t>(batch_size), static_cast<int64_t>(num_time_steps), static_cast<int64_t>(path_dim)},
        path.options()
    );

    dim3 threads(256);
    dim3 blocks((batch_size * num_time_steps * path_dim + 255) / 256);

    // Launching kernel
    sig_backprop::incrementGradToPathGrad<<<blocks, threads, /*shmem=*/0, stream>>>(
        increment_grads.data_ptr<Scalar>(),
        path_grads.data_ptr<Scalar>(),
        batch_size, num_time_steps, path_dim);
    C10_CUDA_CHECK(cudaGetLastError());


    // Restore previous device
    if (prev_device != dev) {
        C10_CUDA_CHECK(cudaSetDevice(prev_device));
    }

    return path_grads;
}
} // namespace pathsig
// compute_sig_launch.cu - Host function for the forward pass (signature computation)
#include "compute_sig_launch.cuh"
#include "sig_setup.cuh"
#include "SigDecomposition.h"
#include "compute_sig.cuh"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Exceptions.h>
#include <stdexcept>
#include <iostream>
#include <memory>


namespace pathsig {
/// Dispatch function for signature computation
torch::Tensor computeSignature(
    const torch::Tensor& path,
    int truncation_level,
    bool extended_precision,
    const SigDecomposition* decomp)
{
    auto contig_path = path.contiguous();
    auto dtype = contig_path.scalar_type();

    if (truncation_level <= 6) {
        if (dtype == torch::kFloat64) {
            return computeSignatureImpl<double, /*bits=*/10>(
                contig_path, truncation_level, extended_precision, decomp);
        } else if (dtype == torch::kFloat32) {
            return computeSignatureImpl<float, /*bits=*/10>(
                contig_path, truncation_level, extended_precision, decomp);
        }
    } else if (truncation_level <= 8) {
        if (dtype == torch::kFloat64) {
            return computeSignatureImpl<double, /*bits=*/8>(
                contig_path, truncation_level, extended_precision, decomp);
        } else if (dtype == torch::kFloat32) {
            return computeSignatureImpl<float, /*bits=*/8>(
                contig_path, truncation_level, extended_precision, decomp);
        }
    } else {
        if (dtype == torch::kFloat64) {
            return computeSignatureImpl<double, /*bits=*/5>(
                contig_path, truncation_level, extended_precision, decomp);
        } else if (dtype == torch::kFloat32) {
            return computeSignatureImpl<float, /*bits=*/5>(
                contig_path, truncation_level, extended_precision, decomp);
        }
    }
    throw std::runtime_error("Unsupported data type. Only FP32 and FP64 are supported.");
}


/// Signature launch function templated on float, double, and bits
template<typename T, int Bits>
torch::Tensor computeSignatureImpl(
    const torch::Tensor& path,
    int truncation_level,
    bool extended_precision,
    const SigDecomposition* decomp_ptr)
{

    // Validate input
    sig_setup::validateInputs(path, truncation_level);

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

    // Get dimensions from path
    int batch_size = static_cast<int>(path.size(0));
    int num_time_steps = static_cast<int>(path.size(1));
    int path_dim = static_cast<int>(path.size(2));
    int num_increment_steps = num_time_steps - 1;

    // Handle extended_precision for float
    if constexpr (std::is_same_v<T, float>) {
        if (extended_precision) {
            std::cerr << "Warning: extended_precision mode is not supported for FP32. "
                     << "Proceeding without extended precision." << std::endl;
            extended_precision = false;
        }
    }

    // Use provided decomposition or create new
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
    size_t shmem_bytes = (path_dim + partial_sig_sz) * sizeof(T);
    if (extended_precision && std::is_same_v<T, double>) {
        shmem_bytes *= 2;
    }

    // Compute increments of the path
    torch::Tensor path_increments;
    if constexpr (std::is_same_v<T, double>) {
        path_increments = sig_setup::computePathIncrements(path, extended_precision);
    } else {
        path_increments = (path.slice(1, 1, num_time_steps) -
                          path.slice(1, 0, num_time_steps - 1))
                          .contiguous()
                          .view({batch_size * num_increment_steps, path_dim});
    }

    // Allocate a tensor for the signatures
    auto signature_output = torch::zeros(
        {static_cast<int64_t>(batch_size), static_cast<int64_t>(total_sig_size)},
        path.options());

    // Configure kernel launch
    dim3 block(partial_sig_sz);
    dim3 grid(num_partial_sigs, batch_size);

    // Signature computation kernel launch
    if constexpr (std::is_same_v<T, double>) {
        if (extended_precision) {
            compute_sig::computePreciseSig<Bits><<<grid, block, shmem_bytes, stream>>>(
                path_increments.data_ptr<T>(),
                signature_output.data_ptr<T>(),
                path_dim, truncation_level, fixed_len, num_increment_steps,
                total_sig_size,
                reinterpret_cast<uint64_t*>(d_powers_device.data_ptr<int64_t>()),
                reinterpret_cast<uint64_t*>(level_offsets_device.data_ptr<int64_t>()),
                reinterpret_cast<unsigned*>(degrees_device.data_ptr<int32_t>()),
                reinterpret_cast<unsigned*>(varying_words_device.data_ptr<int32_t>()),
                reinterpret_cast<uint64_t*>(prefix_maps_device.data_ptr<int64_t>()));
        } else {
            compute_sig::computeSignature<Bits><<<grid, block, shmem_bytes, stream>>>(
                path_increments.data_ptr<T>(),
                signature_output.data_ptr<T>(),
                path_dim, truncation_level, fixed_len, num_increment_steps,
                total_sig_size,
                reinterpret_cast<uint64_t*>(d_powers_device.data_ptr<int64_t>()),
                reinterpret_cast<uint64_t*>(level_offsets_device.data_ptr<int64_t>()),
                reinterpret_cast<unsigned*>(degrees_device.data_ptr<int32_t>()),
                reinterpret_cast<unsigned*>(varying_words_device.data_ptr<int32_t>()),
                reinterpret_cast<uint64_t*>(prefix_maps_device.data_ptr<int64_t>()));
        }
    } else {
        compute_sig::computeSignature<Bits><<<grid, block, shmem_bytes, stream>>>(
            path_increments.data_ptr<T>(),
            signature_output.data_ptr<T>(),
            path_dim, truncation_level, fixed_len, num_increment_steps,
            total_sig_size,
            reinterpret_cast<uint64_t*>(d_powers_device.data_ptr<int64_t>()),
            reinterpret_cast<uint64_t*>(level_offsets_device.data_ptr<int64_t>()),
            reinterpret_cast<unsigned*>(degrees_device.data_ptr<int32_t>()),
            reinterpret_cast<unsigned*>(varying_words_device.data_ptr<int32_t>()),
            reinterpret_cast<uint64_t*>(prefix_maps_device.data_ptr<int64_t>()));
    }

    // Check for kernel launch errors
    C10_CUDA_CHECK(cudaGetLastError());

    // Restore previous device
    if (prev_device != dev) {
        C10_CUDA_CHECK(cudaSetDevice(prev_device));
    }

    return signature_output;
}
} // namespace pathsig
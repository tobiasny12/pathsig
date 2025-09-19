// sig_setup.cu - Shared setup utilities and increment computation
#include "sig_setup.cuh"
#include "extended_precision.cuh"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Exceptions.h>
#include <stdexcept>
#include <vector>
#include <limits>

namespace pathsig {
namespace sig_setup {

/// Input validation
void validateInputs(const torch::Tensor& path, int truncation_level)
{
    if (!path.is_cuda()) {
        throw std::runtime_error("CPU path support not implemented – supply a CUDA tensor");
    }

    if (path.dim() != 3) {
        throw std::runtime_error("Path must be a 3D tensor (batch, time, features)");
    }

    int batch_size = path.size(0);
    int num_time_steps = path.size(1);
    int path_dim = path.size(2);

    if (batch_size <= 0) {
        throw std::runtime_error("First dimension of the path tensor (batch size) must be positive");
    }

    if (num_time_steps <= 1) {
        throw std::runtime_error("Second dimension of the path tensor (time) must be at least 2");
    }

    if (path_dim < MIN_PATH_DIM || path_dim > MAX_PATH_DIM) {
        throw std::invalid_argument(
            std::string("Path/feature dimension must be between ") +
            std::to_string(MIN_PATH_DIM) + " and " + std::to_string(MAX_PATH_DIM));
    }

    if (truncation_level < MIN_TRUNC_LVL || truncation_level > MAX_TRUNC_LVL) {
        throw std::invalid_argument(
            std::string("Truncation level must be between ") +
            std::to_string(MIN_TRUNC_LVL) + " and " + std::to_string(MAX_TRUNC_LVL));
    }

    // Check dimension/truncation level compatibility
    if (!((truncation_level <= 6 && path_dim <= 1023) ||
          (truncation_level <= 8 && path_dim <= 255) ||
          (truncation_level <= 12 && path_dim <= 31))) {
        throw std::invalid_argument(
            "Truncation level N and path/feature dimension d must satisfy one of:\n"
            " - N <= 12 and d <= 31\n"
            " - N <= 8 and d <= 255\n"
            " - N <= 6 and d <= 1023");
    }
}


/// Function for computing number of elements in a signature
uint64_t computeTotalSignatureSize(int path_dim, int truncation_level)
{
    uint64_t total_terms = 0;
    uint64_t power = 1;
    uint64_t max_uint64 = std::numeric_limits<uint64_t>::max();

    for (int k = 1; k <= truncation_level; ++k) {
        // Check if multiplication would overflow
        if (power > max_uint64 / path_dim) {
            throw std::overflow_error("Overflow in computing signature size: Signature size is too large.");
        }
        power *= path_dim;

        // Check if addition would overflow
        if (total_terms > max_uint64 - power) {
            throw std::overflow_error("Overflow in computing signature size: Signature size is too large.");
        }
        total_terms += power;
    }

    return total_terms;
}


/// Host function for computing path increments
torch::Tensor computePathIncrements(const torch::Tensor& path, bool extended_precision)
{
    // Manual device management to guard the device before any CUDA calls
    int prev_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&prev_device));
    const int dev = path.get_device();
    if (prev_device != dev) {
        C10_CUDA_CHECK(cudaSetDevice(dev));
    }

    // Get stream
    auto torch_stream = c10::cuda::getCurrentCUDAStream(dev);
    cudaStream_t stream = torch_stream.stream();

    // Get dimensions from path
    int batch_size = path.size(0);
    int num_time_steps = path.size(1);
    int path_dim = path.size(2);
    int num_increment_steps = num_time_steps - 1;

    // Determine the size of the increments tensor
    std::vector<int64_t> increments_shape;
    if (extended_precision) {
        // For extended_precision mode, we need twice the size
        increments_shape = {
            (int64_t)batch_size,
            (int64_t)num_increment_steps,
            (int64_t)(2 * path_dim)
        };
    } else {
        increments_shape = {
            (int64_t)batch_size,
            (int64_t)num_increment_steps,
            (int64_t)path_dim
        };
    }

    // Allocate tensor for increments
    auto path_increments = torch::empty(
        increments_shape,
        path.options().dtype(torch::kFloat64)
    );

    // Launch kernel to compute increments
    int total_elements = batch_size * num_increment_steps * path_dim;
    dim3 threads_per_block(256);
    dim3 num_blocks((total_elements + threads_per_block.x - 1) / threads_per_block.x);

    computePreciseIncrements<<<num_blocks, threads_per_block, 0, stream>>>(
        path.data_ptr<double>(),
        path_increments.data_ptr<double>(),
        batch_size,
        num_time_steps,
        path_dim,
        num_increment_steps,
        extended_precision
    );

    // Check for CUDA errors
    C10_CUDA_CHECK(cudaGetLastError());

    // Restore previous device
    if (prev_device != dev) {
        C10_CUDA_CHECK(cudaSetDevice(prev_device));
    }

    return path_increments;
}


/// Kernel for computing path increments with optional extended precision
__global__ void computePreciseIncrements(
    const double* path,
    double* increments_out,
    int batch_size,
    int num_time_steps,
    int path_dim,
    int num_increments,
    bool extended_precision)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_increments * path_dim;

    if (global_idx >= total_elements) return;

    // Decompose global index
    int batch_idx = global_idx / (num_increments * path_dim);
    int time_idx = (global_idx / path_dim) % num_increments;
    int dim_idx = global_idx % path_dim;

    // Compute indices for current and next time points
    int next_idx = batch_idx * num_time_steps * path_dim +
                   (time_idx + 1) * path_dim + dim_idx;
    int curr_idx = batch_idx * num_time_steps * path_dim +
                   time_idx * path_dim + dim_idx;

    // Load values
    double next_value = path[next_idx];
    double curr_value = path[curr_idx];

    // Compute difference with extended precision
    double108_t extended_next = make_double2(0.0, next_value);
    double108_t extended_curr = make_double2(0.0, curr_value);
    double108_t extended_diff = extended_prec::sub_double108_t(extended_next, extended_curr);

    if (extended_precision) {
        // Output as double-double (low, high)
        int out_idx = batch_idx * num_increments * path_dim * 2 +
                      time_idx * path_dim * 2 + dim_idx * 2;
        increments_out[out_idx] = extended_diff.x;      // Low part
        increments_out[out_idx + 1] = extended_diff.y;  // High part
    } else {
        // Output as standard double
        int out_idx = batch_idx * num_increments * path_dim +
                      time_idx * path_dim + dim_idx;
        increments_out[out_idx] = extended_diff.y + extended_diff.x;
    }
}

} // namespace sig_setup
} // namespace pathsig
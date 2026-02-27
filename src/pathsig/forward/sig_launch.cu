// sig_launch.cu
#include "sig_launch.cuh"
#include "sig_kernels.cuh"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace pathsig::launch::forward {

namespace {

// Helpers
inline uint32_t choose_block_size(uint64_t level_size) {
    if (level_size <= 32ULL)  return 32U;
    if (level_size >= 256ULL) return 256U;
    return (uint32_t)(((level_size + 31ULL) / 32ULL) * 32ULL);
}

constexpr int bits_for_degree(int degree) {
    return (degree == 1) ? 32 : (64 / degree);
}

struct LaunchConfig {
    dim3 block;
    dim3 grid;
    size_t shmem;
};

inline LaunchConfig make_launch_config(
    uint64_t level_size,
    uint32_t batch_size,
    uint32_t num_windows,
    size_t shmem_bytes
) {
    const uint32_t block_size = choose_block_size(level_size);
    const uint32_t grid_x = static_cast<uint32_t>((level_size + static_cast<uint64_t>(block_size) - 1ULL) / static_cast<uint64_t>(block_size));
    return { dim3(block_size, 1, 1), dim3(grid_x, batch_size, num_windows), shmem_bytes };
}

// -------------------------
// Degree dispatch (1..12)
// -------------------------
#define PATHSIG_DISPATCH_DEGREE(deg, CALL)                           \
    do {                                                             \
        switch (deg) {                                               \
            case 1:  { constexpr int DEG = 1;  CALL; } break;        \
            case 2:  { constexpr int DEG = 2;  CALL; } break;        \
            case 3:  { constexpr int DEG = 3;  CALL; } break;        \
            case 4:  { constexpr int DEG = 4;  CALL; } break;        \
            case 5:  { constexpr int DEG = 5;  CALL; } break;        \
            case 6:  { constexpr int DEG = 6;  CALL; } break;        \
            case 7:  { constexpr int DEG = 7;  CALL; } break;        \
            case 8:  { constexpr int DEG = 8;  CALL; } break;        \
            case 9:  { constexpr int DEG = 9;  CALL; } break;        \
            case 10: { constexpr int DEG = 10; CALL; } break;        \
            case 11: { constexpr int DEG = 11; CALL; } break;        \
            case 12: { constexpr int DEG = 12; CALL; } break;        \
            default: TORCH_CHECK(false, "Unsupported degree ", deg); \
        }                                                            \
    } while (0)

} // namespace

// Signature forward
at::Tensor computeSignature(
    const at::Tensor& path_in,
    int64_t depth,
    bool alternative_projection,
    const at::Tensor& encoded_words,          // CUDA int64, ONLY non-full levels concatenated
    c10::ArrayRef<int64_t> level_sizes,       // CPU int64, level_sizes[deg]
    bool use_windows,
    const at::Tensor& windows
) {
    TORCH_CHECK(path_in.is_cuda(), "path must be CUDA");
    TORCH_CHECK(path_in.scalar_type() == at::kFloat || path_in.scalar_type() == at::kDouble,
                "path must be float32 or float64");
    TORCH_CHECK(path_in.dim() == 3, "path must be [B,T,d]");
    TORCH_CHECK(depth >= 1 && depth <= 12, "depth must be in [1,12]");

    const int depth_i = (int)depth;

    if (alternative_projection) {
        TORCH_CHECK(encoded_words.is_cuda(), "encoded_words must be CUDA");
        TORCH_CHECK(encoded_words.scalar_type() == at::kLong, "encoded_words must be int64");
        TORCH_CHECK(encoded_words.is_contiguous(), "encoded_words must be contiguous");
        TORCH_CHECK(encoded_words.dim() == 1, "encoded_words must be 1D");
    }

    c10::cuda::CUDAGuard device_guard(path_in.device());
    const int dev = path_in.device().index();

    const at::Tensor path = path_in.contiguous();
    const uint32_t batch_size = (uint32_t)path.size(0);
    const int path_len = (int)path.size(1);
    const int path_dim = (int)path.size(2);
    TORCH_CHECK(path_len >= 2, "path must have at least 2 time steps");

    const uint64_t* encoded_words_ptr = nullptr;
    if (alternative_projection) {
        encoded_words_ptr = reinterpret_cast<const uint64_t*>(encoded_words.data_ptr<int64_t>());
    }

    const int* windows_ptr = nullptr;
    uint32_t num_windows = 1;
    if (use_windows) {
        TORCH_CHECK(windows.is_cuda(), "windows must be CUDA");
        TORCH_CHECK(windows.is_contiguous(), "windows must be contiguous");
        TORCH_CHECK(windows.dim() == 2 && windows.size(1) == 2, "windows must be [W,2]");
        num_windows = (uint32_t)windows.size(0);
        windows_ptr = windows.data_ptr<int>();
    }

    // sig_size = sum of output level sizes
    uint64_t sig_size = 0ULL;
    if (alternative_projection) {
        for (int degree = 1; degree <= depth_i; ++degree) {
            sig_size += (uint64_t)level_sizes[degree];
        }
    } else {
        uint64_t d_pow = 1ULL;
        for (int degree = 1; degree <= depth_i; ++degree) {
            d_pow *= (uint64_t)path_dim;
            sig_size += d_pow;
        }
    }

    at::Tensor signature =
        use_windows
            ? at::empty({(int64_t)batch_size, (int64_t)num_windows, (int64_t)sig_size}, path.options())
            : at::empty({(int64_t)batch_size, (int64_t)sig_size}, path.options());

    // streams: base + (depth-1) workers
    c10::cuda::CUDAStream base_stream_obj = c10::cuda::getCurrentCUDAStream(dev);
    cudaStream_t base_stream = base_stream_obj.stream();

    cudaEvent_t inputs_ready;
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&inputs_ready, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventRecord(inputs_ready, base_stream));

    std::array<c10::cuda::CUDAStream, 12> worker_streams = {
        base_stream_obj, base_stream_obj, base_stream_obj, base_stream_obj, base_stream_obj, base_stream_obj,
        base_stream_obj, base_stream_obj, base_stream_obj, base_stream_obj, base_stream_obj, base_stream_obj
    };
    for (int i = 1; i < depth_i; ++i) {
        worker_streams[i] = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, dev);
        C10_CUDA_CHECK(cudaStreamWaitEvent(worker_streams[i].stream(), inputs_ready, 0));
    }

    C10_CUDA_CHECK(cudaEventDestroy(inputs_ready));

    AT_DISPATCH_FLOATING_TYPES(path.scalar_type(), "pathsig::computeSignature", [&] {
        const scalar_t* path_ptr = path.data_ptr<scalar_t>();
        scalar_t* signature_ptr  = signature.data_ptr<scalar_t>();

        const size_t shmem_bytes = sizeof(scalar_t) * (size_t)path_dim;

        uint64_t level_offset      = 0ULL;
        uint64_t encoded_words_off = 0ULL; // offset for encoded_words (non-full levels only)
        uint64_t d_pow             = 1ULL; // d^degree

        for (int degree = 1; degree <= depth_i; ++degree) {
            d_pow *= (uint64_t)path_dim;

            uint64_t level_size = d_pow;
            const uint64_t* words_at_lvl = nullptr;

            if (alternative_projection) {
                level_size = (uint64_t)level_sizes[degree];
                const bool non_full = (level_size != d_pow);
                words_at_lvl = non_full ? (encoded_words_ptr + encoded_words_off) : nullptr;
                if (non_full) encoded_words_off += level_size;
            }

            if (level_size == 0ULL) continue;

            cudaStream_t stream = worker_streams[degree - 1].stream();
            const LaunchConfig cfg = make_launch_config(level_size, batch_size, num_windows, shmem_bytes);

            PATHSIG_DISPATCH_DEGREE(degree, ([&] {
                pathsig::kernels::forward::compute_signature_level<scalar_t, bits_for_degree(DEG), DEG>
                    <<<cfg.grid, cfg.block, cfg.shmem, stream>>>(
                        path_ptr,
                        signature_ptr,
                        path_dim,
                        path_len,
                        sig_size,
                        level_size,
                        level_offset,
                        words_at_lvl,
                        windows_ptr
                    );
            }()));

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            level_offset += level_size;
        }

        // sync workers with base stream
        std::array<cudaEvent_t, 12> done_events{};
        for (int i = 1; i < depth_i; ++i) {
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&done_events[i], cudaEventDisableTiming));
        }

        for (int i = 1; i < depth_i; ++i) {
            C10_CUDA_CHECK(cudaEventRecord(done_events[i], worker_streams[i].stream()));
            C10_CUDA_CHECK(cudaStreamWaitEvent(base_stream, done_events[i], 0));
        }

        for (int i = 1; i < depth_i; ++i) {
            C10_CUDA_CHECK(cudaEventDestroy(done_events[i]));
        }
    });

    return signature;
}

// ----------------------------
// Signature -> logsig
// ----------------------------
std::tuple<at::Tensor, at::Tensor> sig_to_logsig(
    const at::Tensor& signature_in,           // [B,S] or [B,W,S]
    int64_t depth,
    int64_t d,
    bool alternative_projection,
    const at::Tensor& encoded_words,          // CUDA int64, optional and only top level
    c10::ArrayRef<int64_t> level_sizes        // CPU int64, level_sizes[deg]
) {
    TORCH_CHECK(signature_in.is_cuda(), "signature must be CUDA");
    TORCH_CHECK(signature_in.scalar_type() == at::kFloat || signature_in.scalar_type() == at::kDouble,
                "signature must be float32 or float64");
    TORCH_CHECK(depth >= 1 && depth <= 12, "depth must be in [1,12]");
    TORCH_CHECK(d >= 1, "d must be >= 1");

    const int depth_i = (int)depth;
    const int d_i     = (int)d;

    if (alternative_projection) {
        TORCH_CHECK(encoded_words.is_cuda(), "encoded_words must be CUDA");
        TORCH_CHECK(encoded_words.scalar_type() == at::kLong, "encoded_words must be int64");
        TORCH_CHECK(encoded_words.is_contiguous(), "encoded_words must be contiguous");
        TORCH_CHECK(encoded_words.dim() == 1, "encoded_words must be 1D");
    }

    c10::cuda::CUDAGuard device_guard(signature_in.device());
    const int dev = signature_in.device().index();

    const at::Tensor signature = signature_in.contiguous();

    uint32_t batch_size = 0;
    uint32_t num_windows = 1;
    uint64_t sig_size = 0ULL;

    if (signature.dim() == 3) {
        batch_size = (uint32_t)signature.size(0);
        num_windows = (uint32_t)signature.size(1);
        sig_size = (uint64_t)signature.size(2);
    } else {
        TORCH_CHECK(signature.dim() == 2, "signature must be [B,S] or [B,W,S]");
        batch_size = (uint32_t)signature.size(0);
        num_windows = 1;
        sig_size = (uint64_t)signature.size(1);
    }

    const uint64_t P_size = sig_size * (uint64_t)depth_i;

    at::Tensor logsig, P_arr;
    if (signature.dim() == 3) {
        logsig = at::zeros({(int64_t)batch_size, (int64_t)num_windows, (int64_t)sig_size}, signature.options());
        P_arr  = at::zeros({(int64_t)batch_size, (int64_t)num_windows, (int64_t)P_size}, signature.options());
    } else {
        logsig = at::zeros({(int64_t)batch_size, (int64_t)sig_size}, signature.options());
        P_arr  = at::zeros({(int64_t)batch_size, (int64_t)P_size}, signature.options());
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev).stream();

    const uint64_t* encoded_words_ptr = nullptr;
    if (alternative_projection) {
        encoded_words_ptr = reinterpret_cast<const uint64_t*>(encoded_words.data_ptr<int64_t>());
    }

    AT_DISPATCH_FLOATING_TYPES(signature.scalar_type(), "pathsig::sig_to_logsig", [&] {
        const scalar_t* signature_ptr = signature.data_ptr<scalar_t>();
        scalar_t* logsig_ptr = logsig.data_ptr<scalar_t>();
        scalar_t* P_ptr = P_arr.data_ptr<scalar_t>();

        uint64_t level_offset = 0ULL;
        uint64_t d_pow = 1ULL;

        for (int degree = 1; degree <= depth_i; ++degree) {
            d_pow *= (uint64_t)d_i;

            uint64_t level_size = d_pow;
            const uint64_t* words_at_lvl = nullptr;

            // Only the top level can be compact in your hybrid layout.
            if (degree == depth_i && alternative_projection) {
                const uint64_t full_lvl_size = level_size;
                level_size = (uint64_t)level_sizes[degree];
                const bool non_full = (level_size != full_lvl_size);
                words_at_lvl = non_full ? encoded_words_ptr : nullptr;
            }

            if (level_size == 0ULL) continue;

            const LaunchConfig cfg = make_launch_config(level_size, batch_size, num_windows, 0);

            PATHSIG_DISPATCH_DEGREE(degree, ([&] {
                pathsig::kernels::forward::sig_to_logsig<scalar_t, bits_for_degree(DEG), DEG>
                    <<<cfg.grid, cfg.block, 0, stream>>>(
                        signature_ptr,
                        P_ptr,
                        logsig_ptr,
                        depth_i,
                        d_i,
                        sig_size,
                        level_size,
                        level_offset,
                        words_at_lvl
                    );
            }()));

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            level_offset += level_size;
        }
    });

    return {logsig, P_arr};
}

#undef PATHSIG_DISPATCH_DEGREE

} // namespace pathsig::launch::forward

TORCH_LIBRARY_IMPL(pathsig, CUDA, m) {
    m.impl("compute_signature", &pathsig::launch::forward::computeSignature);
    m.impl("sig_to_logsig", &pathsig::launch::forward::sig_to_logsig);
}

// sig_backward_launch.cu - launch code for backward kernels
#include "sig_backward_launch.cuh"
#include "sig_backward_kernels.cuh"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace pathsig::launch::backward {

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

static inline constexpr uint32_t block_size = 128;

inline size_t backprop_shmem_bytes(int d, size_t elem_size) {
    // shared: shared_path_increment[d] + shared_letter_grads[d * num_warps]
    constexpr uint32_t num_warps = block_size / 32;
    const size_t n_elems = (size_t)d + (size_t)d * (size_t)num_warps;
    return n_elems * elem_size;
}


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


// Signature backward: grad(signature) -> grad(path)
at::Tensor signature_backward(
    const at::Tensor& path_in,                // [B,T,d]
    const at::Tensor& signature_in,           // [B,(W), sig_size]
    const at::Tensor& grad_signature_in,      // same shape as signature_in
    int64_t depth,
    bool alternative_projection,
    const at::Tensor& encoded_words,
    c10::ArrayRef<int64_t> level_sizes,
    bool use_windows,
    const at::Tensor& windows                 // [W,2] int32 CUDA if use_windows else ignored
) {
    TORCH_CHECK(path_in.is_cuda(), "path must be CUDA");
    TORCH_CHECK(path_in.dim() == 3, "path must be [B,T,d]");
    TORCH_CHECK(path_in.scalar_type() == at::kFloat || path_in.scalar_type() == at::kDouble,
                "path must be float32 or float64");

    TORCH_CHECK(signature_in.is_cuda(), "signature must be CUDA");
    TORCH_CHECK(grad_signature_in.is_cuda(), "grad_signature must be CUDA");
    TORCH_CHECK(signature_in.scalar_type() == path_in.scalar_type(),
                "signature must have same dtype as path");
    TORCH_CHECK(grad_signature_in.scalar_type() == path_in.scalar_type(),
                "grad_signature must have same dtype as path");

    TORCH_CHECK(depth >= 1 && depth <= 12, "depth must be in [1,12]");
    const int depth_i = (int)depth;

    if (alternative_projection) {
        TORCH_CHECK(encoded_words.is_cuda(), "encoded_words must be CUDA");
        TORCH_CHECK(encoded_words.scalar_type() == at::kLong, "encoded_words must be int64");
        TORCH_CHECK(encoded_words.is_contiguous(), "encoded_words must be contiguous");
        TORCH_CHECK(encoded_words.dim() == 1, "encoded_words must be 1D");
    }

    c10::cuda::CUDAGuard device_guard(path_in.device());

    const at::Tensor path          = path_in.contiguous();
    const at::Tensor signature     = signature_in.contiguous();
    const at::Tensor grad_signature = grad_signature_in.contiguous();

    const uint32_t batch_size = (uint32_t)path.size(0);
    const int path_len = (int)path.size(1);
    const int d = (int)path.size(2);
    TORCH_CHECK(path_len >= 2, "path must have at least 2 time steps");

    // signature layout
    uint32_t num_windows = 1;
    uint64_t sig_size = 0ULL;

    if (signature.dim() == 3) {
        TORCH_CHECK(grad_signature.dim() == 3, "grad_signature must match signature shape");
        TORCH_CHECK(signature.sizes() == grad_signature.sizes(), "grad_signature must match signature shape");
        num_windows = (uint32_t)signature.size(1);
        sig_size = (uint64_t)signature.size(2);
    } else {
        TORCH_CHECK(signature.dim() == 2, "signature must be [B,S] or [B,W,S]");
        TORCH_CHECK(grad_signature.dim() == 2, "grad_signature must match signature shape");
        TORCH_CHECK(signature.sizes() == grad_signature.sizes(), "grad_signature must match signature shape");
        num_windows = 1;
        sig_size = (uint64_t)signature.size(1);
    }

    // windows
    const int* windows_ptr = nullptr;
    if (use_windows) {
        TORCH_CHECK(windows.is_cuda(), "windows must be CUDA");
        TORCH_CHECK(windows.is_contiguous(), "windows must be contiguous");
        TORCH_CHECK(windows.scalar_type() == at::kInt, "windows must be int32");
        TORCH_CHECK(windows.dim() == 2 && windows.size(1) == 2, "windows must be [W,2]");
        TORCH_CHECK((uint32_t)windows.size(0) == num_windows,
                    "windows.size(0) must match signature num_windows");
        windows_ptr = windows.data_ptr<int>();
    }

    // encoded words base pointer
    const uint64_t* encoded_words_ptr = nullptr;
    if (alternative_projection) {
        encoded_words_ptr = reinterpret_cast<const uint64_t*>(encoded_words.data_ptr<int64_t>());
    }

    // Allocate increment grads (accumulator): [B, T-1, d]
    at::Tensor increment_grads =
        at::zeros({(int64_t)batch_size, (int64_t)(path_len - 1), (int64_t)d}, path.options());

    // output
    at::Tensor path_grad = at::empty_like(path);

    // ReduceByLetterPos heuristic
    const bool reduce_by_letter_pos = (d >= 30);

    // streams: base + (depth-1) workers
    at::cuda::CUDAStream base_stream = at::cuda::getCurrentCUDAStream();
    at::cuda::CUDAEvent inputs_ready;
    inputs_ready.record(base_stream);

    std::array<at::cuda::CUDAStream, 12> worker_streams = {
        base_stream, base_stream, base_stream, base_stream, base_stream, base_stream,
        base_stream, base_stream, base_stream, base_stream, base_stream, base_stream
    };
    for (int i = 1; i < depth_i; ++i) {
        worker_streams[i] = at::cuda::getStreamFromPool(false, base_stream.device_index());
        inputs_ready.block(worker_streams[i]);
    }

    uint64_t level_offset      = 0ULL;
    uint64_t encoded_words_off = 0ULL; // offset for encoded_words (non-full levels only)
    uint64_t d_pow             = 1ULL; // d^degree
    bool recompute_sig         = false;

    AT_DISPATCH_FLOATING_TYPES(path.scalar_type(), "pathsig::signature_backward", [&] {
        const scalar_t* path_ptr     = path.data_ptr<scalar_t>();
        const scalar_t* sig_ptr      = signature.data_ptr<scalar_t>();
        const scalar_t* grad_sig_ptr = grad_signature.data_ptr<scalar_t>();
        scalar_t* inc_grad_ptr       = increment_grads.data_ptr<scalar_t>();
        scalar_t* path_grad_ptr      = path_grad.data_ptr<scalar_t>();

        const size_t shmem = backprop_shmem_bytes(d, sizeof(scalar_t));

        for (int degree = 1; degree <= depth_i; ++degree) {
            d_pow *= (uint64_t)d;

            uint64_t level_size = d_pow;
            if (alternative_projection) {
                level_size = (uint64_t)level_sizes[degree];
            }

            const bool non_full = (level_size != d_pow);
            const uint64_t* words_at_lvl = non_full ? (encoded_words_ptr + encoded_words_off) : nullptr;
            if (non_full) encoded_words_off += level_size;

            if (level_size == 0ULL) {
                recompute_sig = true;
                continue;
            }

            cudaStream_t stream = worker_streams[degree - 1].stream();

            const dim3 block(block_size, 1, 1);
            const uint32_t grid_x = static_cast<uint32_t>((level_size + static_cast<uint64_t>(block_size) - 1ULL) /
                                      static_cast<uint64_t>(block_size));
            const dim3 grid(grid_x, batch_size, num_windows);

            PATHSIG_DISPATCH_DEGREE(degree, ([&] {
                if (reduce_by_letter_pos) {
                    pathsig::kernels::backward::sig_backprop_level<scalar_t, bits_for_degree(DEG), DEG, true>
                        <<<grid, block, shmem, stream>>>(
                            path_ptr,
                            sig_ptr,
                            grad_sig_ptr,
                            inc_grad_ptr,
                            d,
                            path_len,
                            sig_size,
                            level_size,
                            level_offset,
                            words_at_lvl,
                            windows_ptr,
                            recompute_sig
                        );
                } else {
                    pathsig::kernels::backward::sig_backprop_level<scalar_t, bits_for_degree(DEG), DEG, false>
                        <<<grid, block, shmem, stream>>>(
                            path_ptr,
                            sig_ptr,
                            grad_sig_ptr,
                            inc_grad_ptr,
                            d,
                            path_len,
                            sig_size,
                            level_size,
                            level_offset,
                            words_at_lvl,
                            windows_ptr,
                            recompute_sig
                        );
                }
            }()));
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            if (non_full) recompute_sig = true;
            level_offset += level_size;
        }

        // sync workers back to base
        for (int i = 1; i < depth_i; ++i) {
            at::cuda::CUDAEvent done;
            done.record(worker_streams[i]);
            done.block(base_stream);
        }

        // increment_grads -> path_grad on base stream
        const int total = (int)((int64_t)batch_size * (int64_t)path_len * (int64_t)d);
        const int block2 = 256;
        const int grid2  = (total + block2 - 1) / block2;

        pathsig::kernels::backward::increment_grad_to_path_grad<scalar_t>
            <<<grid2, block2, 0, base_stream.stream()>>>(
                inc_grad_ptr,
                path_grad_ptr,
                (int)batch_size,
                path_len,
                d
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return path_grad;
}


// Logsignature backward: grad(logsig) -> grad(signature)
at::Tensor logsig_backward(
    const at::Tensor& sig_in,
    const at::Tensor& P_in,
    const at::Tensor& grad_logsig_in,
    int64_t depth,
    bool alternative_projection,
    const at::Tensor& encoded_words,
    c10::ArrayRef<int64_t> level_sizes
) {
    TORCH_CHECK(sig_in.is_cuda(), "sig must be CUDA");
    TORCH_CHECK(sig_in.scalar_type() == at::kFloat || sig_in.scalar_type() == at::kDouble,
                "sig must be float32 or float64");

    TORCH_CHECK(P_in.is_cuda(), "P must be CUDA");
    TORCH_CHECK(grad_logsig_in.is_cuda(), "grad_logsig must be CUDA");

    TORCH_CHECK(P_in.scalar_type() == sig_in.scalar_type(), "P must have same dtype as sig");
    TORCH_CHECK(grad_logsig_in.scalar_type() == sig_in.scalar_type(), "grad_logsig must have same dtype as sig");

    TORCH_CHECK(depth >= 1 && depth <= 12, "depth must be in [1,12]");
    const int depth_i = (int)depth;

    if (alternative_projection) {
        TORCH_CHECK(encoded_words.is_cuda(), "encoded_words must be CUDA");
        TORCH_CHECK(encoded_words.scalar_type() == at::kLong, "encoded_words must be int64");
        TORCH_CHECK(encoded_words.is_contiguous(), "encoded_words must be contiguous");
        TORCH_CHECK(encoded_words.dim() == 1, "encoded_words must be 1D");
    }

    TORCH_CHECK((int64_t)level_sizes.size() > depth, "level_sizes must have entries up to depth");

    c10::cuda::CUDAGuard device_guard(sig_in.device());

    const at::Tensor sig         = sig_in.contiguous();
    const at::Tensor P           = P_in.contiguous();
    const at::Tensor grad_logsig = grad_logsig_in.contiguous();

    uint32_t batch_size  = 0;
    uint32_t num_windows = 1;
    uint64_t sig_size    = 0ULL;

    if (sig.dim() == 3) {
        TORCH_CHECK(grad_logsig.dim() == 3, "grad_logsig must match sig shape");
        TORCH_CHECK(sig.sizes() == grad_logsig.sizes(), "grad_logsig must match sig shape");
        batch_size  = (uint32_t)sig.size(0);
        num_windows = (uint32_t)sig.size(1);
        sig_size    = (uint64_t)sig.size(2);

        TORCH_CHECK(P.dim() == 3, "P must be [B,W,S*depth] if sig is [B,W,S]");
        TORCH_CHECK((uint32_t)P.size(0) == batch_size && (uint32_t)P.size(1) == num_windows,
                    "P must match [B,W,*]");
    } else {
        TORCH_CHECK(sig.dim() == 2, "sig must be [B,S] or [B,W,S]");
        TORCH_CHECK(grad_logsig.dim() == 2, "grad_logsig must match sig shape");
        TORCH_CHECK(sig.sizes() == grad_logsig.sizes(), "grad_logsig must match sig shape");
        batch_size  = (uint32_t)sig.size(0);
        num_windows = 1;
        sig_size    = (uint64_t)sig.size(1);

        TORCH_CHECK(P.dim() == 2, "P must be [B,S*depth] if sig is [B,S]");
        TORCH_CHECK((uint32_t)P.size(0) == batch_size, "P must match [B,*]");
    }

    const int trunc_lvl = depth_i;

    // infer d from level_sizes[1]
    const int d_i = (int)level_sizes[1];
    TORCH_CHECK(d_i >= 1, "level_sizes[1] must be >= 1 (infers path_dim)");

    const uint64_t P_size = sig_size * (uint64_t)depth_i;

    at::Tensor grad_sig_out;
    at::Tensor gradP_out;

    if (sig.dim() == 3) {
        grad_sig_out = at::zeros({(int64_t)batch_size, (int64_t)num_windows, (int64_t)sig_size}, sig.options());
        gradP_out    = at::zeros({(int64_t)batch_size, (int64_t)num_windows, (int64_t)P_size}, sig.options());
    } else {
        grad_sig_out = at::zeros({(int64_t)batch_size, (int64_t)sig_size}, sig.options());
        gradP_out    = at::zeros({(int64_t)batch_size, (int64_t)P_size}, sig.options());
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const uint64_t* encoded_words_ptr = nullptr;
    if (alternative_projection) {
        encoded_words_ptr = reinterpret_cast<const uint64_t*>(encoded_words.data_ptr<int64_t>());
    }

    // Precompute per-degree sizes/offsets (hybrid layout)
    uint64_t sizes[13]   = {0};
    uint64_t offsets[13] = {0};

    uint64_t off = 0ULL;
    uint64_t d_pow = 1ULL;
    uint64_t full_top = 0ULL;

    for (int k = 1; k <= depth_i; ++k) {
        d_pow *= (uint64_t)d_i;
        if (k == depth_i) full_top = d_pow;

        uint64_t level_size = d_pow;
        if (k == depth_i && alternative_projection) {
            level_size = (uint64_t)level_sizes[k];
        }

        sizes[k]   = level_size;
        offsets[k] = off;
        off += level_size;
    }

    AT_DISPATCH_FLOATING_TYPES(sig.scalar_type(), "pathsig::logsig_backward", [&] {
        const scalar_t* sig_ptr         = sig.data_ptr<scalar_t>();
        const scalar_t* P_ptr           = P.data_ptr<scalar_t>();
        const scalar_t* grad_logsig_ptr = grad_logsig.data_ptr<scalar_t>();
        scalar_t* grad_sig_ptr          = grad_sig_out.data_ptr<scalar_t>();
        scalar_t* gradP_ptr             = gradP_out.data_ptr<scalar_t>();

        for (int degree = depth_i; degree >= 1; --degree) {
            const uint64_t level_size   = sizes[degree];
            const uint64_t level_offset = offsets[degree];
            if (level_size == 0ULL) continue;

            const uint64_t* words_at_lvl = nullptr;
            if (degree == depth_i && alternative_projection) {
                const bool non_full = (level_size != full_top);
                words_at_lvl = non_full ? encoded_words_ptr : nullptr;
            }

            const LaunchConfig cfg = make_launch_config(level_size, batch_size, num_windows, 0);

            PATHSIG_DISPATCH_DEGREE(degree, ([&] {
                pathsig::kernels::backward::logsig_to_sig_grads<scalar_t, bits_for_degree(DEG), DEG>
                    <<<cfg.grid, cfg.block, 0, stream>>>(
                        sig_ptr,
                        P_ptr,
                        grad_logsig_ptr,
                        grad_sig_ptr,
                        gradP_ptr,
                        trunc_lvl,
                        d_i,
                        sig_size,
                        level_size,
                        level_offset,
                        words_at_lvl
                    );
            }()));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });

    return grad_sig_out;
}

#undef PATHSIG_DISPATCH_DEGREE

} // namespace pathsig::launch::backward


// PyTorch registration
TORCH_LIBRARY_IMPL(pathsig, CUDA, m) {
    m.impl("signature_backward", &pathsig::launch::backward::signature_backward);
    m.impl("logsig_backward", &pathsig::launch::backward::logsig_backward);
}

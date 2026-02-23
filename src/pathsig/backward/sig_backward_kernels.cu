// sig_backward_kernels.cu - CUDA kernels for signature/logsignature backpropagation
#include "sig_backward_kernels.cuh"
#include "utils/word_mappings.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace pathsig::kernels::backward
{
/// Constant memory
constexpr int MAX_TRUNC_LEVEL = 12;

// Single-precision constant memory
__constant__ float c_reciprocals_f32[MAX_TRUNC_LEVEL + 1] = {
    0.0f,       // 0 (unused sentinel)
    1.0f,       // 1/1
    0.5f,       // 1/2
    1.0f/3.0f,  // 1/3
    0.25f,      // 1/4
    1.0f/5.0f,  // 1/5
    1.0f/6.0f,  // 1/6
    1.0f/7.0f,  // 1/7
    0.125f,     // 1/8
    1.0f/9.0f,  // 1/9
    1.0f/10.0f, // 1/10
    1.0f/11.0f, // 1/11
    1.0f/12.0f  // 1/12
};

// Double-precision constant memory
__constant__ double c_reciprocals_f64[MAX_TRUNC_LEVEL + 1] = {
    0.0,        // 0 (unused sentinel)
    1.0,        // 1/1
    0.5,        // 1/2
    1.0/3.0,    // 1/3
    0.25,       // 1/4
    1.0/5.0,    // 1/5
    1.0/6.0,    // 1/6
    1.0/7.0,    // 1/7
    0.125,      // 1/8
    1.0/9.0,    // 1/9
    1.0/10.0,   // 1/10
    1.0/11.0,   // 1/11
    1.0/12.0    // 1/12
};

__device__ __forceinline__ float  recip(int n, float)  {return c_reciprocals_f32[n];}
__device__ __forceinline__ double recip(int n, double) {return c_reciprocals_f64[n];}

/// Fused multiply and addition wrappers
__device__ __forceinline__ float fma_wrapper(float a, float b, float c) {
    return fmaf(a, b, c);
}

__device__ __forceinline__ double fma_wrapper(double a, double b, double c) {
    return fma(a, b, c);
}


/// CUDA kernel: signature backward (per-level)
template<typename scalar_t, int bits, int degree, bool reduce_by_pos>
__global__ void __launch_bounds__(128, 4) sig_backprop_level(
    const scalar_t* __restrict__ paths,           // [B, T, d]
    const scalar_t* __restrict__ signatures,      // [B,(W), sig_size]
    const scalar_t* __restrict__ incoming_grads,  // [B,(W), sig_size]
    scalar_t* __restrict__ increment_grads,       // [B, T-1, d] (accumulator)
    const int d,
    const int path_len,
    const uint64_t sig_size,
    const uint64_t level_size,
    const uint64_t level_offset,
    const uint64_t* __restrict__ words_at_lvl,
    const int* __restrict__ windows,           // [W,2] or nullptr
    const bool recompute_sig
) {
    // shared memory layout
    extern __shared__ char shared_mem[];
    scalar_t* shared_path_increment = reinterpret_cast<scalar_t *>(shared_mem);  // [d]
    scalar_t* shared_letter_grads   = shared_path_increment + d;   // [d * num_warps]

    // Assignment of thread to signature term corresponding to a word
    const uint64_t word_idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const bool active = (word_idx < level_size);

    // Get base-d encoding of word and pack the letters of the word
    const uint64_t encoded_word = (words_at_lvl && active) ? words_at_lvl[word_idx] : word_idx;
    const uint64_t letters = word_mappings::packLetters<bits>(encoded_word, degree, d);

    // window selection
    int start = 0;
    int end   = path_len;
    if (windows) {
        start = windows[2 * (int)blockIdx.z + 0];
        end   = windows[2 * (int)blockIdx.z + 1];
    }
    const int steps = end - start - 1;  // #increments in [start, end)

    // base pointers via pointer arithmetic
    const scalar_t* path = paths + (((uint64_t)blockIdx.y * (uint64_t)path_len + (uint64_t)start) * (uint64_t)d);
    const scalar_t* sig = signatures + (((uint64_t)blockIdx.y * (uint64_t)gridDim.z + (uint64_t)blockIdx.z) * (uint64_t)sig_size);
    const scalar_t* grad_in = incoming_grads + (((uint64_t)blockIdx.y * (uint64_t)gridDim.z + (uint64_t)blockIdx.z) * (uint64_t)sig_size);
    scalar_t* inc_grad = increment_grads + ((uint64_t)blockIdx.y * (uint64_t)(path_len - 1) * (uint64_t)d + (uint64_t)start * (uint64_t)d);

    // incoming grad for this word
    const scalar_t grad_val = active ? grad_in[level_offset + word_idx] : scalar_t(0);

    // Find highest matching letter position for words in a warp (alternative reduction method for high d)
    int matching_lvl = 0;
    if constexpr (reduce_by_pos) {
        for (int deg = 1; deg < degree; ++deg) {
            const int ref = __shfl_sync(0xffffffff, word_mappings::unpack<bits>(letters, deg - 1), 0);
            bool same = true;
            if (active && word_mappings::unpack<bits>(letters, deg - 1) != ref) same = false;
            if (!__all_sync(0xffffffff, same)) break;
            matching_lvl++;
        }
    }

    // Register arrays for prefix/suffix values and per-letter grads
    scalar_t pref_sig_vals[degree + 1];
    scalar_t suf_sig_vals[degree + 1];
    scalar_t letter_grads[degree];

    pref_sig_vals[0] = scalar_t(1);
    suf_sig_vals[0]  = scalar_t(1);

    #pragma unroll
    for (int i = 1; i <= degree; ++i) {
        pref_sig_vals[i] = scalar_t(0);
        suf_sig_vals[i]  = scalar_t(0);
    }

    // load/reconstruct prefix sig values up to degree
    if (recompute_sig) {
        for (int t = 0; t < steps; ++t) {
            __syncthreads();
            for (int i = threadIdx.x; i < d; i += blockDim.x) {
                shared_path_increment[i] = path[(t + 1) * d + i] - path[t * d + i];
            }
            __syncthreads();

            if (!active) continue;

            #pragma unroll
            for (int sig_degree = degree-1; sig_degree > 0; --sig_degree) {
                scalar_t h = scalar_t(0);

                #pragma unroll
                for (int k = 0; k <= sig_degree - 1; ++k) {
                    const int letter = word_mappings::unpack<bits>(letters, k);
                    const scalar_t scale = shared_path_increment[letter] * recip(sig_degree - k, scalar_t{});

                    h = scale * (pref_sig_vals[k] + h);
                }
                pref_sig_vals[sig_degree] += h;
            }
        }
    } else if (active) {
        uint64_t off = 0;
        uint64_t d_pow = (uint64_t)d;
        uint64_t pref_word = 0;

        #pragma unroll
        for (int lvl = 1; lvl < degree; ++lvl) {
            const uint64_t letter = (uint64_t)word_mappings::unpack<bits>(letters, lvl - 1);
            pref_word = pref_word * (uint64_t)d + letter;
            pref_sig_vals[lvl] = sig[off + pref_word];
            off += d_pow;
            d_pow *= (uint64_t)d;
        }
    }
    // Load signature value of full-word
    if (active) pref_sig_vals[degree] = sig[level_offset + word_idx];

    // iterate backward over time
    for (int t = steps - 1; t >= 0; --t) {
        #pragma unroll
        for (int i = 0; i < degree; ++i) letter_grads[i] = scalar_t(0);

        // load current increment into shared memory
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            shared_path_increment[i] = path[(t + 1) * d + i] - path[t * d + i];
        }
        __syncthreads();

        // backward prefix signature update
        if (active) {
            #pragma unroll
            for (int sig_degree = degree; sig_degree > 0; --sig_degree) {
                scalar_t h = scalar_t(0);

                #pragma unroll
                for (int k = 0; k <= sig_degree - 1; ++k) {
                    const int letter = word_mappings::unpack<bits>(letters, k);
                    const scalar_t scale = -shared_path_increment[letter] * recip(sig_degree - k, scalar_t{});

                    h = scale * (pref_sig_vals[k] + h);
                }
                pref_sig_vals[sig_degree] += h;
            }
        }

    // Compute and accumulate per-letter grads
    if (active) {
        #pragma unroll
        for (int pref_len = 0; pref_len < degree; ++pref_len) {
            const scalar_t pref_val = pref_sig_vals[pref_len];

            scalar_t prev_prod = scalar_t(1);

            // handle letter_pos = pref_len
            {
                const int letter_pos = pref_len;

                scalar_t temp_prod = prev_prod;
                scalar_t temp_grad = temp_prod * suf_sig_vals[degree - letter_pos - 1];

                int denom_pos = 2;

                #pragma unroll
                for (int pos = letter_pos + 1; pos < degree; ++pos, ++denom_pos) {
                    const int pos_letter = word_mappings::unpack<bits>(letters, pos);
                    temp_prod *= shared_path_increment[pos_letter] * recip(denom_pos, scalar_t{});
                    temp_grad = fma_wrapper(temp_prod, suf_sig_vals[degree - pos - 1], temp_grad);
                }

                letter_grads[letter_pos] = fma_wrapper(temp_grad, pref_val, letter_grads[letter_pos]);
            }

            // remaining letter_pos = pref_len+1 .. degree-1
            int denom_lp = 2;

            #pragma unroll
            for (int letter_pos = pref_len + 1; letter_pos < degree; ++letter_pos, ++denom_lp) {
                const int prev_letter = word_mappings::unpack<bits>(letters, letter_pos - 1);
                prev_prod *= shared_path_increment[prev_letter] * recip(denom_lp, scalar_t{});

                scalar_t temp_prod = prev_prod;
                scalar_t temp_grad = temp_prod * suf_sig_vals[degree - letter_pos - 1];

                int denom_pos = denom_lp + 1;

                #pragma unroll
                for (int pos = letter_pos + 1; pos < degree; ++pos, ++denom_pos) {
                    const int pos_letter = word_mappings::unpack<bits>(letters, pos);
                    temp_prod *= shared_path_increment[pos_letter] * recip(denom_pos, scalar_t{});
                    temp_grad = fma_wrapper(temp_prod, suf_sig_vals[degree - pos - 1], temp_grad);
                }

                letter_grads[letter_pos] = fma_wrapper(temp_grad, pref_val, letter_grads[letter_pos]);
            }
        }
    }


        // warp reduction into shared_letter_grads
        const unsigned warp_id   = (unsigned)(threadIdx.x >> 5);
        const unsigned lane      = (unsigned)(threadIdx.x & 31);
        const unsigned num_warps = (unsigned)(blockDim.x >> 5);

        if constexpr (reduce_by_pos) {
            for (int i = threadIdx.x; i < (int)(d * num_warps); i += blockDim.x) {
                shared_letter_grads[i] = scalar_t(0);
            }
            __syncthreads();
        }

        if constexpr (reduce_by_pos) {
            #pragma unroll
            for (int letter_pos = 0; letter_pos < degree; ++letter_pos) {
                const int letter = word_mappings::unpack<bits>(letters, letter_pos);
                scalar_t val = letter_grads[letter_pos] * grad_val;

                if (letter_pos < matching_lvl) {
                    val += __shfl_down_sync(0xffffffff, val, 16);
                    val += __shfl_down_sync(0xffffffff, val, 8);
                    val += __shfl_down_sync(0xffffffff, val, 4);
                    val += __shfl_down_sync(0xffffffff, val, 2);
                    val += __shfl_down_sync(0xffffffff, val, 1);
                }

                if ((val != scalar_t(0)) && (lane == 0 || letter_pos >= matching_lvl)) {
                    atomicAdd(&shared_letter_grads[letter * num_warps + warp_id], val);
                }
            }
        } else {
            for (int letter = 0; letter < d; ++letter) {
                scalar_t val = scalar_t(0);

                #pragma unroll
                for (int letter_pos = 0; letter_pos < degree; ++letter_pos) {
                    if (word_mappings::unpack<bits>(letters, letter_pos) == letter) val += letter_grads[letter_pos];
                }

                val *= grad_val;

                val += __shfl_down_sync(0xffffffff, val, 16);
                val += __shfl_down_sync(0xffffffff, val, 8);
                val += __shfl_down_sync(0xffffffff, val, 4);
                val += __shfl_down_sync(0xffffffff, val, 2);
                val += __shfl_down_sync(0xffffffff, val, 1);

                if (lane == 0) {
                    shared_letter_grads[letter * num_warps + warp_id] = val;
                }
            }
        }

        __syncthreads();

        // Accumulate and write to global memory
        for (int letter = threadIdx.x; letter < d; letter += blockDim.x) {
            scalar_t sum = scalar_t(0);
            const int base = letter * (int)num_warps;
            #pragma unroll
            for (int w = 0; w < (int)num_warps; ++w) sum += shared_letter_grads[base + w];
            atomicAdd(&inc_grad[(uint64_t)t * (uint64_t)d + (uint64_t)letter], sum);
        }

        // suffix signature update
        if (active) {
            #pragma unroll
            for (int m = degree - 1; m > 0; --m) {
                const int base = degree - m;

                scalar_t h = scalar_t(0);

                #pragma unroll
                for (int p = m; p >= 1; --p) {
                    const int letter_pos = base + (p - 1);
                    const int letter = word_mappings::unpack<bits>(letters, letter_pos);
                    const scalar_t scale = shared_path_increment[letter] * recip(p,scalar_t{});

                    h = scale * (suf_sig_vals[m - p] + h);
                }

                suf_sig_vals[m] = suf_sig_vals[m] + h;
            }
        }

        __syncthreads();
    }
}


/// CUDA kernel: increment gradient -> path gradient
template<typename scalar_t>
__global__ void increment_grad_to_path_grad(
    const scalar_t* __restrict__ inc_grad,  // [B, T-1, d]
    scalar_t* __restrict__ path_grad,       // [B, T, d]
    const int batch_size,
    const int num_time_steps,
    const int d
) {
    const int num_increments = num_time_steps - 1;
    const int total = batch_size * num_time_steps * d;

    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    const int time_stride = num_time_steps * d;
    const int b = idx / time_stride;
    const int rem = idx - b * time_stride;
    const int t = rem / d;
    const int j = rem - t * d;

    const int inc_stride = num_increments * d;
    const int base = b * inc_stride + j;

    if (t == 0) {
        path_grad[idx] = -inc_grad[base];
    } else if (t == num_time_steps - 1) {
        path_grad[idx] = inc_grad[base + (num_increments - 1) * d];
    } else {
        path_grad[idx] = inc_grad[base + (t - 1) * d] - inc_grad[base + t * d];
    }
}


/// CUDA kernel: logsig grad -> signature grad
template<typename scalar_t, int bits, int degree>
__global__ void logsig_to_sig_grads(
    const scalar_t* __restrict__ sig_arr,          // [B,(W), total_sig_size]
    const scalar_t* __restrict__ P_arr,            // [B,(W), total_sig_size * trunc_lvl]
    const scalar_t* __restrict__ grad_logsig_arr,  // [B,(W), total_sig_size]
    scalar_t* __restrict__ grad_sig_arr,           // [B,(W), total_sig_size] (accumulator)
    scalar_t* __restrict__ gradP_arr,              // [B,(W), total_sig_size * trunc_lvl] (accumulator)
    const int trunc_lvl,
    const int d,
    const uint64_t total_sig_size,
    const uint64_t level_size,
    const uint64_t level_offset,
    const uint64_t* __restrict__ words_at_lvl // [sig_size, 1] or nullptr
) {
    // Assignment of thread to signature term corresponding to a word
    const uint64_t word_idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const bool active = word_idx < level_size;
    if (!active) return;

    // Get base-d encoding of word and pack the letters of the word
    const uint64_t encoded_word = (words_at_lvl) ? words_at_lvl[word_idx] : word_idx;
    const uint64_t letters = word_mappings::packLetters<bits>(encoded_word, degree, d);

    // Base pointers via pointer arithmetic
    const scalar_t* sig = sig_arr
        + (((uint64_t)blockIdx.y * (uint64_t)gridDim.z + (uint64_t)blockIdx.z) * (uint64_t)total_sig_size);

    const scalar_t* P = P_arr
        + (((uint64_t)blockIdx.y * (uint64_t)gridDim.z + (uint64_t)blockIdx.z)
           * (uint64_t)(total_sig_size * (uint64_t)trunc_lvl));

    const scalar_t* grad_logsig = grad_logsig_arr
        + (((uint64_t)blockIdx.y * (uint64_t)gridDim.z + (uint64_t)blockIdx.z) * (uint64_t)total_sig_size);

    scalar_t* grad_sig = grad_sig_arr
        + (((uint64_t)blockIdx.y * (uint64_t)gridDim.z + (uint64_t)blockIdx.z) * (uint64_t)total_sig_size);

    scalar_t* gradP = gradP_arr
        + (((uint64_t)blockIdx.y * (uint64_t)gridDim.z + (uint64_t)blockIdx.z)
           * (uint64_t)(total_sig_size * (uint64_t)trunc_lvl));

    // hybrid/global index of this word
    const uint64_t w_global = level_offset + word_idx;
    const uint64_t gradP_row_off = w_global * (uint64_t)trunc_lvl;

    // Register arrays
    scalar_t    gradP_vals[degree];     // grad for P_m(w) stored at index (m-1)
    uint64_t pref_inds[degree + 1];  // dense indices for prefixes (length 1..degree-1)
    uint64_t suf_inds[degree + 1];   // dense indices for suffixes (length 1..degree-1)

    // build prefix/suffix indices for lengths < degree
    uint64_t level_off = 0;
    uint64_t d_pow     = 1;
    uint64_t pref_word = 0;

    #pragma unroll
    for (int i = 1; i < degree; ++i) {
        d_pow *= (uint64_t)d;

        pref_word = pref_word * (uint64_t)d + (uint64_t)word_mappings::unpack<bits>(letters, i - 1);
        pref_inds[i] = pref_word + level_off;

        const uint64_t suf_word = encoded_word % d_pow;
        suf_inds[i] = level_off + suf_word;

        level_off += d_pow;
    }

    const scalar_t grad_logsig_w = grad_logsig[w_global];
    #pragma unroll
    for (int m = 1; m <= degree; ++m) {
        const scalar_t coeff_m = (m & 1) ? recip(m,scalar_t{}) : -recip(m,scalar_t{});
        const scalar_t acc = gradP[gradP_row_off + (uint64_t)(m - 1)];
        gradP_vals[m - 1] = fma_wrapper(coeff_m, grad_logsig_w, acc);
    }

    // base: P_1(w) = S_w
    atomicAdd(&grad_sig[w_global], gradP_vals[0]);

    #pragma unroll
    for (int m = 2; m <= degree; ++m) {
        const scalar_t gPm = gradP_vals[m - 1];

        #pragma unroll
        for (int k = m - 1; k < degree; ++k) {
            const uint64_t prefix_idx = pref_inds[k];
            const int suffix_len = degree - k;
            const uint64_t suffix_idx = suf_inds[suffix_len];

            const scalar_t suffix_sig = sig[suffix_idx];
            const scalar_t prefix_P   = P[prefix_idx * (uint64_t)trunc_lvl + (uint64_t)(m - 2)];

            atomicAdd(&gradP[prefix_idx * (uint64_t)trunc_lvl + (uint64_t)(m - 2)], gPm * suffix_sig);
            atomicAdd(&grad_sig[suffix_idx], gPm * prefix_P);
        }
    }
}

/// Explicit template instantiations
constexpr int bits_for_degree(int deg) { return (deg == 1) ? 32 : (64 / deg); }

#define FOR_EACH_DEGREE(M) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) M(10) M(11) M(12)

#define INSTANTIATE_SIG_BACKPROP_TYPE(T, N)                                                \
    template __global__ void sig_backprop_level<T, bits_for_degree(N), (N), true>(         \
        const T* __restrict__,                                                             \
        const T* __restrict__,                                                             \
        const T* __restrict__,                                                             \
        T* __restrict__,                                                                   \
        const int, const int,                                                              \
        const uint64_t, const uint64_t, const uint64_t,                                    \
        const uint64_t* __restrict__,                                                      \
        const int* __restrict__,                                                           \
        const bool);                                                                       \
    template __global__ void sig_backprop_level<T, bits_for_degree(N), (N), false>(        \
        const T* __restrict__,                                                             \
        const T* __restrict__,                                                             \
        const T* __restrict__,                                                             \
        T* __restrict__,                                                                   \
        const int, const int,                                                              \
        const uint64_t, const uint64_t, const uint64_t,                                    \
        const uint64_t* __restrict__,                                                      \
        const int* __restrict__,                                                           \
        const bool);

#define INSTANTIATE_SIG_BACKPROP(N)            \
    INSTANTIATE_SIG_BACKPROP_TYPE(float, N)   \
    INSTANTIATE_SIG_BACKPROP_TYPE(double, N)

FOR_EACH_DEGREE(INSTANTIATE_SIG_BACKPROP)
#undef INSTANTIATE_SIG_BACKPROP
#undef INSTANTIATE_SIG_BACKPROP_TYPE

// increment_grad_to_path_grad
template __global__ void increment_grad_to_path_grad<float>(
    const float* __restrict__, float* __restrict__, const int, const int, const int);
template __global__ void increment_grad_to_path_grad<double>(
    const double* __restrict__, double* __restrict__, const int, const int, const int);

// logsig_to_sig_grads
#define INSTANTIATE_LOGSIG_TO_SIG_GRADS_TYPE(T, N)                                          \
    template __global__ void logsig_to_sig_grads<T, bits_for_degree(N), (N)>(               \
        const T* __restrict__,                                                              \
        const T* __restrict__,                                                              \
        const T* __restrict__,                                                              \
        T* __restrict__,                                                                    \
        T* __restrict__,                                                                    \
        const int, const int,                                                               \
        const uint64_t, const uint64_t, const uint64_t,                                     \
        const uint64_t* __restrict__);

#define INSTANTIATE_LOGSIG_TO_SIG_GRADS(N)            \
    INSTANTIATE_LOGSIG_TO_SIG_GRADS_TYPE(float, N)   \
    INSTANTIATE_LOGSIG_TO_SIG_GRADS_TYPE(double, N)

FOR_EACH_DEGREE(INSTANTIATE_LOGSIG_TO_SIG_GRADS)
#undef INSTANTIATE_LOGSIG_TO_SIG_GRADS
#undef INSTANTIATE_LOGSIG_TO_SIG_GRADS_TYPE

#undef FOR_EACH_DEGREE
} // namespace pathsig::kernels::backward

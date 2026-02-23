// sig_kernels.cu - CUDA kernels for signature computation
#include "sig_kernels.cuh"
#include "utils/word_mappings.cuh"
#include <cuda_runtime.h>
#include <cstdint>


namespace pathsig::kernels::forward
{
/// Constant Memory
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

__device__ __forceinline__ float  recip(int n, float)  { return c_reciprocals_f32[n]; }
__device__ __forceinline__ double recip(int n, double) { return c_reciprocals_f64[n]; }

/// Device helper: Kahan summation for improved numerical stability
__forceinline__ __device__ void kahanAdd(float& sum, float to_add, float& comp)
{
    float y = __fsub_rn(to_add, comp);
    float temp = __fadd_rn(sum, y);
    comp = __fsub_rn(__fsub_rn(temp, sum), y);
    sum = temp;
}

// Overloaded Kahan summation for double
__forceinline__ __device__ void kahanAdd(double& sum, double to_add, double& comp)
{
    double y = __dsub_rn(to_add, comp);
    double temp = __dadd_rn(sum, y);
    comp = __dsub_rn(__dsub_rn(temp, sum), y);
    sum = temp;
}

/// Fused multiply and addition wrappers
__device__ __forceinline__ float fma_wrapper(float a, float b, float c) {
    return fmaf(a, b, c);
}

__device__ __forceinline__ double fma_wrapper(double a, double b, double c) {
    return fma(a, b, c);
}


/// CUDA kernel: compute the signature coefficients for a level
template<typename scalar_t, int bits, int degree> __global__ void compute_signature_level(
    const scalar_t* __restrict__ paths, // [B, T, d]
    scalar_t* __restrict__ signatures, // [B,(W), sig_size]
    const int d,
    const int path_len,
    const uint64_t sig_size,
    const uint64_t level_size,
    const uint64_t level_offset,
    const uint64_t* __restrict__ words_at_lvl, // [sig_size, 1] or nullptr
    const int* __restrict__  windows // [W, 2] or nullptr
) {
    // shared memory layout
    extern __shared__ char smem[];
    scalar_t * shared_path_increment = reinterpret_cast<scalar_t *>(smem); // [d]

    // Assignemnt of thread to signature term corresponding to a word
    const uint64_t word_idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const bool active = word_idx < level_size;

    // Get base-d encoding of word and pack the letters of the word
    const uint64_t encoded_word = (words_at_lvl && active) ? words_at_lvl[word_idx] : word_idx;
    const uint64_t letters = word_mappings::packLetters<bits>(encoded_word, degree, d);

    // Register arrays for prefix signature terms and their kahan compensators
    scalar_t pref_sig_vals[degree + 1];
    scalar_t kahan_comps[degree + 1];

    // Initialize prefix signature values
    #pragma unroll
    for (int i = 1; i <= degree; ++i) {
        pref_sig_vals[i] = scalar_t(0);
        kahan_comps[i] = scalar_t(0);
    }
    pref_sig_vals[0] = scalar_t(1); // signature value of empty word
    kahan_comps[0] = scalar_t(0);

    // window selection
    int start = 0;
    int end   = path_len;
    if (windows) {
        start = windows[2 * (int)blockIdx.z + 0];
        end   = windows[2 * (int)blockIdx.z + 1];
    }
    const int steps = end - start - 1;  // #increments in [start,end)

    // Pointers to path and signature level for batch (and possibly window) via pointer arithmetic
    const scalar_t* path = paths + (((uint64_t)blockIdx.y * (uint64_t)path_len + (uint64_t)start) * (uint64_t)d);
    scalar_t* sig_level = signatures + (((uint64_t)blockIdx.y * (uint64_t)gridDim.z + (uint64_t)blockIdx.z) * (uint64_t)sig_size) + level_offset;

    // Loop over each time step
    for (int t = 0; t < steps; ++t) {
        __syncthreads();
        // Compute current path increments and write it to shared memory
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            shared_path_increment[i] = path[(t + 1) * d + i] - path[t * d + i];
        }
        __syncthreads();
        if (!active) continue;

        #pragma unroll
        for (int sig_degree = degree; sig_degree > 0; --sig_degree) {
            scalar_t h = scalar_t(0);

            #pragma unroll
            for (int k = 0; k <= sig_degree - 1; ++k) {
                const int letter = word_mappings::unpack<bits>(letters, k);
                const scalar_t scale = shared_path_increment[letter] * recip(sig_degree - k, scalar_t{});

                h = scale * (pref_sig_vals[k] + h);
            }

            // Kahan summation for accumulation over time steps
            kahanAdd(pref_sig_vals[sig_degree], h, kahan_comps[sig_degree]);
        }

    }
    // Write computed signature value to global memory
    if (active) sig_level[word_idx] = pref_sig_vals[degree];
}


/// CUDA kernel: convert signature coefficients to log-signature coefficients
template<typename scalar_t, int bits, int degree>
__global__ void sig_to_logsig(
    const scalar_t* __restrict__ signatures,   // [B,(W), sig_size]
    scalar_t* __restrict__ P_arr,              // [B,(W), sig_size * trunc_lvl]
    scalar_t* __restrict__ logsig_arr,         // [B,(W), sig_size]
    const int trunc_lvl,
    const int d,
    const uint64_t sig_size,
    const uint64_t level_size,
    const uint64_t level_offset,
    const uint64_t* __restrict__ words_at_lvl // [sig_size, 1] or nullptr
) {
    // Assignemnt of thread to signature term corresponding to a word
    const uint64_t word_idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const bool active = word_idx < level_size;
    if (!active) return;

    // Get base-d encoding of word and pack the letters of the word
    const uint64_t encoded_word = (words_at_lvl && active) ? words_at_lvl[word_idx] : word_idx;
    const uint64_t letters = word_mappings::packLetters<bits>(encoded_word, degree, d);

    // Base pointers via pointer arithmetic
    const scalar_t* sig = signatures + (((uint64_t)blockIdx.y * (uint64_t)gridDim.z + (uint64_t)blockIdx.z) * (uint64_t)sig_size);
    scalar_t* P = P_arr + (((uint64_t)blockIdx.y * (uint64_t)gridDim.z + (uint64_t)blockIdx.z) * ((uint64_t)sig_size * (uint64_t)trunc_lvl));
    scalar_t* logsig = logsig_arr + (((uint64_t)blockIdx.y * (uint64_t)gridDim.z + (uint64_t)blockIdx.z) * (uint64_t)sig_size);

	// Register arrays for suffix signature values of the word and indices of prefix signature terms
    scalar_t    suf_sig_vals[degree + 1];
    uint32_t pref_inds[degree + 1];

    // Getting suffix signature values and computing indices of prefix words
    uint64_t level_off = 0;
    uint64_t d_pow     = 1;
    uint64_t pref_word = 0;

    #pragma unroll
    for (int i = 1; i < degree; ++i) {
        d_pow *= (uint64_t)d;

        pref_word = pref_word * (uint64_t)d + (uint64_t)word_mappings::unpack<bits>(letters, i - 1);
        pref_inds[i] = pref_word + level_off;

        const uint64_t suf_word = encoded_word % d_pow;
        suf_sig_vals[i] = sig[level_off + suf_word];

        level_off += d_pow;
    }
    // full-word signature index
    pref_inds[degree] = level_offset + word_idx;

    // full-word signature coefficient
    suf_sig_vals[degree] = sig[pref_inds[degree]];

    // base: P_1(w) = S_w
    const uint64_t P_off = pref_inds[degree] * (uint64_t)trunc_lvl;
    P[P_off + 0] = suf_sig_vals[degree];
	scalar_t log_sig_w = suf_sig_vals[degree];

    // compute P_m(w) for m=2..degree
    #pragma unroll
    for (int m = 2; m <= degree; ++m) {
        scalar_t p_val = scalar_t(0);

        #pragma unroll
        for (int k = m - 1; k < degree; ++k) {
            // k = prefix length
            p_val = fma_wrapper(
                P[pref_inds[k] * (uint64_t)trunc_lvl + (uint64_t)(m - 2)], // P_{m-1}(prefix)
                suf_sig_vals[degree - k],                                  // S(suffix)
                p_val
            );
        }
		// Acummulate into log signature value
		const scalar_t coeff = (m & 1) ? recip(m, scalar_t{}) : -recip(m, scalar_t{});
        log_sig_w = fma_wrapper(coeff, p_val, log_sig_w);
        // store P_m(w)
        P[P_off + (uint64_t)(m - 1)] = p_val;
    }
	// Store log signature value for word
	logsig[pref_inds[degree]] = log_sig_w;
}


/// Explicit template instantiations
constexpr int bits_for_degree(int deg) { return (deg == 1) ? 32 : (64 / deg); }

#define INSTANTIATE_SIG_TYPE(T, N) \
template __global__ void compute_signature_level<T, bits_for_degree(N), (N)>( \
const T*, T*, const int, const int, const uint64_t, const uint64_t, const uint64_t, const uint64_t*, const int*);

#define INSTANTIATE_SIG2LOGSIG_TYPE(T, N) \
template __global__ void sig_to_logsig<T, bits_for_degree(N), (N)>( \
const T*, T*, T*, const int, const int, const uint64_t, const uint64_t, const uint64_t, const uint64_t*);

#define FOR_EACH_DEGREE(M, T) \
M(T, 1)  M(T, 2)  M(T, 3)  M(T, 4)  M(T, 5)  M(T, 6) \
M(T, 7)  M(T, 8)  M(T, 9)  M(T, 10) M(T, 11) M(T, 12)

FOR_EACH_DEGREE(INSTANTIATE_SIG_TYPE, float)
FOR_EACH_DEGREE(INSTANTIATE_SIG2LOGSIG_TYPE, float)
FOR_EACH_DEGREE(INSTANTIATE_SIG_TYPE, double)
FOR_EACH_DEGREE(INSTANTIATE_SIG2LOGSIG_TYPE, double)

#undef INSTANTIATE_SIG_TYPE
#undef INSTANTIATE_SIG2LOGSIG_TYPE
#undef FOR_EACH_DEGREE

} // namespace pathsig::kernels::forward
// sig_backprop.cu - CUDA signature backpropagation kernels
#include "sig_backprop.cuh"
#include "word_mappings.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;


namespace pathsig {
namespace sig_backprop {

/// Constant Memory
constexpr int MAX_TRUNC_LEVEL = 12;

// Double-precision constant memory
__constant__ double c_reciprocals[MAX_TRUNC_LEVEL + 1] = {
    0.0,        // 0 (sentinel)
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

// Single-precision constant memory
__constant__ float c_reciprocals_f32[MAX_TRUNC_LEVEL + 1] = {
    0.0f,       // 0 (sentinel)
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


/// Macros used to lower register usage
// Shared memory access macros
#define SHARED_PATH_INCREMENTS(base, idx) ((base)[(idx)])
#define SHARED_PARTIAL_SIG(base, idx) ((base)[shmem_partial_sig_offset + (idx)])
#define SHARED_COMMON_TERMS(base, idx) ((base)[shmem_common_terms_offset + (idx)])
#define SHARED_LETTER_GRADS(base, idx) ((base)[shmem_letter_grads_offset + (idx)])
#define SIG_BATCH_OFFSET (total_sig_size * blockIdx.y)
#define INC_BATCH_OFFSET (num_time_steps * d * blockIdx.y)


/// Double precision signature backpropagation
template<int bits, int TRUNC_LVL>
__global__ void __launch_bounds__(1024, 1) signatureBackwardPass(
    const double* __restrict__ path_increments,
    const double* __restrict__ signature,
    const double* __restrict__ incoming_gradients,
    double* __restrict__ increment_grads,
    const uint64_t* __restrict__ d_powers,
    const uint64_t* __restrict__ level_offsets,
    int d,
    int fixed_len,
    int num_time_steps,
    uint64_t total_sig_size,
    const unsigned* __restrict__ degrees,
    const unsigned* __restrict__ varying_words,
    const uint64_t* __restrict__ prefix_maps,
    int num_lower)
{
    // Thread and block information
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const uint64_t fixed_word = blockIdx.x;

    // Pack warp info into single register to reduce pressure
    const unsigned wid = threadIdx.x / warpSize;
    const unsigned num_warps = (blockDim.x + warpSize - 1) / warpSize;
    const unsigned my_degree = degrees[threadIdx.x];

    // Single shared memory pointer accessed via offsets
    extern __shared__ double shmem[];

    // Compute shared memory offsets
    int shmem_partial_sig_offset = d;
    int shmem_common_terms_offset = d + num_lower;
    int shmem_letter_grads_offset = d + num_lower + num_warps * num_lower;

    // Load per-thread data from constants
    const uint64_t my_prefix_map = prefix_maps[threadIdx.x];

    // Build full word
    uint64_t my_word;
    if (my_degree > fixed_len) {
        my_word = fixed_word * d_powers[my_degree - fixed_len] + varying_words[threadIdx.x];
    } else {
        my_word = fixed_word;
        // Shave off letters for my_degree < fixed_len
        for (int len = fixed_len; len > my_degree; --len) {
            my_word /= d;
        }
    }


    // Pack letters from back and rightmost_mask
    const uint64_t my_letters_from_back = word_mappings::packLettersFromBack<bits>(
        my_word, my_degree, d);
    const unsigned rightmost_mask = word_mappings::packRightmost<bits>(
        my_letters_from_back, my_degree);

    // Check if word should contribute (i.e. that it is the canonical one if appearing in multiple blocks)
    const bool has_contribution = (my_degree >= fixed_len || (fixed_word % d_powers[fixed_len - my_degree])==0);

    // Load incoming gradient for this word
    const double my_grad_val = (has_contribution) ? incoming_gradients[
        SIG_BATCH_OFFSET + level_offsets[my_degree] + my_word] : 0.0;

    // Per-thread array for S_{t,T}(X, suffix)
    double sig_suffix_vals[TRUNC_LVL];

    // Initialize array
    sig_suffix_vals[0] = 1.0;  // identity element
    #pragma unroll 1
    for (int k = 1; k < TRUNC_LVL; ++k) {
        sig_suffix_vals[k] = 0.0;
    }

    // Get time T signature value of word assigned to thread
    double my_sig_val = signature[
        SIG_BATCH_OFFSET + level_offsets[my_degree] + my_word];

    // Share time T signature value
    if (threadIdx.x < num_lower) {
        SHARED_PARTIAL_SIG(shmem, threadIdx.x) = my_sig_val;
    }

    for (int t = num_time_steps - 1; t >= 0; --t) {
        __syncthreads();

        // Load path increments
        if (threadIdx.x < d) {
            SHARED_PATH_INCREMENTS(shmem, threadIdx.x) =
                path_increments[INC_BATCH_OFFSET + t * d + threadIdx.x];
        }

        // Initialization of COMMON_TERMS and LETTER_GRADS
        const unsigned max_size = max(num_lower * num_warps, d * num_warps);
        for (unsigned i = threadIdx.x; i < max_size; i += blockDim.x) {
            if (i < num_lower * num_warps)
                SHARED_COMMON_TERMS(shmem, i) = 0.0;
            if (i < d * num_warps)
                SHARED_LETTER_GRADS(shmem, i) = 0.0;
        }
        __syncthreads();

        // Step 1: Update S_{0,t}(X, my_word)
        double running_prod = 1.0;
        double temp_sum = 0.0;

        // Sum over all prefix-suffix splits (except full prefix and full suffix)
        for (unsigned suffix_len = 1; suffix_len < my_degree; ++suffix_len) {
            // Update running suffix product
            running_prod *= -SHARED_PATH_INCREMENTS(shmem,
                word_mappings::unpack<bits>(my_letters_from_back, suffix_len - 1));
            running_prod *= c_reciprocals[suffix_len];

            // Add contribution from this split
            temp_sum = __fma_rn(
                SHARED_PARTIAL_SIG(shmem,
                    word_mappings::unpack<bits>(my_prefix_map, my_degree - suffix_len - 1)),
                running_prod,
                temp_sum);
        }

        // Add final terms (full prefix and full suffix)
        running_prod *= -SHARED_PATH_INCREMENTS(shmem,
            word_mappings::unpack<bits>(my_letters_from_back, my_degree - 1)) *
            c_reciprocals[my_degree];
        temp_sum += running_prod;
        my_sig_val = t == 0 ? 0.0 : (my_sig_val + temp_sum);

        // Store S_{0,t} in shared memory for steps 2, and 3 (and step 1 in the next time)
        __syncthreads();
        if (threadIdx.x < num_lower) {
            SHARED_PARTIAL_SIG(shmem, threadIdx.x) = my_sig_val;  // S_{0,t}
        }

        // Step 2: Compute common factors CF_t(*)
        double my_common_factor = my_grad_val;  // my_degree term of CF_t(*)

        // Common factors CF_t(*) for words at levels < TRUNC_LVL
        #pragma unroll 1
        for (int lvl = TRUNC_LVL - 1; lvl > 0; --lvl) {
            const bool contributes = (lvl < my_degree);
            unsigned pref_tid = 1024;
            if (contributes) {
                pref_tid = word_mappings::unpack<bits>(my_prefix_map, lvl - 1);
            }

            // Warp-level reduction
            warp.sync();
            auto group = cg::labeled_partition(warp, pref_tid);
            double contrib = (contributes) ?
                sig_suffix_vals[my_degree - lvl] * my_grad_val : 0.0;
            double sum = cg::reduce(group, contrib, cg::plus<double>());

            if (pref_tid != 1024 && group.thread_rank() == 0) {
                atomicAdd(&SHARED_COMMON_TERMS(shmem, pref_tid * num_warps + wid), sum);
            }
        }
        __syncthreads();

        // Accumulate common factor from higher degree terms
        if (my_degree < TRUNC_LVL) {
            for (int i = 0; i < num_warps; ++i) {
                my_common_factor += SHARED_COMMON_TERMS(shmem, threadIdx.x * num_warps + i);
            }
        }

        // Step 3: Compute dS_{0,T}(w)/dΔX_t * CF_t(w)
        running_prod = 1.0;

        #pragma unroll 1
        for (int pref_len = TRUNC_LVL - 1; pref_len >= 0; --pref_len) {
            const unsigned suf_len = (my_degree > pref_len) ?
                my_degree - pref_len : my_degree;

            // Update running_prod with prior increment
            if (suf_len >= 2 && pref_len < my_degree) {
                running_prod *= SHARED_PATH_INCREMENTS(shmem,
                    word_mappings::unpack<bits>(my_letters_from_back, suf_len - 2)) *
                    c_reciprocals[suf_len - 1];
            }

            // Process only if this is the rightmost occurrence
            const bool is_rightmost = (pref_len < my_degree) &&
                (((rightmost_mask >> (suf_len - 1)) & 1ULL) != 0);

            // Get current suffix letter
            unsigned suffix_letter = (is_rightmost) ?
                word_mappings::unpack<bits>(my_letters_from_back, suf_len - 1) : 1024;

            // Initialize gradient value
            double current_grad_val = 0.0;

            if (is_rightmost) {
                double base_suffix = running_prod * c_reciprocals[suf_len];

                // First term contribution
                current_grad_val = (suf_len != my_degree) ?
                    base_suffix * SHARED_PARTIAL_SIG(shmem,
                        word_mappings::unpack<bits>(my_prefix_map, pref_len - 1)) :
                    base_suffix;

                // Process remaining terms
                int count = 1;
                #pragma unroll 1
                for (unsigned tmp_suf_len = suf_len + 1; tmp_suf_len <= my_degree; ++tmp_suf_len) {
                    uint8_t letter = word_mappings::unpack<bits>(
                        my_letters_from_back, tmp_suf_len - 1);
                    base_suffix *= SHARED_PATH_INCREMENTS(shmem, letter) *
                                  c_reciprocals[tmp_suf_len];
                    count += (letter == suffix_letter);

                    if (tmp_suf_len != my_degree) {
                        current_grad_val += count * base_suffix *
                            SHARED_PARTIAL_SIG(shmem,
                                word_mappings::unpack<bits>(my_prefix_map, my_degree - tmp_suf_len - 1));
                    } else {
                        current_grad_val += count * base_suffix;
                    }
                }
            }

            current_grad_val *= my_common_factor;

            // Use labeled reduce for accumulation (scoped to limit registers)
            {
                auto group = cg::labeled_partition(warp, suffix_letter);
                double v_sum = cg::reduce(group, current_grad_val, cg::plus<double>());
                if (suffix_letter < 1024 && group.thread_rank() == 0) {
                    atomicAdd(&SHARED_LETTER_GRADS(shmem, suffix_letter * num_warps + wid), v_sum);
                }
            }
        }
        __syncthreads();

        // Step 4: Reduce gradient contributions and atomic add to global memory
        if (threadIdx.x < d) {
            double letter_grad = 0.0;
            for (int w = 0; w < num_warps; ++w) {
                letter_grad += SHARED_LETTER_GRADS(shmem, threadIdx.x * num_warps + w);
            }
            if (letter_grad != 0.0) {
                atomicAdd(&increment_grads[INC_BATCH_OFFSET + t * d + threadIdx.x], letter_grad);
            }
        }

        // Step 5: Update S_{t,T}(X, suffix) for next time step (in-place)
        #pragma unroll 1
        for (int sig_degree = my_degree - 1; sig_degree > 0; --sig_degree) {
            running_prod = 1.0;
            temp_sum = 0.0;

            #pragma unroll 1
            for (int pref_len = 1; pref_len <= sig_degree; ++pref_len) {
                running_prod *= SHARED_PATH_INCREMENTS(shmem,
                    word_mappings::unpack<bits>(my_letters_from_back, sig_degree - pref_len)) *
                    c_reciprocals[pref_len];
                temp_sum += running_prod * sig_suffix_vals[sig_degree - pref_len];
            }
            sig_suffix_vals[sig_degree] += temp_sum;
        }
    }  // End time loop
}  // function signatureBackwardPass (double)


// Single-precision signature backpropagation
template<int bits, int TRUNC_LVL>
__global__ void __launch_bounds__(1024, 1) signatureBackwardPass(
    const float* __restrict__ path_increments,
    const float* __restrict__ signature,
    const float* __restrict__ incoming_gradients,
    float* __restrict__ increment_grads,
    const uint64_t* __restrict__ d_powers,
    const uint64_t* __restrict__ level_offsets,
    int d,
    int fixed_len,
    int num_time_steps,
    uint64_t total_sig_size,
    const unsigned* __restrict__ degrees,
    const unsigned* __restrict__ varying_words,
    const uint64_t* __restrict__ prefix_maps,
    int num_lower)
{
    // Thread and block information
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const uint64_t fixed_word = blockIdx.x;

    // Pack warp info into single register to reduce pressure
    const unsigned wid = threadIdx.x / warpSize;
    const unsigned num_warps = (blockDim.x + warpSize - 1) / warpSize;
    const unsigned my_degree = degrees[threadIdx.x];

    // Single shared memory pointer accessed via offsets
    extern __shared__ float shmem_FP32[];

    // Compute shared memory offsets
    int shmem_partial_sig_offset = d;
    int shmem_common_terms_offset = d + num_lower;
    int shmem_letter_grads_offset = d + num_lower + num_warps * num_lower;

    // Load per-thread data from constants
    const uint64_t my_prefix_map = prefix_maps[threadIdx.x];

    // Build full word
    uint64_t my_word;
    if (my_degree > fixed_len) {
        my_word = fixed_word * d_powers[my_degree - fixed_len] + varying_words[threadIdx.x];
    } else {
        my_word = fixed_word;
        // Shave off letters for my_degree < fixed_len
        for (int len = fixed_len; len > my_degree; --len) {
            my_word /= d;
        }
    }

    // Pack letters from back and rightmost_mask
    const uint64_t my_letters_from_back = word_mappings::packLettersFromBack<bits>(
        my_word, my_degree, d);
    const unsigned rightmost_mask = word_mappings::packRightmost<bits>(
        my_letters_from_back, my_degree);

    // Load incoming gradient for this word
    const float my_grad_val = (my_degree >= fixed_len || (fixed_word % d_powers[fixed_len - my_degree])==0) ? incoming_gradients[
        SIG_BATCH_OFFSET + level_offsets[my_degree] + my_word] : 0.0f;

    // Per-thread array for S_{t,T}(X, suffix)
    float sig_suffix_vals[TRUNC_LVL];

    // Initialize array
    sig_suffix_vals[0] = 1.0f;  // identity element
    #pragma unroll 1
    for (int k = 1; k < TRUNC_LVL; ++k) {
        sig_suffix_vals[k] = 0.0f;
    }

    // Get time T signature value of word assigned to thread
    float my_sig_val = signature[
        SIG_BATCH_OFFSET + level_offsets[my_degree] + my_word];

    // Share time T signature value
    if (threadIdx.x < num_lower) {
        SHARED_PARTIAL_SIG(shmem_FP32, threadIdx.x) = my_sig_val;
    }

    for (int t = num_time_steps - 1; t >= 0; --t) {
        __syncthreads();

        // Load path increments
        if (threadIdx.x < d) {
            SHARED_PATH_INCREMENTS(shmem_FP32, threadIdx.x) =
                path_increments[INC_BATCH_OFFSET + t * d + threadIdx.x];
        }

        // Fused initialization of COMMON_TERMS and LETTER_GRADS
        const unsigned max_size = max(num_lower * num_warps, d * num_warps);
        for (unsigned i = threadIdx.x; i < max_size; i += blockDim.x) {
            if (i < num_lower * num_warps)
                SHARED_COMMON_TERMS(shmem_FP32, i) = 0.0f;
            if (i < d * num_warps)
                SHARED_LETTER_GRADS(shmem_FP32, i) = 0.0f;
        }
        __syncthreads();

        // Step 1: Update S_{0,t}(X, my_word)
        float running_prod = 1.0f;
        float temp_sum = 0.0f;

        // Sum over all prefix-suffix splits (except full prefix and full suffix)
        for (unsigned suffix_len = 1; suffix_len < my_degree; ++suffix_len) {
            // Update running suffix product
            running_prod *= -SHARED_PATH_INCREMENTS(shmem_FP32,
                word_mappings::unpack<bits>(my_letters_from_back, suffix_len - 1));
            running_prod *= c_reciprocals_f32[suffix_len];

            // Add contribution from this split
            temp_sum = __fmaf_rn(
                SHARED_PARTIAL_SIG(shmem_FP32,
                    word_mappings::unpack<bits>(my_prefix_map, my_degree - suffix_len - 1)),
                running_prod,
                temp_sum);
        }

        // Add final terms (full prefix and full suffix)
        running_prod *= -SHARED_PATH_INCREMENTS(shmem_FP32,
            word_mappings::unpack<bits>(my_letters_from_back, my_degree - 1)) *
            c_reciprocals_f32[my_degree];
        temp_sum += running_prod;
        my_sig_val = t == 0 ? 0.0f : (my_sig_val + temp_sum);

        // Store S_{0,t} in shared memory for steps 2, and 3 (and step 1 in the next time)
        __syncthreads();
        if (threadIdx.x < num_lower) {
            SHARED_PARTIAL_SIG(shmem_FP32, threadIdx.x) = my_sig_val;  // S_{0,t}
        }

        // Step 2: Compute common factors CF_t(*)
        float my_common_factor = my_grad_val;  // my_degree term of CF_t(*)

        // Common factors CF_t(*) for words at levels < TRUNC_LVL
        #pragma unroll 1
        for (int lvl = TRUNC_LVL - 1; lvl > 0; --lvl) {
            const bool contributes = (lvl < my_degree);
            unsigned pref_tid = 1024;
            if (contributes) {
                pref_tid = word_mappings::unpack<bits>(my_prefix_map, lvl - 1);
            }

            // Warp-level reduction
            warp.sync();
            auto group = cg::labeled_partition(warp, pref_tid);
            float contrib = (contributes) ?
                sig_suffix_vals[my_degree - lvl] * my_grad_val : 0.0f;
            float sum = cg::reduce(group, contrib, cg::plus<float>());

            if (pref_tid != 1024 && group.thread_rank() == 0) {
                atomicAdd(&SHARED_COMMON_TERMS(shmem_FP32, pref_tid * num_warps + wid), sum);
            }
        }
        __syncthreads();

        // Accumulate common factor from higher degree terms
        if (threadIdx.x < num_lower) {
            for (int i = 0; i < num_warps; ++i) {
                my_common_factor += SHARED_COMMON_TERMS(shmem_FP32, threadIdx.x * num_warps + i);
            }
        }

        // Step 3: Compute dS_{0,T}(w)/dΔX_t * CF_t(w)
        running_prod = 1.0f;

        #pragma unroll 1
        for (int pref_len = TRUNC_LVL - 1; pref_len >= 0; --pref_len) {
            const unsigned suf_len = (my_degree > pref_len) ?
                my_degree - pref_len : my_degree;

            // Update running_prod with prior increment
            if (suf_len >= 2 && pref_len < my_degree) {
                running_prod *= SHARED_PATH_INCREMENTS(shmem_FP32,
                    word_mappings::unpack<bits>(my_letters_from_back, suf_len - 2)) *
                    c_reciprocals_f32[suf_len - 1];
            }

            // Process only if this is the rightmost occurrence
            const bool is_rightmost = (pref_len < my_degree) &&
                (((rightmost_mask >> (suf_len - 1)) & 1ULL) != 0);

            // Get current suffix letter
            unsigned suffix_letter = (is_rightmost) ?
                word_mappings::unpack<bits>(my_letters_from_back, suf_len - 1) : 1024;

            // Initialize gradient value
            float current_grad_val = 0.0f;

            if (is_rightmost) {
                float base_suffix = running_prod * c_reciprocals_f32[suf_len];

                // First term contribution
                current_grad_val = (suf_len != my_degree) ?
                    base_suffix * SHARED_PARTIAL_SIG(shmem_FP32,
                        word_mappings::unpack<bits>(my_prefix_map, pref_len - 1)) :
                    base_suffix;

                // Process remaining terms
                int count = 1;
                #pragma unroll 1
                for (unsigned tmp_suf_len = suf_len + 1; tmp_suf_len <= my_degree; ++tmp_suf_len) {
                    uint8_t letter = word_mappings::unpack<bits>(
                        my_letters_from_back, tmp_suf_len - 1);
                    base_suffix *= SHARED_PATH_INCREMENTS(shmem_FP32, letter) *
                                  c_reciprocals_f32[tmp_suf_len];
                    count += (letter == suffix_letter);

                    if (tmp_suf_len != my_degree) {
                        current_grad_val += count * base_suffix *
                            SHARED_PARTIAL_SIG(shmem_FP32,
                                word_mappings::unpack<bits>(my_prefix_map, my_degree - tmp_suf_len - 1));
                    } else {
                        current_grad_val += count * base_suffix;
                    }
                }
            }

            current_grad_val *= my_common_factor;

            // Use labeled reduce for accumulation (scoped to limit registers)
            {
                auto group = cg::labeled_partition(warp, suffix_letter);
                float v_sum = cg::reduce(group, current_grad_val, cg::plus<float>());
                if (suffix_letter < 1024 && group.thread_rank() == 0) {
                    atomicAdd(&SHARED_LETTER_GRADS(shmem_FP32, suffix_letter * num_warps + wid), v_sum);
                }
            }
        }
        __syncthreads();

        // Step 4: Reduce gradient contributions and atomic add to global memory
        if (threadIdx.x < d) {
            float letter_grad = 0.0f;
            for (int w = 0; w < num_warps; ++w) {
                letter_grad += SHARED_LETTER_GRADS(shmem_FP32, threadIdx.x * num_warps + w);
            }
            if (letter_grad != 0.0f) {
                atomicAdd(&increment_grads[INC_BATCH_OFFSET + t * d + threadIdx.x], letter_grad);
            }
        }

        // Step 5: Update S_{t,T}(X, suffix) for next time step (in-place)
        #pragma unroll 1
        for (int sig_degree = my_degree - 1; sig_degree > 0; --sig_degree) {
            running_prod = 1.0f;
            temp_sum = 0.0f;

            #pragma unroll 1
            for (int pref_len = 1; pref_len <= sig_degree; ++pref_len) {
                running_prod *= SHARED_PATH_INCREMENTS(shmem_FP32,
                    word_mappings::unpack<bits>(my_letters_from_back, sig_degree - pref_len)) *
                    c_reciprocals_f32[pref_len];
                temp_sum += running_prod * sig_suffix_vals[sig_degree - pref_len];
            }
            sig_suffix_vals[sig_degree] += temp_sum;
        }
    }  // End time loop
}  // function signatureBackwardPass (float)


/// Increment gradient to path gradient computation
template<typename Scalar>
__global__ void incrementGradToPathGrad(
    const Scalar* __restrict__ increment_gradients,
    Scalar* __restrict__ path_gradients,
    int batch_size, int num_time_steps, int path_dim)
{
    int num_increments = num_time_steps - 1;
    int total_output_elements = batch_size * num_time_steps * path_dim;
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_index >= total_output_elements) return;

    int time_dim_stride = num_time_steps * path_dim;
    int batch_index = global_index / time_dim_stride;
    int time_dim_remainder = global_index % time_dim_stride;
    int time_index = time_dim_remainder / path_dim;
    int dim_index = time_dim_remainder % path_dim;

    int increment_batch_stride = num_increments * path_dim;
    int increment_base_index = batch_index * increment_batch_stride + dim_index;

    if (time_index == 0) {
        path_gradients[global_index] = -increment_gradients[increment_base_index];
    } else if (time_index == num_time_steps - 1) {
        path_gradients[global_index] =
            increment_gradients[increment_base_index + (num_increments - 1) * path_dim];
    } else {
        int previous_increment_index = increment_base_index + (time_index - 1) * path_dim;
        int current_increment_index = increment_base_index + time_index * path_dim;
        path_gradients[global_index] =
            increment_gradients[previous_increment_index] -
            increment_gradients[current_increment_index];
    }
} // function incrementGradToPathGrad


/// Cleanup macros
#undef SIG_BATCH_OFFSET
#undef INC_BATCH_OFFSET
#undef SHARED_PATH_INCREMENTS
#undef SHARED_PARTIAL_SIG
#undef SHARED_COMMON_TERMS
#undef SHARED_LETTER_GRADS


/// Explicit template instantiations
// float, Bits=10, TRUNC_LVL 2..6
template __global__ void signatureBackwardPass<10, 2>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<10, 3>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<10, 4>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<10, 5>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<10, 6>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);

// float, Bits=8, TRUNC_LVL 7..8
template __global__ void signatureBackwardPass<8, 7>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<8, 8>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);

// float, Bits=5, TRUNC_LVL 9..12
template __global__ void signatureBackwardPass<5,  9>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<5, 10>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<5, 11>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<5, 12>(const float*, const float*, const float*, float*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);

// double, Bits=10, TRUNC_LVL 2..6
template __global__ void signatureBackwardPass<10, 2>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<10, 3>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<10, 4>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<10, 5>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<10, 6>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);

// double, Bits=8, TRUNC_LVL 7..8
template __global__ void signatureBackwardPass<8, 7>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<8, 8>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);

// double, Bits=5, TRUNC_LVL 9..12
template __global__ void signatureBackwardPass<5,  9>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<5, 10>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<5, 11>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);
template __global__ void signatureBackwardPass<5, 12>(const double*, const double*, const double*, double*, const uint64_t*, const uint64_t*, int, int, int, uint64_t, const unsigned*, const unsigned*, const uint64_t*, int);

// incrementGradToPathGrad instantiations
template __global__ void incrementGradToPathGrad<float>(
    const float*, float*, int, int, int);
template __global__ void incrementGradToPathGrad<double>(
    const double*, double*, int, int, int);

}  // namespace sig_backprop
} // namespace pathsig
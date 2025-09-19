// compute_sig.cu - CUDA kernels for signature computation
#include "compute_sig.cuh"
#include "word_mappings.cuh"
#include "extended_precision.cuh"
#include <cuda_runtime.h>
#include <cstdint>


namespace pathsig {
namespace compute_sig {

/// Constant Memory allocation for Forward Pass
// Constant memory needed for signature computation
constexpr int MAX_TRUNC_LEVEL = 12;

template<int N>
__device__ __host__ constexpr double108_t make_dd_recip()
{
    // Check for exact representations
    if (N == 4) return {0.0, 0.25};
    if (N == 8) return {0.0, 0.125};

    // For non-exact, compute high and low parts
    double hi = 1.0 / static_cast<double>(N);
    // Error term: (1.0 - hi * N) / N
    double lo = (1.0 - hi * static_cast<double>(N)) / static_cast<double>(N);
    return {lo, hi};
}

// Extended-precision constant memory
__constant__ double108_t c_inv_ints[MAX_TRUNC_LEVEL + 1] = {
 {0.0, 0.0},     // 0 (sentinel)
 {0.0, 1.0},     // 1/1
 {0.0, 0.5},     // 1/2
    make_dd_recip<3>(),   // 1/3
    make_dd_recip<4>(),   // 1/4
    make_dd_recip<5>(),   // 1/5
    make_dd_recip<6>(),   // 1/6
    make_dd_recip<7>(),   // 1/7
    make_dd_recip<8>(),   // 1/8
    make_dd_recip<9>(),   // 1/9
    make_dd_recip<10>(),  // 1/10
    make_dd_recip<11>(),  // 1/11
    make_dd_recip<12>()   // 1/12
};

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


/// Device helper: Kahan summation for improved numerical stability
__forceinline__ __device__ void kahanAdd(double& sum, double to_add, double& comp)
{
    double y = __dsub_rn(to_add, comp);
    double temp = __dadd_rn(sum, y);
    comp = __dsub_rn(__dsub_rn(temp, sum), y);
    sum = temp;
}

/// Overloaded Kahan summation for float
__forceinline__ __device__ void kahanAdd(float& sum, float to_add, float& comp)
{
    float y = __fsub_rn(to_add, comp);
    float temp = __fadd_rn(sum, y);
    comp = __fsub_rn(__fsub_rn(temp, sum), y);
    sum = temp;
}

/// Kernel for computing signature with double precision (FP64)
template<int bits>
__global__ void computeSignature(
    const double* path_increments,
    double* signature,
    int d,
    int trunc_lvl,
    int fixed_len,
    int num_time_steps,
    uint64_t total_sig_size,
    const uint64_t* d_powers,
    const uint64_t* level_offsets,
    const unsigned* degrees,
    const unsigned* varying_words,
    const uint64_t* prefix_maps)
{
    unsigned tid = threadIdx.x;
    uint64_t fixed_word = blockIdx.x;

    // Shared memory allocation
    extern __shared__ double shared_mem[];
    double* shared_path_increment = shared_mem;
    double* shared_partial_sig = &shared_mem[d];

    // Load thread-specific data
    int my_degree = degrees[tid];
    int my_varying_degree = (my_degree > fixed_len) ? my_degree - fixed_len : 0;
    uint64_t my_varying_word = varying_words[tid];
    uint64_t my_prefix_map = prefix_maps[tid];

    // Construct full word index
    uint64_t my_word = fixed_word * d_powers[my_varying_degree] + my_varying_word;

    // Remove letters from the back for threads with degree < fixed_len
    for (int len = fixed_len; len > my_degree; --len) {
        my_word /= d;
    }

    // Find index of the word in the signature array
    uint64_t my_sig_idx = level_offsets[my_degree] + my_word;

    // Compute letters in reverse order for efficient access
    uint64_t my_letters_from_back = word_mappings::packLettersFromBack<bits>(
        my_word, my_degree, d);

    // Batch element (signature) assigned to this block
    unsigned batch_idx = blockIdx.y;

    // Batch offset for path increments
    unsigned batch_path_offset = batch_idx * num_time_steps * d;

    // Load first path increments to shared memory
    __syncthreads();
    if (tid < d) {
        shared_path_increment[tid] = path_increments[batch_path_offset + tid];
    }
    __syncthreads();

    // Initialize signature value to 0 before first time step
    double my_sig_val = 0.0;
    double my_sig_val_comp = 0.0;  // Kahan summation compensator

    // Process each time step
    for (int t = 0; t < num_time_steps; ++t) {
        __syncthreads();

        // Share lower level signature terms
        if (my_degree < trunc_lvl) {
            shared_partial_sig[tid] = my_sig_val;
        }

        // Load current path increments
        if (tid < d) {
            shared_path_increment[tid] = path_increments[batch_path_offset + t * d + tid];
        }
        __syncthreads();

        // Apply Chen's relation to update signature
        double running_suffix = 1.0;
        double inner_sig_sum = 0.0;

        // Sum over all prefix-suffix decompositions (except full suffix, full prefix)
        for (int suffix_len = 1; suffix_len < my_degree; ++suffix_len) {
            int prefix_len = my_degree - suffix_len;

            // Update running suffix product
            running_suffix *= shared_path_increment[
                word_mappings::unpack<bits>(my_letters_from_back, suffix_len - 1)];
            running_suffix *= c_reciprocals[suffix_len];

            // Add contribution from this decomposition
            inner_sig_sum = __fma_rn(
                shared_partial_sig[
                    word_mappings::unpack<bits>(my_prefix_map, prefix_len - 1)],
                running_suffix,
                inner_sig_sum);
        }

        // Full suffix term
        running_suffix *= shared_path_increment[
            word_mappings::unpack<bits>(my_letters_from_back, my_degree - 1)] *
            c_reciprocals[my_degree];
        inner_sig_sum += running_suffix;

        // Update signature value with Kahan summation (full prefix term)
        kahanAdd(my_sig_val, inner_sig_sum, my_sig_val_comp);

    } // End time loop

    // Write result to global memory
    uint64_t my_sig_offset = total_sig_size * batch_idx;
    signature[my_sig_offset + my_sig_idx] = my_sig_val;
} // function computeSignature (double)


/// Kernel for computing signature with single precision (float)
template<int bits>
__global__ void computeSignature(
    const float* path_increments,
    float* signature,
    int d,
    int trunc_lvl,
    int fixed_len,
    int num_time_steps,
    uint64_t total_sig_size,
    const uint64_t* d_powers,
    const uint64_t* level_offsets,
    const unsigned* degrees,
    const unsigned* varying_words,
    const uint64_t* prefix_maps)
{
    unsigned tid = threadIdx.x;
    uint64_t fixed_word = blockIdx.x;

    // Shared memory allocation
    extern __shared__ float shmem_FP32[];
    float* shared_path_increment = shmem_FP32;
    float* shared_partial_sig = &shmem_FP32[d];

    // Load thread-specific data
    int my_degree = degrees[tid];
    int my_varying_degree = (my_degree > fixed_len) ? my_degree - fixed_len : 0;
    uint64_t my_varying_word = varying_words[tid];
    uint64_t my_prefix_map = prefix_maps[tid];

    // Construct full word index
    uint64_t my_word = fixed_word * d_powers[my_varying_degree] + my_varying_word;

    // Remove letters from the back for threads with degree < fixed_len
    for (int len = fixed_len; len > my_degree; --len) {
        my_word /= d;
    }

    // Find index of the word in the signature array
    uint64_t my_sig_idx = level_offsets[my_degree] + my_word;

    // Compute letters in reverse order for efficient access
    uint64_t my_letters_from_back = word_mappings::packLettersFromBack<bits>(
        my_word, my_degree, d);

    // Batch element (signature) assigned to this block
    unsigned batch_idx = blockIdx.y;

    // Batch offset for path increments
    unsigned batch_path_offset = batch_idx * num_time_steps * d;

    // Load first path increments to shared memory
    __syncthreads();
    if (tid < d) {
        shared_path_increment[tid] = path_increments[batch_path_offset + tid];
    }
    __syncthreads();

    // Initialize signature value to 0 before first time step
    float my_sig_val = 0.0f;
    float my_sig_val_comp = 0.0f;  // Kahan summation compensator

    // Process each time step
    for (int t = 0; t < num_time_steps; ++t) {
        __syncthreads();

        // Share lower level signature terms
        if (my_degree < trunc_lvl) {
            shared_partial_sig[tid] = my_sig_val;
        }

        // Load current path increments
        if (tid < d) {
            shared_path_increment[tid] = path_increments[batch_path_offset + t * d + tid];
        }
        __syncthreads();

        // Apply Chen's relation to update signature
        float running_suffix = 1.0f;
        float inner_sig_sum = 0.0f;

        // Sum over all prefix-suffix decompositions (except full suffix, full prefix)
        for (int suffix_len = 1; suffix_len < my_degree; ++suffix_len) {
            int prefix_len = my_degree - suffix_len;

            // Update running suffix product
            running_suffix *= shared_path_increment[
                word_mappings::unpack<bits>(my_letters_from_back, suffix_len - 1)];
            running_suffix *= c_reciprocals_f32[suffix_len];

            // Add contribution from this decomposition
            inner_sig_sum = __fmaf_rn(
                shared_partial_sig[
                    word_mappings::unpack<bits>(my_prefix_map, prefix_len - 1)],
                running_suffix,
                inner_sig_sum);
        }

        // Add full suffix term
        running_suffix *= shared_path_increment[
            word_mappings::unpack<bits>(my_letters_from_back, my_degree - 1)] *
            c_reciprocals_f32[my_degree];
        inner_sig_sum += running_suffix;

        // Update signature value with Kahan summation (full prefix term)
        kahanAdd(my_sig_val, inner_sig_sum, my_sig_val_comp);

    } // End time loop

    // Write result to global memory
    uint64_t my_sig_offset = total_sig_size * batch_idx;
    signature[my_sig_offset + my_sig_idx] = my_sig_val;
} // function computeSignature (float)


/// kernel for computing signature with extended precision (double-double)
template<int bits>
__global__ void computePreciseSig(
    const double* path_increments,
    double* signature,
    int d,
    int trunc_lvl,
    int fixed_len,
    int num_time_steps,
    uint64_t total_sig_size,
    const uint64_t* d_powers,
    const uint64_t* level_offsets,
    const unsigned* degrees,
    const unsigned* varying_words,
    const uint64_t* prefix_maps)
{
    unsigned tid = threadIdx.x;
    uint64_t my_fixed_word = blockIdx.x;

    // Shared memory allocation for double-double precision
    extern __shared__ double shared_mem[];
    double108_t* shared_path_increment = (double108_t*)shared_mem;
    double108_t* shared_partial_sig = (double108_t*)&shared_mem[2 * d];

    // Load thread-specific data
    int my_degree = degrees[tid];
    int my_varying_degree = (my_degree > fixed_len) ? my_degree - fixed_len : 0;
    uint64_t my_varying_word = varying_words[tid];
    uint64_t my_prefix_map = prefix_maps[tid];

    // Construct full word index
    uint64_t my_word = my_fixed_word * d_powers[my_varying_degree] + my_varying_word;

    // Remove letters from the back for threads with degree < fixed_len
    for (int my_len = fixed_len; my_len > my_degree; --my_len) {
        my_word /= d;
    }

    // Global signature index
    uint64_t my_sig_idx = level_offsets[my_degree] + my_word;

    // Pack letters from back for efficient access
    uint64_t my_letters_from_back = word_mappings::packLettersFromBack<bits>(
        my_word, my_degree, d);

    // Batch element (signature) assigned to this block
    unsigned batch_idx = blockIdx.y;

    // Batch offset for path increments
    unsigned batch_path_offset = batch_idx * num_time_steps * (d * 2);

    // Initialize signature value with extended precision
    double108_t my_sig_val = make_double2(0.0, 0.0);

    // Load initial increments with extended precision
    __syncthreads();
    if (tid < d) {
        double lo = path_increments[batch_path_offset + tid * 2];
        double hi = path_increments[batch_path_offset + tid * 2 + 1];
        shared_path_increment[tid] = make_double2(lo, hi);
    }
    __syncthreads();

    // Process each time step
    for (int t = 0; t < num_time_steps; ++t) {
        __syncthreads();

        // Share lower level signature terms
        if (my_degree < trunc_lvl) {
            shared_partial_sig[tid] = my_sig_val;
        }

        // Load current path increments (double-double format)
        if (tid < d) {
            double lo = path_increments[batch_path_offset + t * (d * 2) + tid * 2];
            double hi = path_increments[batch_path_offset + t * (d * 2) + tid * 2 + 1];
            shared_path_increment[tid] = make_double2(lo, hi);
        }
        __syncthreads();

        // Apply Chen's relation with extended precision
        double108_t my_suffix_prod = make_double2(0.0, 1.0);  // Identity element

        // Sum over all prefix-suffix decompositions (except full suffix, full prefix)
        for (int suffix_len = 1; suffix_len < my_degree; ++suffix_len) {
            // Get path increment for current suffix letter
            double108_t suffix_inc = shared_path_increment[
                word_mappings::unpack<bits>(my_letters_from_back, suffix_len - 1)];

            // Update suffix product with extended precision
            my_suffix_prod = extended_prec::mul_double108(my_suffix_prod, suffix_inc);
            my_suffix_prod = extended_prec::div_double108_by_int(
                my_suffix_prod, suffix_len, c_inv_ints);

            // Get prefix signature value
            int prefix_len = my_degree - suffix_len;
            double108_t my_prefix_sig_val = shared_partial_sig[
                word_mappings::unpack<bits>(my_prefix_map, prefix_len - 1)];

            // Add contribution with extended precision
            double108_t to_add = extended_prec::mul_double108(
                my_suffix_prod, my_prefix_sig_val);
            my_sig_val = extended_prec::add_double108_t(my_sig_val, to_add);
        }

        // Add full suffix term
        double108_t last_inc = shared_path_increment[
            word_mappings::unpack<bits>(my_letters_from_back, my_degree - 1)];
        my_suffix_prod = extended_prec::mul_double108(my_suffix_prod, last_inc);
        my_suffix_prod = extended_prec::div_double108_by_int(
            my_suffix_prod, my_degree, c_inv_ints);

        // Update signature value (full prefix term)
        my_sig_val = extended_prec::add_double108_t(my_sig_val, my_suffix_prod);
    } // End time loop

    // Write result to global memory (converted back to double from extended)
    uint64_t my_sig_offset = total_sig_size * batch_idx;
    signature[my_sig_offset + my_sig_idx] = my_sig_val.y + my_sig_val.x;
}  // function computePreciseSig


/// Explicit template instantiations
// Explicit template instantiations for standard signature computation with double
template __global__ void computeSignature<5>(
    const double*, double*, int, int, int, int, uint64_t,
    const uint64_t*, const uint64_t*, const unsigned*, const unsigned*, const uint64_t*);
template __global__ void computeSignature<8>(
    const double*, double*, int, int, int, int, uint64_t,
    const uint64_t*, const uint64_t*, const unsigned*, const unsigned*, const uint64_t*);
template __global__ void computeSignature<10>(
    const double*, double*, int, int, int, int, uint64_t,
    const uint64_t*, const uint64_t*, const unsigned*, const unsigned*, const uint64_t*);

// Explicit template instantiations for standard signature computation with float
template __global__ void computeSignature<5>(
    const float*, float*, int, int, int, int, uint64_t,
    const uint64_t*, const uint64_t*, const unsigned*, const unsigned*, const uint64_t*);
template __global__ void computeSignature<8>(
    const float*, float*, int, int, int, int, uint64_t,
    const uint64_t*, const uint64_t*, const unsigned*, const unsigned*, const uint64_t*);
template __global__ void computeSignature<10>(
    const float*, float*, int, int, int, int, uint64_t,
    const uint64_t*, const uint64_t*, const unsigned*, const unsigned*, const uint64_t*);

// Explicit template instantiations for signature computation with extended precision
template __global__ void computePreciseSig<5>(
    const double*, double*, int, int, int, int, uint64_t,
    const uint64_t*, const uint64_t*, const unsigned*, const unsigned*, const uint64_t*);
template __global__ void computePreciseSig<8>(
    const double*, double*, int, int, int, int, uint64_t,
    const uint64_t*, const uint64_t*, const unsigned*, const unsigned*, const uint64_t*);
template __global__ void computePreciseSig<10>(
    const double*, double*, int, int, int, int, uint64_t,
    const uint64_t*, const uint64_t*, const unsigned*, const unsigned*, const uint64_t*);

}  // namespace compute_sig
} // namespace pathsig
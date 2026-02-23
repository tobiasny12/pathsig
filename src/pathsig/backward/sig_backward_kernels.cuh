#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace pathsig::kernels::backward {

    template<typename scalar_t, int bits, int degree, bool reduce_by_pos>
    __global__ void __launch_bounds__(128, 4) sig_backprop_level(
        const scalar_t* __restrict__ paths,
        const scalar_t* __restrict__ signatures,
        const scalar_t* __restrict__ incoming_grads,
        scalar_t* __restrict__ increment_grads,
        const int d,
        const int path_len,
        const uint64_t sig_size,
        const uint64_t level_size,
        const uint64_t level_offset,
        const uint64_t* __restrict__ words_at_lvl,
        const int* __restrict__ windows,
        const bool recompute_sig
    );


    template<typename scalar_t>
    __global__ void increment_grad_to_path_grad(
        const scalar_t* __restrict__ inc_grad,
        scalar_t* __restrict__ path_grad,
        const int batch_size,
        const int num_time_steps,
        const int d
    );

    template<typename scalar_t, int bits, int degree>
    __global__ void logsig_to_sig_grads(
        const scalar_t* __restrict__ sig_arr,
        const scalar_t* __restrict__ P_arr,
        const scalar_t* __restrict__ grad_logsig_arr,
        scalar_t* __restrict__ grad_sig_arr,
        scalar_t* __restrict__ gradP_arr,
        const int trunc_lvl,
        const int d,
        const uint64_t total_sig_size,
        const uint64_t level_size,
        const uint64_t level_offset,
        const uint64_t* __restrict__ words_at_lvl
    );

} // namespace pathsig::kernels::backward
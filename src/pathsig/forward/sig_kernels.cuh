#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace pathsig::kernels::forward {

    template <typename scalar_t, int bits, int degree>
    __global__ void compute_signature_level(
        const scalar_t* __restrict__ paths,
        scalar_t* __restrict__ signatures,
        const int d,
        const int path_len,
        const uint64_t sig_size,
        const uint64_t level_size,
        const uint64_t level_offset,
        const uint64_t* __restrict__ words_at_lvl,
        const int* __restrict__ windows
    );

    template <typename scalar_t, int bits, int degree>
    __global__ void sig_to_logsig(
        const scalar_t* __restrict__ signatures,
        scalar_t* __restrict__ P_arr,
        scalar_t* __restrict__ logsig_arr,
        const int trunc_lvl,
        const int d,
        const uint64_t sig_size,
        const uint64_t level_size,
        const uint64_t level_offset,
        const uint64_t* __restrict__ words_at_lvl
    );

} // namespace pathsig::kernels::forward

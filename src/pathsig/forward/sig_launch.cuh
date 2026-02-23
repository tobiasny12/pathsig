#pragma once

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>
#include <tuple>

namespace pathsig::launch::forward {

    at::Tensor computeSignature(
        const at::Tensor& path_in,
        int64_t depth,
        bool alternative_projection,
        const at::Tensor& encoded_words,
        c10::ArrayRef<int64_t> level_sizes,
        bool use_windows,
        const at::Tensor& windows
    );

    std::tuple<at::Tensor, at::Tensor> sig_to_logsig(
        const at::Tensor& signature_in,
        int64_t depth,
        int64_t d,
        bool alternative_projection,
        const at::Tensor& encoded_words,
        c10::ArrayRef<int64_t> level_sizes
    );

} // namespace pathsig::launch::forward
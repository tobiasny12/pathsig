#pragma once

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>

namespace pathsig::launch::backward {

    at::Tensor signature_backward(
        const at::Tensor& path_in,
        const at::Tensor& signature_in,
        const at::Tensor& grad_signature_in,
        int64_t depth,
        bool alternative_projection,
        const at::Tensor& encoded_words,
        const c10::ArrayRef<int64_t> level_sizes,
        bool use_windows,
        const at::Tensor& windows
    );

    at::Tensor logsig_backward(
        const at::Tensor& sig_in,
        const at::Tensor& P_in,
        const at::Tensor& grad_logsig_in,
        int64_t depth,
        bool alternative_projection,
        const at::Tensor& encoded_words,
        const c10::ArrayRef<int64_t> level_sizes
    );

} // namespace pathsig::launch::backward
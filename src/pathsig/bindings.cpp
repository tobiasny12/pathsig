// bindings.cpp
#include <torch/library.h>

#include "forward/sig_launch.cuh"
#include "backward/sig_backward_launch.cuh"

TORCH_LIBRARY(pathsig, m) {
    // Forward: signature
    m.def(
        "compute_signature("
        "Tensor path_in, "
        "int depth, "
        "bool alternative_projection, "
        "Tensor encoded_words, "
        "int[] level_sizes, "
        "bool use_windows, "
        "Tensor windows"
        ") -> Tensor",
        {at::Tag::pt2_compliant_tag}
    );

    // Backward: grad(signature) -> grad(path)
    m.def(
        "signature_backward("
        "Tensor path_in, "
        "Tensor signature_in, "
        "Tensor grad_signature_in, "
        "int depth, "
        "bool alternative_projection, "
        "Tensor encoded_words, "
        "int[] level_sizes, "
        "bool use_windows, "
        "Tensor windows"
        ") -> Tensor",
        {at::Tag::pt2_compliant_tag}
    );

    // Forward: signature -> logsig
    m.def(
        "sig_to_logsig("
        "Tensor signature_in, "
        "int depth, "
        "int d, "
        "bool alternative_projection, "
        "Tensor encoded_words, "
        "int[] level_sizes"
        ") -> (Tensor, Tensor)",
        {at::Tag::pt2_compliant_tag}
    );

    // Backward: grad(logsig) -> grad(signature)
    m.def(
        "logsig_backward("
        "Tensor signature_in, "
        "Tensor P_in, "
        "Tensor grad_logsig_in, "
        "int depth, "
        "bool alternative_projection, "
        "Tensor encoded_words, "
        "int[] level_sizes"
        ") -> Tensor",
        {at::Tag::pt2_compliant_tag}
    );
}

// SigDecomposition.cpp
#include "SigDecomposition.h"
#include <torch/torch.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <string>
#include <cstdint>


namespace pathsig {
uint64_t computeSigSize(int path_dim, int truncation_level)
{
    uint64_t total_terms = 0;
    uint64_t power = 1;
    uint64_t max_uint64 = std::numeric_limits<uint64_t>::max();

    for (int k = 1; k <= truncation_level; ++k) {
        // Check if multiplication would overflow
        if (power > max_uint64 / path_dim) {
            throw std::overflow_error("Overflow in computing signature size: Signature size is too large.");
        }
        power *= path_dim;

        // Check if addition would overflow
        if (total_terms > max_uint64 - power) {
            throw std::overflow_error("Overflow in computing signature size: Signature size is too large.");
        }
        total_terms += power;
    }

    return total_terms;
}
/// Packs maps to threads of the prefix words of a word
template<unsigned bits>
uint64_t packPrefixMappings(
    uint64_t varying_word, int varying_degree, int fixed_len, int d,
    const int64_t level_offsets[]) {
    constexpr uint64_t mask = (1ULL << bits) - 1;
    uint64_t packed = 0;
    // Fixed prefixes
    for (int j = 0; j < fixed_len; ++j) {
        packed |= uint64_t(j & mask) << (bits * j);
    }
    // Varying prefixes
    uint64_t current_word = varying_word / d;
    int current_degree = varying_degree - 1;
    while (current_degree > 0) {
        uint64_t prefix_tid = level_offsets[current_degree] + current_word + fixed_len;
        packed |= (prefix_tid & mask) << (bits * (fixed_len + current_degree - 1));
        current_word /= d;
        --current_degree;
    }
    return packed;
}

/// Constructor for SigDecomposition class
SigDecomposition::SigDecomposition(int path_dim, int trunc_level) {
    // Get maximum allowed lower degree terms (degree < trunc_level) in decomposition
    int max_num_lower = getMaxLower(trunc_level);

    // Calculate optimal number of free (varying) letters (N - m)
    num_free_letters_ = calcNumFreeLetters(path_dim, trunc_level, max_num_lower);

    // Compute the partial signature size (threads per block)
    partial_sig_sz_ = computePartialSignatureSize(path_dim, trunc_level, num_free_letters_);

    // Compute number of partial signatures (blocks needed)
    num_partial_sigs_ = calcNumPartialSigs(path_dim, trunc_level, num_free_letters_);

    // Compute number of lower degree terms
    num_lower_ = calcNumLower(path_dim, partial_sig_sz_, num_free_letters_);

    // Initialize setup data
    int bits = getNumBits(trunc_level);
    int fixed_len = trunc_level - num_free_letters_;
    initializeSetupData(path_dim, trunc_level, fixed_len, partial_sig_sz_, bits);
}


int SigDecomposition::getNumBits(int trunc_level) {
    if (trunc_level <= 6) {
        return 10;  // Can pack 6 values in 60 bits
    } else if (trunc_level <= 8) {
        return 8;   // Can pack 8 values in 64 bits
    } else {
        return 5;   // Can pack 12 values in 60 bits
    }
}


int SigDecomposition::getMaxLower(int trunc_level) {
    if (trunc_level <= 6) {
        return 1023;  // 2^10 - 1
    } else if (trunc_level <= 8) {
        return 255;   // 2^8 - 1
    } else {
        return 31;    // 2^5 - 1
    }
}


int SigDecomposition::computePartialSignatureSize(int dim, int trunc_level, int num_free) {
    // Number of signature terms with degree <= fixed_len in decomposition
    int psig_size = trunc_level - num_free;

    // Number of signature terms with degree > fixed_len in decomposition
    int dim_pow = dim;
    for (int k = 1; k <= num_free; ++k) {
        psig_size += dim_pow;
        if (psig_size > MAX_PARTIAL_SIG_SIZE) {
            throw std::overflow_error("Partial signature size exceeds maximum thread block size");
        }
        dim_pow *= dim;
    }
    return psig_size;
}

/// Calculates the optimal number of free letters
int SigDecomposition::calcNumFreeLetters(int dim, int trunc_level, int max_num_lower) {
    if (dim > 31) {
        return 1;
    }
    int free_letters = 0;
    int fl_cap = trunc_level;  // Maximum possible free letters
    int sum = trunc_level - 1;  // Initial partial sig size - 1
    int d_pow = dim;  // dim^(free_letters + 1)

    while (free_letters < fl_cap) {
        if (d_pow > (max_num_lower - sum)) {
            break;  // Exceeds max_num_lower
        }
        sum += (d_pow - 1);
        ++free_letters;
        d_pow *= dim;
        if (sum + 1 > MAX_PARTIAL_SIG_SIZE) {
            --free_letters;  // Rollback
            break;
        }
    }

    // Check for one more level if space allows
    if (free_letters < fl_cap && d_pow <= (MAX_PARTIAL_SIG_SIZE - sum - 1)) {
        ++free_letters;
    }
    return free_letters;
}


/// Calculates the number of lower degree terms.
int SigDecomposition::calcNumLower(int path_dim, int partial_sig_sz, int free_letters) {
    // Lower terms = all terms except those at truncation level
    int trunc_level_terms = 1;
    for (int i = 0; i < free_letters; ++i) {
        trunc_level_terms *= path_dim;
    }
    return partial_sig_sz - trunc_level_terms;
}


/// Calculates the number of partial signatures needed to cover signature
int SigDecomposition::calcNumPartialSigs(int dim, int trunc_level, int num_free_letters) {
    int num_fixed_letters = trunc_level - num_free_letters;
    int result = 1;
    for (int i = 0; i < num_fixed_letters; ++i) {
        result *= dim;
    }
    return result;
}


/// Initializes setup data tensors for decomposition
void SigDecomposition::initializeSetupData(
    int path_dim,
    int trunc_level,
    int fixed_len,
    int partial_sig_sz,
    int bits) {
    auto pinned_options = torch::TensorOptions()
        .device(torch::kCPU)
        .pinned_memory(true)
        .dtype(torch::kInt64);

    d_powers_ = torch::ones(trunc_level + 1, pinned_options);
    level_offsets_ = torch::zeros(trunc_level + 1, pinned_options);

    auto degrees_opts = pinned_options.dtype(torch::kInt32);
    degrees_ = torch::empty(partial_sig_sz, degrees_opts);
    varying_words_ = torch::empty(partial_sig_sz, degrees_opts);
    prefix_maps_ = torch::empty(partial_sig_sz, pinned_options);

    auto* d_powers_ptr = d_powers_.data_ptr<int64_t>();
    auto* level_offsets_ptr = level_offsets_.data_ptr<int64_t>();
    auto* degrees_ptr = degrees_.data_ptr<int32_t>();
    auto* varying_words_ptr = varying_words_.data_ptr<int32_t>();
    auto* prefix_maps_ptr = prefix_maps_.data_ptr<int64_t>();

    // Compute powers of path dim
    d_powers_ptr[0] = 1;
    for (int j = 1; j <= trunc_level; ++j) {
        d_powers_ptr[j] = d_powers_ptr[j - 1] * static_cast<int64_t>(path_dim);
    }

    // Compute level offsets
    level_offsets_ptr[0] = 0;
    level_offsets_ptr[1] = 0;
    for (int j = 2; j <= trunc_level; ++j) {
        level_offsets_ptr[j] = level_offsets_ptr[j - 1] + d_powers_ptr[j - 1];
    }

    total_sig_size_ = static_cast<uint64_t>(level_offsets_ptr[trunc_level] + d_powers_ptr[trunc_level]);

    // Compute per-thread parameters
    for (int tid = 0; tid < partial_sig_sz; ++tid) {
        int varying_degree = 0;
        int full_degree = 0;
        uint64_t varying_word = 0;

        if (tid < fixed_len) {
            full_degree = tid + 1;
        } else {
            int varying_idx = tid - fixed_len;
            varying_degree = 1;
            int varying_offset = 0;
            int next_offset = path_dim;
            while (varying_idx >= next_offset) {
                ++varying_degree;
                varying_offset = next_offset;
                next_offset += static_cast<int>(d_powers_ptr[varying_degree]);
            }
            varying_word = static_cast<uint64_t>(varying_idx - varying_offset);
            full_degree = fixed_len + varying_degree;
        }

        uint64_t prefix_mappings;
        switch (bits) {
            case 5:
                prefix_mappings = packPrefixMappings<5>(
                    varying_word, varying_degree, fixed_len, path_dim, level_offsets_ptr);
                break;
            case 8:
                prefix_mappings = packPrefixMappings<8>(
                    varying_word, varying_degree, fixed_len, path_dim, level_offsets_ptr);
                break;
            case 10:
                prefix_mappings = packPrefixMappings<10>(
                    varying_word, varying_degree, fixed_len, path_dim, level_offsets_ptr);
                break;
            default:
                throw std::runtime_error("Unsupported bits value encoutered in SigDecomposition: " + std::to_string(bits));
        }

        degrees_ptr[tid] = full_degree;
        varying_words_ptr[tid] = static_cast<int32_t>(varying_word);
        prefix_maps_ptr[tid] = static_cast<int64_t>(prefix_mappings);
    }
}

}  // namespace pathsig
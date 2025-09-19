// SigDecomposition.h
#pragma once
#include <cstdint>
#include <torch/types.h>

namespace pathsig {
uint64_t computeSigSize(int path_dim, int truncation_level);
/**
 * @brief Class for signature decomposition.
 *
 * This class determines and stores the optimal decomposition of the full signature
 * into partial signatures that can be computed efficiently in CUDA blocks.
 */
class SigDecomposition {
public:
    /**
     * @brief Constructs the SigDecomposition object and calculates optimal parameters.
     *
     * @param path_dim The feature dimension of the path (d).
     * @param trunc_level The truncation level (N).
     */
    SigDecomposition(int path_dim, int trunc_level);

    /**
     * @brief Default destructor
     */
    ~SigDecomposition() = default;

    // ============================================================================
    // Getters for computed parameters
    // ============================================================================

    /**
     * @brief Gets the size of each partial signature (number of threads per block).
     *
     * @return Number of signature terms computed by each thread block.
     */
    int getPartialSigSize() const { return partial_sig_sz_; }

    /**
     * @brief Gets the number of partial signatures (number of blocks per signature/batch element).
     *
     * @return Number of CUDA blocks needed to compute the full signature.
     */
    int getNumPartialSigs() const { return num_partial_sigs_; }

    /**
     * @brief Gets the number of free letters in the varying part.
     *
     * @return Number of letters that vary within a partial signature.
     */
    int getNumFreeLetters() const { return num_free_letters_; }

    /**
     * @brief Gets the number of lower degree terms in the partial signature.
     *
     * @return Number of terms with degree < truncation level.
     */
    int getNumLower() const { return num_lower_; }

    // ============================================================================
    // Getters for setup data
    // ============================================================================

    const torch::Tensor& getDPowers() const { return d_powers_; }

    const torch::Tensor& getLevelOffsets() const { return level_offsets_; }

    const torch::Tensor& getDegrees() const { return degrees_; }

    const torch::Tensor& getVaryingWords() const { return varying_words_; }

    const torch::Tensor& getPrefixMaps() const { return prefix_maps_; }

    uint64_t getTotalSigSize() const { return total_sig_size_; }

    // ============================================================================
    // Static utility functions
    // ============================================================================

    /**
     * @brief Gets the number of bits needed for packing word indices.
     *
     * @param trunc_lvl The truncation level.
     * @return Number of bits (5, 8, or 10) based on truncation level.
     */
    static int getNumBits(int trunc_lvl);

    /**
     * @brief Gets the maximum number of lower degree terms allowed.
     *
     * @param trunc_lvl The truncation level.
     * @return Maximum number of terms with degree < truncation level.
     */
    static int getMaxLower(int trunc_lvl);

    /**
     * @brief Computes the size of a partial signature.
     *
     * @param dim Path dimension (d).
     * @param trunc_level Truncation level (N).
     * @param num_free Number of free letters.
     * @return Total number of terms in the partial signature.
     */
    static int computePartialSignatureSize(int dim, int trunc_level, int num_free);

    /**
     * @brief Calculates the number of partial signatures needed.
     *
     * @param dim Path dimension (d).
     * @param trunc_level Truncation level (N).
     * @param num_free_letters Number of free letters.
     * @return Number of partial signatures (d^(N - num_free_letters)).
     */
    static int calcNumPartialSigs(int dim, int trunc_level, int num_free_letters);

    /**
     * @brief Calculates the number of lower degree terms.
     *
     * @param path_dim Path dimension (d).
     * @param partial_sig_sz Total partial signature size.
     * @param free_letters Number of free letters.
     * @return Number of terms with degree < truncation level.
     */
    static int calcNumLower(int path_dim, int partial_sig_sz, int free_letters);

    /**
     * @brief Calculates the optimal number of free letters.
     *
     * @param dim Path dimension (d).
     * @param trunc_level Truncation level (N).
     * @param max_num_lower Maximum allowed lower degree terms.
     * @return Optimal number of free letters for efficiency.
     */
    static int calcNumFreeLetters(int dim, int trunc_level, int max_num_lower);

private:
    // Maximum number of threads per block (CUDA constraint)
    static constexpr int MAX_PARTIAL_SIG_SIZE = 1024;

    // Computed parameters for the decomposition
    int partial_sig_sz_;     // Size of each partial signature
    int num_partial_sigs_;   // Number of partial signatures
    int num_free_letters_;   // Number of varying letters
    int num_lower_;          // Number of signature terms of degree lower than

    // Setup data members
    torch::Tensor d_powers_;
    torch::Tensor level_offsets_;
    torch::Tensor degrees_;
    torch::Tensor varying_words_;
    torch::Tensor prefix_maps_;
    uint64_t total_sig_size_;

    // Private initialization method for setup data
    void initializeSetupData(int path_dim, int trunc_level, int fixed_len, int partial_sig_sz, int bits);
};

} // namespace pathsig
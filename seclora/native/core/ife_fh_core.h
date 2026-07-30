#ifndef IFE_FH_CORE_H
#define IFE_FH_CORE_H

#include <array>
#include <vector>
#include <utility>
#include <mcl/bn.hpp>

// Function-hiding inner-product FE core (Lin construction, dual-mode KeyGen).
// Group convention: A-side ciphertext in G1, B-side key in G2, row hash t_lb in G2.
// Dimension N = 2*rank + 3:
//   data R | shadow R | two-dimensional mask | auxiliary 1.
//
// Offline/online split: Enc and KeyGen are factored into a data-independent
// precompute (per-round msk + ciphertext randomness r1/r2 + the reserved x slots,
// all chosen before the LoRA values arrive) and a cheap online step that only
// folds in the R real factor entries. Slots 0..R-1 are the online data slots;
// everything else (structural zeros and the two mask slots) is precomputed.
class iFE_FH_Core {
public:
    // Data-independent part of one A-side ciphertext (r1 and x baked in).
    struct EncPrecomp {
        std::vector<mcl::bn::G1> c2_base;   // length N+1; online adds g1^{a_k} to slots 1..R
        mcl::bn::G1 c1_base;                // online adds g1^{sum s2[k+1] a_k}
    };
    // Data-independent part of one B-side key (r2 and t_lb baked in).
    struct KeyPrecomp {
        mcl::bn::G2 k1;                     // g2^{-r2}
        mcl::bn::G2 k2_0_base;              // online adds g2^{sum s1[k] b_k}
        std::vector<mcl::bn::G2> k2_base;   // length N; online adds g2^{b_k} to slots 0..R-1
    };

private:
    int m_dim;                    // LoRA rank R
    int N;                        // 2R+3
    mcl::bn::G1 g1_base;
    mcl::bn::G2 g2_base;
    std::vector<mcl::bn::Fr> s1;  // msk, length N
    std::vector<mcl::bn::Fr> s2;  // msk, length N+1

public:
    iFE_FH_Core(int m_rank, const mcl::bn::G1& g1, const mcl::bn::G2& g2);
    void u_setup();   // refresh msk (per round)

    // KeyGen, split. a_offline / the reserved structure carries no real data;
    // b_real (length R) holds the R real B-row factors folded in online.
    KeyPrecomp keygen_precompute(
        const std::array<mcl::bn::G2, 2>& tag);
    std::pair<mcl::bn::G2, std::vector<mcl::bn::G2>>
    keygen_online(const KeyPrecomp& pre, const std::vector<mcl::bn::Fr>& b_real);

    // Enc, split. a_offline (length N) carries only data-independent entries
    // (the two-dimensional x mask); a_real (length R) holds the real A-column
    // factors.
    EncPrecomp encrypt_precompute(const std::vector<mcl::bn::Fr>& a_offline);
    std::pair<mcl::bn::G1, std::vector<mcl::bn::G1>>
    encrypt_online(const EncPrecomp& pre, const std::vector<mcl::bn::Fr>& a_real);

    // Overloads taking pre-sampled randomness (no setByCSPRNG inside), so the
    // group-op-heavy precompute can run in parallel after sampling serially.
    // These only read msk (s1/s2/bases), so concurrent calls are thread-safe.
    EncPrecomp encrypt_precompute(const std::vector<mcl::bn::Fr>& a_offline, const mcl::bn::Fr& r1);
    KeyPrecomp keygen_precompute(
        const std::array<mcl::bn::G2, 2>& tag,
        const mcl::bn::Fr& r2);
};

#endif  // IFE_FH_CORE_H

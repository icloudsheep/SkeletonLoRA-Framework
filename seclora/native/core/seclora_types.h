#ifndef SECLORA_TYPES_H
#define SECLORA_TYPES_H

#include <vector>
#include <string>
#include <mcl/bn.hpp>

// Public parameters for PC-MCFE secure aggregation (no ZK material).
//   g0       : G1 generator for the iFE ciphertext group / DMCFE mask base.
//   g1       : second base for the long-term key commitment C_{s,i}.
//   h_blinding: Pedersen blinding base for C_{s,i}.
//   g2_base  : G2 generator for the iFE key group.
//   cmt_s    : per-client long-term key commitments C_{s,i}.
struct SecLoRA_PP {
    mcl::bn::G1 g0;
    mcl::bn::G1 g1;
    mcl::bn::G1 h_blinding;
    mcl::bn::G2 g2_base;
    std::vector<mcl::bn::G1> cmt_s;
};

// A-side ciphertext slot (per column). Ciphertext lives in G1.
struct A_Ciphertext_Slot {
    mcl::bn::G1 ife_c1;
    std::vector<mcl::bn::G1> ife_c2;
    mcl::bn::G1 c_i;            // DMCFE mask term c_i = mu_a * s_i + g0 * x_v
    bool is_zero;
};

// B-side secret-key slot (per row). Key lives in G2.
struct B_SecretKey_Slot {
    mcl::bn::G2 ife_k1;        // standard k1, or dual-mode compact key for a zero row
    std::vector<mcl::bn::G2> ife_k2;
    bool is_zero;
};

// Deterministic labels shared by Enc / KeyGen / Decrypt (must match byte-for-byte).
inline std::string labelA(int layer, int pos, int v, int q, int bit) {
    return std::to_string(layer) + "_" + std::to_string(pos) + "_" +
           std::to_string(v) + "_" + std::to_string(q) + "_" + std::to_string(bit);
}
inline std::string labelB(int layer, int pos, int u, int q) {
    return std::to_string(layer) + "_" + std::to_string(pos) + "_" +
           std::to_string(u) + "_" + std::to_string(q);
}

#endif  // SECLORA_TYPES_H

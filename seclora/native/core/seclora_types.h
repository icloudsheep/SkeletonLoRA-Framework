#ifndef SECLORA_TYPES_H
#define SECLORA_TYPES_H

#include <string>
#include <vector>

#include <mcl/bn.hpp>

#include "dmcfe_mask.h"

// Public bases shared by FH-IPFE and the fixed-base DMCFE output.
struct SecLoRA_PP {
    mcl::bn::G1 g0;
    mcl::bn::G2 g2_base;
};

// A-side ciphertext for one complete column label.
struct A_Ciphertext_Slot {
    mcl::bn::G1 ife_c1;
    std::vector<mcl::bn::G1> ife_c2;
    DmcfeCiphertext2 dfe_ct;
};

// B-side FH-IPFE key for one complete row label.
struct B_SecretKey_Slot {
    mcl::bn::G2 ife_k1;
    std::vector<mcl::bn::G2> ife_k2;
};

inline std::string labelA(int layer, int pos, int v, int q) {
    return "SecLoRA|A|q=" + std::to_string(q) +
           "|layer=" + std::to_string(layer) +
           "|pos=" + std::to_string(pos) +
           "|column=" + std::to_string(v);
}

inline std::string labelB(int layer, int pos, int u, int q) {
    return "SecLoRA|B|q=" + std::to_string(q) +
           "|layer=" + std::to_string(layer) +
           "|pos=" + std::to_string(pos) +
           "|row=" + std::to_string(u);
}

inline std::string labelBTag(
    int layer, int pos, int u, int q, int coordinate) {
    return labelB(layer, pos, u, q) +
           "|mask-coordinate=" + std::to_string(coordinate);
}

#endif  // SECLORA_TYPES_H

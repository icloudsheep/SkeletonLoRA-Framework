#ifndef SECLORA_DMCFE_MASK_H
#define SECLORA_DMCFE_MASK_H

#include <array>
#include <string>
#include <vector>

#include <mcl/bn.hpp>

// Two-dimensional mask DMCFE used by PC-DMCFE. Each coordinate is an
// independent ABG19 decentralized MCFE instance built from the fully secure
// ALS16 DDH IPFE and pairwise PRF zero shares.
struct DmcfeScalarPublicParams {
    mcl::bn::G1 g;
    mcl::bn::G1 h;
    std::vector<mcl::bn::G1> h_coord;
};

struct DmcfeScalarClientSecret {
    std::vector<mcl::bn::Fr> s_share;
    std::vector<mcl::bn::Fr> t_share;
    std::vector<mcl::bn::Fr> pairwise_prf_keys;
};

struct DmcfeClientSecret2 {
    int client_id = -1;
    std::array<DmcfeScalarClientSecret, 2> channel;
};

struct DmcfeScalarCiphertext {
    mcl::bn::G1 c;
    mcl::bn::G1 d;
    std::vector<mcl::bn::G1> e;
};

struct DmcfeCiphertext2 {
    std::array<DmcfeScalarCiphertext, 2> channel;
};

struct DmcfeScalarKeyShare {
    mcl::bn::Fr s;
    mcl::bn::Fr t;
};

struct DmcfeKeyShare2 {
    std::array<DmcfeScalarKeyShare, 2> channel;
};

struct DmcfeFunctionalKey2 {
    std::array<DmcfeScalarKeyShare, 2> channel;
};

struct DmcfePublicParams2 {
    int clients = 0;
    std::array<DmcfeScalarPublicParams, 2> channel;
};

class ABG19DmcfeMask2 {
public:
    static void Setup(int clients, const mcl::bn::G1& fixed_base,
                      DmcfePublicParams2& public_params,
                      std::vector<DmcfeClientSecret2>& client_secrets);

    static DmcfeCiphertext2 Encrypt(
        const DmcfePublicParams2& public_params,
        const DmcfeClientSecret2& client_secret,
        const std::string& label,
        const std::array<mcl::bn::Fr, 2>& message,
        const std::array<mcl::bn::Fr, 2>& randomness);

    static DmcfeKeyShare2 DKeyShareGen(
        const DmcfeClientSecret2& client_secret,
        const std::vector<mcl::bn::Fr>& weights);

    static DmcfeFunctionalKey2 DKeyComb(
        const std::vector<DmcfeKeyShare2>& shares);

    static std::array<mcl::bn::G1, 2> Dec(
        const DmcfePublicParams2& public_params,
        const DmcfeFunctionalKey2& key,
        const std::vector<mcl::bn::Fr>& weights,
        const std::vector<DmcfeCiphertext2>& ciphertexts);

    static long long SerializedBytes(const DmcfeCiphertext2& ciphertext);
};

#endif  // SECLORA_DMCFE_MASK_H

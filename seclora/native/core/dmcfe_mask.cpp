#include "dmcfe_mask.h"

#include <cybozu/sha2.hpp>

#include <stdexcept>

using namespace mcl::bn;

namespace {

Fr prf_to_fr(const Fr& key, int channel, int coordinate,
             const std::string& label) {
    const std::string key_bytes = key.getStr(mcl::IoSerialize);
    const std::string input =
        "SecLoRA|ABG19|PRF|v1|channel=" + std::to_string(channel) +
        "|coordinate=" + std::to_string(coordinate) +
        "|label_bytes=" + std::to_string(label.size()) + "|" + label;
    unsigned char digest[32];
    cybozu::hmac256(
        digest, key_bytes.data(), key_bytes.size(), input.data(), input.size());
    Fr value;
    value.setBigEndianMod(digest, sizeof(digest));
    return value;
}

DmcfeScalarCiphertext encrypt_scalar(
    const DmcfeScalarPublicParams& pp,
    const DmcfeScalarClientSecret& sk,
    int client_id,
    int channel,
    const std::string& label,
    const Fr& message,
    const Fr& randomness) {
    const int clients = static_cast<int>(pp.h_coord.size());
    if (client_id < 0 || client_id >= clients ||
        static_cast<int>(sk.pairwise_prf_keys.size()) != clients) {
        throw std::invalid_argument("invalid DMCFE client secret");
    }

    std::vector<Fr> pad(static_cast<size_t>(clients), Fr(0));
    for (int peer = 0; peer < clients; ++peer) {
        if (peer == client_id) continue;
        for (int coordinate = 0; coordinate < clients; ++coordinate) {
            const Fr value = prf_to_fr(
                sk.pairwise_prf_keys[static_cast<size_t>(peer)],
                channel, coordinate, label);
            if (peer > client_id) pad[static_cast<size_t>(coordinate)] += value;
            else pad[static_cast<size_t>(coordinate)] -= value;
        }
    }
    pad[static_cast<size_t>(client_id)] += message;

    DmcfeScalarCiphertext ciphertext;
    G1::mul(ciphertext.c, pp.g, randomness);
    G1::mul(ciphertext.d, pp.h, randomness);
    ciphertext.e.resize(static_cast<size_t>(clients));
    for (int coordinate = 0; coordinate < clients; ++coordinate) {
        G1 public_term;
        G1 message_term;
        G1::mul(
            public_term, pp.h_coord[static_cast<size_t>(coordinate)],
            randomness);
        G1::mul(message_term, pp.g, pad[static_cast<size_t>(coordinate)]);
        G1::add(
            ciphertext.e[static_cast<size_t>(coordinate)],
            public_term, message_term);
    }
    return ciphertext;
}

}  // namespace

void ABG19DmcfeMask2::Setup(
    int clients, const G1& fixed_base, DmcfePublicParams2& public_params,
    std::vector<DmcfeClientSecret2>& client_secrets) {
    if (clients <= 0) throw std::invalid_argument("DMCFE needs clients > 0");

    public_params.clients = clients;
    client_secrets.assign(static_cast<size_t>(clients), DmcfeClientSecret2());
    for (int i = 0; i < clients; ++i) {
        client_secrets[static_cast<size_t>(i)].client_id = i;
    }

    for (int channel = 0; channel < 2; ++channel) {
        DmcfeScalarPublicParams& pp =
            public_params.channel[static_cast<size_t>(channel)];
        pp.g = fixed_base;
        const std::string h_label =
            "SecLoRA|ABG19|ALS16|h|channel=" + std::to_string(channel);
        hashAndMapToG1(pp.h, h_label.data(), h_label.size());
        pp.h_coord.assign(static_cast<size_t>(clients), G1());
        for (G1& point : pp.h_coord) point.clear();

        for (int i = 0; i < clients; ++i) {
            DmcfeScalarClientSecret& sk =
                client_secrets[static_cast<size_t>(i)]
                    .channel[static_cast<size_t>(channel)];
            sk.s_share.resize(static_cast<size_t>(clients));
            sk.t_share.resize(static_cast<size_t>(clients));
            sk.pairwise_prf_keys.assign(static_cast<size_t>(clients), Fr(0));
            for (int coordinate = 0; coordinate < clients; ++coordinate) {
                sk.s_share[static_cast<size_t>(coordinate)].setByCSPRNG();
                sk.t_share[static_cast<size_t>(coordinate)].setByCSPRNG();

                G1 gs;
                G1 ht;
                G1 contribution;
                G1::mul(
                    gs, pp.g, sk.s_share[static_cast<size_t>(coordinate)]);
                G1::mul(
                    ht, pp.h, sk.t_share[static_cast<size_t>(coordinate)]);
                G1::add(contribution, gs, ht);
                pp.h_coord[static_cast<size_t>(coordinate)] += contribution;
            }
        }

        for (int i = 0; i < clients; ++i) {
            for (int j = i + 1; j < clients; ++j) {
                Fr pairwise_key;
                pairwise_key.setByCSPRNG();
                client_secrets[static_cast<size_t>(i)]
                    .channel[static_cast<size_t>(channel)]
                    .pairwise_prf_keys[static_cast<size_t>(j)] = pairwise_key;
                client_secrets[static_cast<size_t>(j)]
                    .channel[static_cast<size_t>(channel)]
                    .pairwise_prf_keys[static_cast<size_t>(i)] = pairwise_key;
            }
        }
    }
}

DmcfeCiphertext2 ABG19DmcfeMask2::Encrypt(
    const DmcfePublicParams2& public_params,
    const DmcfeClientSecret2& client_secret,
    const std::string& label,
    const std::array<Fr, 2>& message,
    const std::array<Fr, 2>& randomness) {
    DmcfeCiphertext2 ciphertext;
    for (int channel = 0; channel < 2; ++channel) {
        ciphertext.channel[static_cast<size_t>(channel)] = encrypt_scalar(
            public_params.channel[static_cast<size_t>(channel)],
            client_secret.channel[static_cast<size_t>(channel)],
            client_secret.client_id, channel, label,
            message[static_cast<size_t>(channel)],
            randomness[static_cast<size_t>(channel)]);
    }
    return ciphertext;
}

DmcfeKeyShare2 ABG19DmcfeMask2::DKeyShareGen(
    const DmcfeClientSecret2& client_secret,
    const std::vector<Fr>& weights) {
    DmcfeKeyShare2 share;
    for (int channel = 0; channel < 2; ++channel) {
        const DmcfeScalarClientSecret& sk =
            client_secret.channel[static_cast<size_t>(channel)];
        if (sk.s_share.size() != weights.size() ||
            sk.t_share.size() != weights.size()) {
            throw std::invalid_argument("DMCFE weight dimension mismatch");
        }
        share.channel[static_cast<size_t>(channel)].s = Fr(0);
        share.channel[static_cast<size_t>(channel)].t = Fr(0);
        for (size_t coordinate = 0; coordinate < weights.size(); ++coordinate) {
            share.channel[static_cast<size_t>(channel)].s +=
                sk.s_share[coordinate] * weights[coordinate];
            share.channel[static_cast<size_t>(channel)].t +=
                sk.t_share[coordinate] * weights[coordinate];
        }
    }
    return share;
}

DmcfeFunctionalKey2 ABG19DmcfeMask2::DKeyComb(
    const std::vector<DmcfeKeyShare2>& shares) {
    DmcfeFunctionalKey2 key;
    for (int channel = 0; channel < 2; ++channel) {
        key.channel[static_cast<size_t>(channel)].s = Fr(0);
        key.channel[static_cast<size_t>(channel)].t = Fr(0);
        for (const DmcfeKeyShare2& share : shares) {
            key.channel[static_cast<size_t>(channel)].s +=
                share.channel[static_cast<size_t>(channel)].s;
            key.channel[static_cast<size_t>(channel)].t +=
                share.channel[static_cast<size_t>(channel)].t;
        }
    }
    return key;
}

std::array<G1, 2> ABG19DmcfeMask2::Dec(
    const DmcfePublicParams2& public_params,
    const DmcfeFunctionalKey2& key,
    const std::vector<Fr>& weights,
    const std::vector<DmcfeCiphertext2>& ciphertexts) {
    const int clients = public_params.clients;
    if (static_cast<int>(weights.size()) != clients ||
        static_cast<int>(ciphertexts.size()) != clients) {
        throw std::invalid_argument("incomplete DMCFE label family");
    }

    std::array<G1, 2> result;
    for (int channel = 0; channel < 2; ++channel) {
        result[static_cast<size_t>(channel)].clear();
        const DmcfeScalarKeyShare& functional_key =
            key.channel[static_cast<size_t>(channel)];
        for (int i = 0; i < clients; ++i) {
            const DmcfeScalarCiphertext& ciphertext =
                ciphertexts[static_cast<size_t>(i)]
                    .channel[static_cast<size_t>(channel)];
            if (static_cast<int>(ciphertext.e.size()) != clients) {
                throw std::invalid_argument("invalid DMCFE ciphertext dimension");
            }

            G1 local;
            local.clear();
            for (int coordinate = 0; coordinate < clients; ++coordinate) {
                G1 term;
                G1::mul(
                    term, ciphertext.e[static_cast<size_t>(coordinate)],
                    weights[static_cast<size_t>(coordinate)]);
                local += term;
            }
            G1 c_term;
            G1 d_term;
            G1::mul(c_term, ciphertext.c, functional_key.s);
            G1::mul(d_term, ciphertext.d, functional_key.t);
            local -= c_term;
            local -= d_term;
            result[static_cast<size_t>(channel)] += local;
        }
    }
    return result;
}

long long ABG19DmcfeMask2::SerializedBytes(
    const DmcfeCiphertext2& ciphertext) {
    long long bytes = 0;
    for (const DmcfeScalarCiphertext& channel : ciphertext.channel) {
        bytes += static_cast<long long>(
            channel.c.getStr(mcl::IoSerialize).size());
        bytes += static_cast<long long>(
            channel.d.getStr(mcl::IoSerialize).size());
        for (const G1& element : channel.e) {
            bytes += static_cast<long long>(
                element.getStr(mcl::IoSerialize).size());
        }
    }
    return bytes;
}

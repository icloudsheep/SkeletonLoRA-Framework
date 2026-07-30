#include "pc_mcfe_server.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

using namespace mcl::bn;

namespace {

double now_sec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

PC_MCFE_Server::PC_MCFE_Server(
    int K, const SecLoRA_PP& global_pp,
    const DmcfePublicParams2& global_dfe_pp)
    : K_clients(K), pp(global_pp), dfe_pp(global_dfe_pp) {}

DmcfeFunctionalKey2 PC_MCFE_Server::DKeyComb(
    const std::vector<DmcfeKeyShare2>& client_shares) const {
    if (static_cast<int>(client_shares.size()) != K_clients) {
        throw std::invalid_argument("incomplete DMCFE key-share family");
    }
    return ABG19DmcfeMask2::DKeyComb(client_shares);
}

double PC_MCFE_Server::PrepareDfeMaskCacheRefs(
    const std::vector<const std::vector<A_Ciphertext_Slot>*>& all_A_cts,
    const std::vector<Fr>& p_weights,
    const DmcfeFunctionalKey2& key,
    const std::vector<int>& columns,
    int threads,
    double& worker_thread_sum_sec) {
    if (static_cast<int>(all_A_cts.size()) != K_clients ||
        static_cast<int>(p_weights.size()) != K_clients) {
        throw std::invalid_argument("incomplete PC-DMCFE client family");
    }

    size_t matrix_cols = 0;
    for (const auto* client_cts : all_A_cts) {
        if (client_cts == nullptr) {
            throw std::invalid_argument("null A ciphertext family");
        }
        matrix_cols = std::max(matrix_cols, client_cts->size());
    }
    for (int column : columns) {
        if (column < 0 || static_cast<size_t>(column) >= matrix_cols) {
            throw std::invalid_argument("DMCFE A-label column is out of range");
        }
        for (const auto* client_cts : all_A_cts) {
            if (static_cast<size_t>(column) >= client_cts->size()) {
                throw std::invalid_argument(
                    "missing DMCFE A-label ciphertext");
            }
        }
    }

    dfe_mask_cache_.assign(matrix_cols, std::array<G1, 2>());
    dfe_mask_ready_.assign(matrix_cols, 0);
    const int worker_count = columns.empty()
        ? 0
        : std::min(
              std::max(1, threads), static_cast<int>(columns.size()));
    std::vector<double> worker_seconds(
        static_cast<size_t>(worker_count), 0.0);

    const double wall_start = now_sec();
    auto worker = [&](int thread_id) {
        for (size_t index = static_cast<size_t>(thread_id);
             index < columns.size();
             index += static_cast<size_t>(worker_count)) {
            const double started = now_sec();
            const int column = columns[index];
            std::vector<DmcfeCiphertext2> ciphertexts(
                static_cast<size_t>(K_clients));
            for (int client = 0; client < K_clients; ++client) {
                ciphertexts[static_cast<size_t>(client)] =
                    (*all_A_cts[static_cast<size_t>(client)])
                    [static_cast<size_t>(column)].dfe_ct;
            }
            dfe_mask_cache_[static_cast<size_t>(column)] =
                ABG19DmcfeMask2::Dec(
                    dfe_pp, key, p_weights, ciphertexts);
            dfe_mask_ready_[static_cast<size_t>(column)] = 1;
            worker_seconds[static_cast<size_t>(thread_id)] +=
                now_sec() - started;
        }
    };

    std::vector<std::thread> pool;
    for (int thread_id = 0; thread_id < worker_count; ++thread_id) {
        pool.emplace_back(worker, thread_id);
    }
    for (std::thread& thread : pool) thread.join();

    worker_thread_sum_sec = 0.0;
    for (double seconds : worker_seconds) {
        worker_thread_sum_sec += seconds;
    }
    return now_sec() - wall_start;
}

void PC_MCFE_Server::build_bsgs(
    const GT& base, long long bound,
    const std::function<void(long long, long long)>& progress) {
    if (bound <= 0) {
        throw std::invalid_argument("BSGS bound must be positive");
    }
    base_gt_ = base;
    bsgs_.bound = bound;
    bsgs_.m = static_cast<long long>(
        std::ceil(std::sqrt(static_cast<double>(bound))));
    bsgs_.baby.clear();
    bsgs_.baby.reserve(static_cast<size_t>(bsgs_.m) * 2);

    GT current;
    GT::pow(current, base, Fr(0));
    const long long progress_step = std::max(1LL, bsgs_.m / 20);
    for (long long j = 0; j < bsgs_.m; ++j) {
        bsgs_.baby.emplace(current.getStr(mcl::IoSerialize), j);
        current *= base;
        if (progress &&
            ((j + 1) % progress_step == 0 || j + 1 == bsgs_.m)) {
            progress(j + 1, bsgs_.m);
        }
    }
    GT::inv(bsgs_.giant_inv, current);
    bsgs_.built = true;
}

long long PC_MCFE_Server::bsgs_table_bytes_estimate() const {
    const long long key_bytes =
        static_cast<long long>(base_gt_.getStr(mcl::IoSerialize).size());
    return static_cast<long long>(bsgs_.baby.size()) *
           (key_bytes + static_cast<long long>(sizeof(long long)));
}

long long PC_MCFE_Server::bsgs_search(
    const GT& target, bool& found) const {
    if (!bsgs_.built) {
        throw std::logic_error("BSGS table has not been built");
    }
    found = false;
    GT positive = target;
    GT negative;
    GT::inv(negative, target);
    for (long long i = 0; i <= bsgs_.m; ++i) {
        const auto positive_it =
            bsgs_.baby.find(positive.getStr(mcl::IoSerialize));
        if (positive_it != bsgs_.baby.end()) {
            const long long exponent =
                i * bsgs_.m + positive_it->second;
            if (exponent <= bsgs_.bound) {
                found = true;
                return exponent;
            }
        }
        const auto negative_it =
            bsgs_.baby.find(negative.getStr(mcl::IoSerialize));
        if (negative_it != bsgs_.baby.end()) {
            const long long exponent =
                i * bsgs_.m + negative_it->second;
            if (exponent <= bsgs_.bound) {
                found = true;
                return -exponent;
            }
        }
        positive *= bsgs_.giant_inv;
        negative *= bsgs_.giant_inv;
    }
    return 0;
}

GT PC_MCFE_Server::eval_one_cell_group_refs(
    const std::vector<const std::vector<A_Ciphertext_Slot>*>& all_A_cts,
    const std::vector<const std::vector<B_SecretKey_Slot>*>& all_B_sks,
    const std::vector<Fr>& p_weights,
    int layer_id, int pos_y, int round_q, int u, int v) {
    if (static_cast<int>(all_A_cts.size()) != K_clients ||
        static_cast<int>(all_B_sks.size()) != K_clients ||
        static_cast<int>(p_weights.size()) != K_clients) {
        throw std::invalid_argument("incomplete PC-DMCFE client family");
    }
    if (v < 0 || static_cast<size_t>(v) >= dfe_mask_ready_.size() ||
        !dfe_mask_ready_[static_cast<size_t>(v)]) {
        throw std::logic_error(
            "DMCFE mask cache is not ready for A label");
    }

    std::array<G2, 2> tag;
    for (int channel = 0; channel < 2; ++channel) {
        const std::string label =
            labelBTag(layer_id, pos_y, u, round_q, channel);
        hashAndMapToG2(
            tag[static_cast<size_t>(channel)],
            label.data(), label.size());
    }

    GT result;
    GT::pow(result, base_gt_, Fr(0));
    for (int client = 0; client < K_clients; ++client) {
        const auto* a_slots = all_A_cts[static_cast<size_t>(client)];
        const auto* b_slots = all_B_sks[static_cast<size_t>(client)];
        if (a_slots == nullptr || b_slots == nullptr ||
            static_cast<size_t>(v) >= a_slots->size() ||
            u < 0 || static_cast<size_t>(u) >= b_slots->size()) {
            throw std::invalid_argument("missing FH-IPFE slot");
        }
        const auto& ciphertext = (*a_slots)[static_cast<size_t>(v)];
        const auto& key = (*b_slots)[static_cast<size_t>(u)];
        if (ciphertext.ife_c2.size() != key.ife_k2.size()) {
            throw std::invalid_argument(
                "FH-IPFE ciphertext/key dimension mismatch");
        }

        GT local;
        pairing(local, ciphertext.ife_c1, key.ife_k1);
        for (size_t element = 0;
             element < ciphertext.ife_c2.size(); ++element) {
            GT pairing_value;
            pairing(
                pairing_value, ciphertext.ife_c2[element],
                key.ife_k2[element]);
            local *= pairing_value;
        }
        GT weighted;
        GT::pow(
            weighted, local, p_weights[static_cast<size_t>(client)]);
        result *= weighted;
    }

    GT mask_0;
    GT mask_1;
    pairing(
        mask_0, dfe_mask_cache_[static_cast<size_t>(v)][0], tag[0]);
    pairing(
        mask_1, dfe_mask_cache_[static_cast<size_t>(v)][1], tag[1]);
    const GT mask = mask_0 * mask_1;
    GT inverse_mask;
    GT::pow(inverse_mask, mask, Fr(-1));
    result *= inverse_mask;
    return result;
}

#include "pc_mcfe_client.h"

#include <stdexcept>
#include <thread>

using namespace mcl::bn;

template <class F>
static void parallel_for(int n, int threads, F f) {
    if (threads <= 1 || n <= 1) {
        for (int i = 0; i < n; ++i) f(i);
        return;
    }
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t]() {
            for (int i = t; i < n; i += threads) f(i);
        });
    }
    for (auto& thread : pool) thread.join();
}

PC_MCFE_Client::PC_MCFE_Client(
    int id, int rank, const SecLoRA_PP& global_pp,
    const DmcfePublicParams2& global_dfe_pp,
    const DmcfeClientSecret2& client_dfe_secret)
    : client_id(id), m_rank(rank), pp(global_pp),
      dfe_pp(global_dfe_pp), dfe_secret(client_dfe_secret),
      ife(rank, global_pp.g0, global_pp.g2_base) {
    if (dfe_secret.client_id != client_id) {
        throw std::invalid_argument("DMCFE client id mismatch");
    }
}

DmcfeKeyShare2 PC_MCFE_Client::KeyGenShare(
    const std::vector<Fr>& p_weights) const {
    return ABG19DmcfeMask2::DKeyShareGen(dfe_secret, p_weights);
}

static Fr int_to_fr(long long value) {
    Fr result;
    if (value >= 0) {
        result = Fr(value);
    } else {
        result = Fr(-value);
        Fr::neg(result, result);
    }
    return result;
}

void PC_MCFE_Client::SetLoraMatrices(
    const std::vector<std::vector<long long>>& A,
    const std::vector<std::vector<long long>>& B) {
    local_A.assign(A.size(), {});
    for (size_t k = 0; k < A.size(); ++k) {
        local_A[k].resize(A[k].size());
        for (size_t v = 0; v < A[k].size(); ++v) {
            local_A[k][v] = int_to_fr(A[k][v]);
        }
    }
    local_B.assign(B.size(), {});
    for (size_t u = 0; u < B.size(); ++u) {
        local_B[u].resize(B[u].size());
        for (size_t k = 0; k < B[u].size(); ++k) {
            local_B[u][k] = int_to_fr(B[u][k]);
        }
    }
}

void PC_MCFE_Client::BuildPlaintextShare(
    int eb, int ea, std::vector<std::vector<Fr>>& Bp,
    std::vector<std::vector<Fr>>& Ap) const {
    const int rows = static_cast<int>(local_B.size());
    const int cols = local_A.empty() ? 0 : static_cast<int>(local_A[0].size());
    Bp.assign(rows, std::vector<Fr>(m_rank, Fr(0)));
    for (int u = eb; u < rows; ++u) {
        for (int k = 0; k < m_rank; ++k) Bp[u][k] = local_B[u][k];
    }
    Ap.assign(m_rank, std::vector<Fr>(cols, Fr(0)));
    for (int k = 0; k < m_rank; ++k) {
        for (int v = ea; v < cols; ++v) Ap[k][v] = local_A[k][v];
    }
}

Fr PC_MCFE_Client::GetCellExpectedProduct(int u, int v) const {
    Fr sum = 0;
    for (int k = 0; k < m_rank; ++k) {
        sum += local_B[u][k] * local_A[k][v];
    }
    return sum;
}

void PC_MCFE_Client::precompute_encA_indices_mt(
    int layer_id, int pos_y, int round_q, int matrix_cols,
    const std::vector<int>& cols, int threads) {
    encA_pre.ife.resize(matrix_cols);
    encA_pre.dfe.resize(matrix_cols);
    encA_pre.x.resize(matrix_cols);

    std::vector<Fr> r1_pool(cols.size());
    std::vector<std::array<Fr, 2>> x_pool(cols.size());
    std::vector<std::array<Fr, 2>> dfe_randomness(cols.size());
    for (size_t i = 0; i < cols.size(); ++i) {
        r1_pool[i].setByCSPRNG();
        for (int channel = 0; channel < 2; ++channel) {
            x_pool[i][static_cast<size_t>(channel)].setByCSPRNG();
            dfe_randomness[i][static_cast<size_t>(channel)].setByCSPRNG();
        }
    }

    parallel_for(static_cast<int>(cols.size()), threads, [&](int index) {
        const int col = cols[static_cast<size_t>(index)];
        encA_pre.x[col] = x_pool[static_cast<size_t>(index)];

        std::vector<Fr> a_offline(2 * m_rank + 3, Fr(0));
        a_offline[2 * m_rank] =
            x_pool[static_cast<size_t>(index)][0];
        a_offline[2 * m_rank + 1] =
            x_pool[static_cast<size_t>(index)][1];
        encA_pre.ife[col] = ife.encrypt_precompute(
            a_offline, r1_pool[static_cast<size_t>(index)]);

        const std::string label =
            labelA(layer_id, pos_y, col, round_q);
        encA_pre.dfe[col] = ABG19DmcfeMask2::Encrypt(
            dfe_pp, dfe_secret, label,
            x_pool[static_cast<size_t>(index)],
            dfe_randomness[static_cast<size_t>(index)]);
    });
}

std::vector<A_Ciphertext_Slot> PC_MCFE_Client::encA_indices_mt(
    int, int, int, int matrix_cols, const std::vector<int>& cols,
    int threads) {
    std::vector<A_Ciphertext_Slot> ciphertexts(matrix_cols);
    parallel_for(static_cast<int>(cols.size()), threads, [&](int index) {
        const int col = cols[static_cast<size_t>(index)];
        std::vector<Fr> a_vector(m_rank);
        for (int k = 0; k < m_rank; ++k) {
            a_vector[k] = local_A[k][col];
        }
        const auto ciphertext =
            ife.encrypt_online(encA_pre.ife[col], a_vector);
        ciphertexts[col].ife_c1 = ciphertext.first;
        ciphertexts[col].ife_c2 = ciphertext.second;
        ciphertexts[col].dfe_ct = encA_pre.dfe[col];
    });
    return ciphertexts;
}

void PC_MCFE_Client::precompute_encB_indices_mt(
    int layer_id, int pos_y, int round_q, int matrix_rows,
    const std::vector<int>& rows, int threads) {
    encB_pre.key.resize(matrix_rows);
    std::vector<Fr> r2_pool(rows.size());
    for (Fr& value : r2_pool) value.setByCSPRNG();

    parallel_for(static_cast<int>(rows.size()), threads, [&](int index) {
        const int row = rows[static_cast<size_t>(index)];
        std::array<G2, 2> tag;
        for (int channel = 0; channel < 2; ++channel) {
            const std::string label =
                labelBTag(layer_id, pos_y, row, round_q, channel);
            hashAndMapToG2(
                tag[static_cast<size_t>(channel)],
                label.data(), label.size());
        }
        encB_pre.key[row] = ife.keygen_precompute(
            tag, r2_pool[static_cast<size_t>(index)]);
    });
}

std::vector<B_SecretKey_Slot> PC_MCFE_Client::encB_indices_mt(
    int, int, int, int matrix_rows, const std::vector<int>& rows,
    int threads) {
    std::vector<B_SecretKey_Slot> keys(matrix_rows);
    parallel_for(static_cast<int>(rows.size()), threads, [&](int index) {
        const int row = rows[static_cast<size_t>(index)];
        std::vector<Fr> b_vector(m_rank);
        for (int k = 0; k < m_rank; ++k) {
            b_vector[k] = local_B[row][k];
        }
        const auto key = ife.keygen_online(encB_pre.key[row], b_vector);
        keys[row].ife_k1 = key.first;
        keys[row].ife_k2 = key.second;
    });
    return keys;
}

void PC_MCFE_Client::ClearEncryptionPrecompute() {
    EncA_Precomp empty_a;
    EncB_Precomp empty_b;
    encA_pre = std::move(empty_a);
    encB_pre = std::move(empty_b);
}

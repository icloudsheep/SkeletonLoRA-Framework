#include "pc_mcfe_client.h"
#include <thread>

using namespace mcl::bn;

// Run f(i) for i in [0,n) across `threads` std::threads (strided). Serial if T<=1.
template <class F>
static void parallel_for(int n, int threads, F f) {
    if (threads <= 1 || n <= 1) { for (int i = 0; i < n; ++i) f(i); return; }
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t]() { for (int i = t; i < n; i += threads) f(i); });
    for (auto& th : pool) th.join();
}

PC_MCFE_Client::PC_MCFE_Client(int id, int rank, const SecLoRA_PP& global_pp)
    : client_id(id), m_rank(rank), pp(global_pp),
      ife(rank, global_pp.g0, global_pp.g2_base) {
    s_i.resize(2); s_i[0].setByCSPRNG(); s_i[1].setByCSPRNG(); r_si.setByCSPRNG();
    G1 g0_p, g1_p, h_p, temp;
    G1::mul(g0_p, pp.g0, s_i[0]);
    G1::mul(g1_p, pp.g1, s_i[1]);
    G1::mul(h_p, pp.h_blinding, r_si);
    G1::add(temp, g0_p, g1_p);
    G1::add(C_si, temp, h_p);
}

std::pair<std::vector<Fr>, Fr> PC_MCFE_Client::KeyGenShare(const std::vector<Fr>& p_weights) {
    std::vector<Fr> d_i(2);
    Fr my_weight = p_weights[client_id];
    d_i[0] = my_weight * s_i[0];
    d_i[1] = my_weight * s_i[1];
    Fr r_di = my_weight * r_si;
    return std::make_pair(d_i, r_di);
}

static Fr _int_to_fr(long long v) {
    Fr f;
    if (v >= 0) f = Fr(v);
    else { f = Fr(-v); Fr::neg(f, f); }
    return f;
}

void PC_MCFE_Client::SetLoraMatrices(const std::vector<std::vector<long long>>& A,
                                     const std::vector<std::vector<long long>>& B) {
    local_A.assign(A.size(), {});
    for (size_t k = 0; k < A.size(); ++k) {
        local_A[k].resize(A[k].size());
        for (size_t v = 0; v < A[k].size(); ++v) local_A[k][v] = _int_to_fr(A[k][v]);
    }
    local_B.assign(B.size(), {});
    for (size_t u = 0; u < B.size(); ++u) {
        local_B[u].resize(B[u].size());
        for (size_t k = 0; k < B[u].size(); ++k) local_B[u][k] = _int_to_fr(B[u][k]);
    }
}

// Selective encryption: plaintext share for S_P (encrypted region zeroed out).
void PC_MCFE_Client::BuildPlaintextShare(int eb, int ea,
                                         std::vector<std::vector<Fr>>& Bp,
                                         std::vector<std::vector<Fr>>& Ap) const {
    int m = (int)local_B.size();
    int n = local_A.empty() ? 0 : (int)local_A[0].size();
    Bp.assign(m, std::vector<Fr>(m_rank, Fr(0)));
    for (int u = eb; u < m; ++u)
        for (int k = 0; k < m_rank; ++k) Bp[u][k] = local_B[u][k];   // plaintext rows only
    Ap.assign(m_rank, std::vector<Fr>(n, Fr(0)));
    for (int k = 0; k < m_rank; ++k)
        for (int v = ea; v < n; ++v) Ap[k][v] = local_A[k][v];       // plaintext cols only
}

Fr PC_MCFE_Client::GetCellExpectedProduct(int u, int v) const {
    Fr sum = 0;
    for (int k = 0; k < m_rank; ++k) sum += local_B[u][k] * local_A[k][v];
    return sum;
}

void PC_MCFE_Client::precompute_encA_indices_mt(int layer_id, int pos_y, int round_q, int matrix_cols,
                                               const std::vector<int>& cols, int threads) {
    encA_pre.ife.resize(matrix_cols);
    encA_pre.c_i.resize(matrix_cols);
    encA_pre.u_s_only.resize(matrix_cols);
    encA_pre.x.assign(matrix_cols, Fr(0));

    std::vector<Fr> r1pool(cols.size()), xpool(cols.size());
    for (size_t i = 0; i < cols.size(); ++i) {
        r1pool[i].setByCSPRNG();
        xpool[i].setByCSPRNG();
    }

    parallel_for((int)cols.size(), threads, [&](int idx) {
        int v = cols[(size_t)idx];
        std::string l0 = labelA(layer_id, pos_y, v, round_q, 0);
        std::string l1 = labelA(layer_id, pos_y, v, round_q, 1);
        G1 ua0, ua1; hashAndMapToG1(ua0, l0.c_str(), l0.length()); hashAndMapToG1(ua1, l1.c_str(), l1.length());
        G1 ua0_s0, ua1_s1, u_s_term;
        G1::mul(ua0_s0, ua0, s_i[0]); G1::mul(ua1_s1, ua1, s_i[1]); G1::add(u_s_term, ua0_s0, ua1_s1);
        encA_pre.u_s_only[v] = u_s_term;
        encA_pre.x[v] = xpool[(size_t)idx];
        G1 x_g1; G1::mul(x_g1, pp.g0, xpool[(size_t)idx]); G1::add(encA_pre.c_i[v], u_s_term, x_g1);

        std::vector<Fr> a_offline(2 * m_rank + 2, Fr(0));
        a_offline[2 * m_rank] = xpool[(size_t)idx];
        encA_pre.ife[v] = ife.encrypt_precompute(a_offline, r1pool[(size_t)idx]);
    });
}

std::vector<A_Ciphertext_Slot> PC_MCFE_Client::encA_indices_mt(int, int, int, int matrix_cols,
                                                              const std::vector<int>& cols, int threads) {
    std::vector<A_Ciphertext_Slot> cts_list(matrix_cols);
    last_x_randomness.assign(matrix_cols, Fr(0));

    parallel_for((int)cols.size(), threads, [&](int idx) {
        int v = cols[(size_t)idx];
        std::vector<Fr> a_vec(m_rank);
        bool all_zero = true;
        for (int k = 0; k < m_rank; ++k) { a_vec[k] = local_A[k][v]; if (a_vec[k] != Fr(0)) all_zero = false; }
        if (all_zero) {
            cts_list[v].is_zero = true;
            cts_list[v].c_i = encA_pre.u_s_only[v];
            return;
        }
        cts_list[v].is_zero = false;
        last_x_randomness[v] = encA_pre.x[v];
        auto ct = ife.encrypt_online(encA_pre.ife[v], a_vec);
        cts_list[v].ife_c1 = ct.first;
        cts_list[v].ife_c2 = ct.second;
        cts_list[v].c_i = encA_pre.c_i[v];
    });
    return cts_list;
}

void PC_MCFE_Client::precompute_encB_indices_mt(int layer_id, int pos_y, int round_q, int matrix_rows,
                                               const std::vector<int>& rows, int threads) {
    encB_pre.key.resize(matrix_rows);
    encB_pre.k_hat.resize(matrix_rows);

    std::vector<Fr> r2pool(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) r2pool[i].setByCSPRNG();

    parallel_for((int)rows.size(), threads, [&](int idx) {
        int u = rows[(size_t)idx];
        std::string lb = labelB(layer_id, pos_y, u, round_q);
        G2 t_lb; hashAndMapToG2(t_lb, lb.c_str(), lb.length());
        encB_pre.k_hat[u] = ife.keygen_zero(t_lb);
        encB_pre.key[u] = ife.keygen_precompute(t_lb, r2pool[(size_t)idx]);
    });
}

std::vector<B_SecretKey_Slot> PC_MCFE_Client::encB_indices_mt(int, int, int, int matrix_rows,
                                                             const std::vector<int>& rows, int threads) {
    std::vector<B_SecretKey_Slot> sks_list(matrix_rows);

    parallel_for((int)rows.size(), threads, [&](int idx) {
        int u = rows[(size_t)idx];
        std::vector<Fr> b_vec(m_rank);
        bool all_zero = true;
        for (int k = 0; k < m_rank; ++k) { b_vec[k] = local_B[u][k]; if (b_vec[k] != Fr(0)) all_zero = false; }
        if (all_zero) {
            sks_list[u].is_zero = true;
            sks_list[u].ife_k1 = encB_pre.k_hat[u];
            sks_list[u].ife_k2.clear();
            return;
        }
        sks_list[u].is_zero = false;
        auto sk = ife.keygen_online(encB_pre.key[u], b_vec);
        sks_list[u].ife_k1 = sk.first;
        sks_list[u].ife_k2 = sk.second;
    });
    return sks_list;
}

void PC_MCFE_Client::ClearEncryptionPrecompute() {
    EncA_Precomp empty_a;
    EncB_Precomp empty_b;
    encA_pre = std::move(empty_a);
    encB_pre = std::move(empty_b);
}

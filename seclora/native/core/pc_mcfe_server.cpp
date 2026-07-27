#include "pc_mcfe_server.h"
#include <algorithm>
#include <string>
#include <cmath>
#include <chrono>

using namespace mcl::bn;

static inline double now_sec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

PC_MCFE_Server::PC_MCFE_Server(int K, const SecLoRA_PP& global_pp)
    : K_clients(K), pp(global_pp) {}

// Decentralized key aggregation: d = sum p_i s_i, r_d = sum p_i r_{s,i}.
std::pair<std::vector<Fr>, Fr> PC_MCFE_Server::AggregateKeys(
    const std::vector<std::pair<std::vector<Fr>, Fr>>& client_shares) {
    std::vector<Fr> d(2, Fr(0));
    Fr r_d = 0;
    for (int i = 0; i < K_clients; ++i) {
        d[0] += client_shares[i].first[0];
        d[1] += client_shares[i].first[1];
        r_d += client_shares[i].second;
    }
    return std::make_pair(d, r_d);
}

// VerifyKey: prod C_{s,i}^{p_i} == g0^{d0} g1^{d1} h^{r_d}.
bool PC_MCFE_Server::VerifyKey(const std::vector<Fr>& p_weights,
                               const std::vector<Fr>& dk_d, const Fr& r_d) {
    G1 left; left.clear();
    for (int i = 0; i < K_clients; ++i) {
        G1 term; G1::mul(term, pp.cmt_s[i], p_weights[i]);
        G1::add(left, left, term);
    }
    G1 right, g0p, g1p, hp, tmp;
    G1::mul(g0p, pp.g0, dk_d[0]); G1::mul(g1p, pp.g1, dk_d[1]); G1::mul(hp, pp.h_blinding, r_d);
    G1::add(tmp, g0p, g1p); G1::add(right, tmp, hp);
    return (left == right);
}

// Build the baby-step table for a fixed base. baby[base^j] = j for j in [0, m),
// m = ceil(sqrt(bound)); giant_inv = base^{-m}. Depends only on base, so it is
// built once and reused for every protected skeleton cell.
void PC_MCFE_Server::build_bsgs(
    const GT& base,
    long long bound,
    const std::function<void(long long, long long)>& progress) {
    base_gt_ = base;
    bsgs_.bound = bound;
    bsgs_.m = (long long)std::ceil(std::sqrt((double)bound));
    bsgs_.baby.clear();
    bsgs_.baby.reserve((size_t)bsgs_.m * 2);

    GT cur; GT::pow(cur, base, Fr(0));            // base^0 = identity
    const long long progress_step = std::max(1LL, bsgs_.m / 20);
    for (long long j = 0; j < bsgs_.m; ++j) {
        bsgs_.baby.emplace(cur.getStr(mcl::IoSerialize), j);
        cur *= base;                             // cur = base^{j+1}
        if (progress && ((j + 1) % progress_step == 0 || j + 1 == bsgs_.m)) {
            progress(j + 1, bsgs_.m);
        }
    }
    GT::inv(bsgs_.giant_inv, cur);               // cur == base^m -> base^{-m}
    bsgs_.built = true;
}

long long PC_MCFE_Server::bsgs_table_bytes_estimate() const {
    long long key_bytes = base_gt_.getStr(mcl::IoSerialize).size();
    return (long long)bsgs_.baby.size() * (key_bytes + (long long)sizeof(long long));
}

// Bidirectional BSGS over GT. Solve base^e == target for e in [-bound, bound].
// Positive branch: target * base^{-i m} == base^j  -> e =  (i m + j).
// Negative branch: same against target^{-1}         -> e = -(i m + j).
// Both branches are interleaved so small |e| are found at i = 0.
long long PC_MCFE_Server::bsgs_search(const GT& target, bool& found) const {
    found = false;
    GT gpos = target;
    GT gneg; GT::inv(gneg, target);
    for (long long i = 0; i <= bsgs_.m; ++i) {
        auto itp = bsgs_.baby.find(gpos.getStr(mcl::IoSerialize));
        if (itp != bsgs_.baby.end()) {
            long long e = i * bsgs_.m + itp->second;
            if (e <= bsgs_.bound) { found = true; return e; }
        }
        auto itn = bsgs_.baby.find(gneg.getStr(mcl::IoSerialize));
        if (itn != bsgs_.baby.end()) {
            long long e = i * bsgs_.m + itn->second;
            if (e <= bsgs_.bound) { found = true; return -e; }
        }
        gpos *= bsgs_.giant_inv;
        gneg *= bsgs_.giant_inv;
    }
    return 0;
}

// Decrypt one protected aggregate cell DeltaW[u,v]. Assumes build_bsgs() already ran.
int PC_MCFE_Server::decrypt_one_cell(
    const std::vector<std::vector<A_Ciphertext_Slot>>& all_A_cts,
    const std::vector<std::vector<B_SecretKey_Slot>>& all_B_sks,
    const std::vector<Fr>& p_weights,
    const std::pair<std::vector<Fr>, Fr>& dk_d,
    int layer_id, int pos_y, int round_q, int u, int v, bool& found) {

    GT v_group = eval_one_cell_group(all_A_cts, all_B_sks, p_weights, dk_d,
                                     layer_id, pos_y, round_q, u, v);
    return (int)bsgs_search(v_group, found);
}

GT PC_MCFE_Server::eval_one_cell_group(
    const std::vector<std::vector<A_Ciphertext_Slot>>& all_A_cts,
    const std::vector<std::vector<B_SecretKey_Slot>>& all_B_sks,
    const std::vector<Fr>& p_weights,
    const std::pair<std::vector<Fr>, Fr>& dk_d,
    int layer_id, int pos_y, int round_q, int u, int v) {

    std::vector<const std::vector<A_Ciphertext_Slot>*> a_refs;
    std::vector<const std::vector<B_SecretKey_Slot>*> b_refs;
    a_refs.reserve(all_A_cts.size());
    b_refs.reserve(all_B_sks.size());
    for (const auto& slots : all_A_cts) a_refs.push_back(&slots);
    for (const auto& slots : all_B_sks) b_refs.push_back(&slots);
    return eval_one_cell_group_refs(
        a_refs, b_refs, p_weights, dk_d,
        layer_id, pos_y, round_q, u, v);
}

GT PC_MCFE_Server::eval_one_cell_group_refs(
    const std::vector<const std::vector<A_Ciphertext_Slot>*>& all_A_cts,
    const std::vector<const std::vector<B_SecretKey_Slot>*>& all_B_sks,
    const std::vector<Fr>& p_weights,
    const std::pair<std::vector<Fr>, Fr>& dk_d,
    int layer_id, int pos_y, int round_q, int u, int v) {

    std::string lb = labelB(layer_id, pos_y, u, round_q);
    G2 t_lb; hashAndMapToG2(t_lb, lb.c_str(), lb.length());

    GT v_group; GT::pow(v_group, base_gt_, Fr(0));
    G1 combined; combined.clear();

    for (int i = 0; i < K_clients; ++i) {
        const auto& ct_v = (*all_A_cts[i])[v];
        const auto& sk_u = (*all_B_sks[i])[u];
        Fr p_i = p_weights[i];

        G1 wc; G1::mul(wc, ct_v.c_i, p_i); combined += wc;
        if (ct_v.is_zero) continue;

        GT local;
        if (sk_u.is_zero) {
            const auto& c2 = ct_v.ife_c2;
            int sz = (int)c2.size();
            GT p1, p2;
            pairing(p1, c2[0], sk_u.ife_k1);
            pairing(p2, c2[sz - 2], t_lb);
            local = p1 * p2;
        } else {
            pairing(local, ct_v.ife_c1, sk_u.ife_k1);
            for (size_t j = 0; j < ct_v.ife_c2.size(); ++j) {
                GT bp; pairing(bp, ct_v.ife_c2[j], sk_u.ife_k2[j]); local *= bp;
            }
        }
        GT w; GT::pow(w, local, p_i); v_group *= w;
    }

    std::string a0 = labelA(layer_id, pos_y, v, round_q, 0);
    std::string a1 = labelA(layer_id, pos_y, v, round_q, 1);
    G1 ua0, ua1; hashAndMapToG1(ua0, a0.c_str(), a0.length()); hashAndMapToG1(ua1, a1.c_str(), a1.length());
    G1 ud0, ud1, u_d;
    G1::mul(ud0, ua0, dk_d.first[0]); G1::mul(ud1, ua1, dk_d.first[1]); G1::add(u_d, ud0, ud1);
    G1 isolated; G1::sub(isolated, combined, u_d);

    GT noise; pairing(noise, isolated, t_lb);
    GT inv_noise; GT::pow(inv_noise, noise, Fr(-1));
    v_group *= inv_noise;

    return v_group;
}

int PC_MCFE_Server::decrypt_one_cell_timed(
    const std::vector<std::vector<A_Ciphertext_Slot>>& all_A_cts,
    const std::vector<std::vector<B_SecretKey_Slot>>& all_B_sks,
    const std::vector<Fr>& p_weights,
    const std::pair<std::vector<Fr>, Fr>& dk_d,
    int layer_id, int pos_y, int round_q, int u, int v, bool& found,
    double& bsgs_sec) {

    GT v_group = eval_one_cell_group(all_A_cts, all_B_sks, p_weights, dk_d,
                                     layer_id, pos_y, round_q, u, v);

    double t0 = now_sec();
    int ret = (int)bsgs_search(v_group, found);
    bsgs_sec = now_sec() - t0;
    return ret;
}

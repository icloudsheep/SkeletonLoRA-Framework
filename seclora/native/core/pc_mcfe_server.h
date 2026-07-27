#ifndef PC_MCFE_SERVER_H
#define PC_MCFE_SERVER_H

#include <vector>
#include <utility>
#include <string>
#include <functional>
#include <unordered_map>
#include <mcl/bn.hpp>
#include "seclora_types.h"

// PC-MCFE server: aggregates KeyGen shares and decrypts the weighted sum
// of the clients' B*A products without seeing any individual matrix.
class PC_MCFE_Server {
private:
    int K_clients;
    const SecLoRA_PP& pp;

    // Baby-step / giant-step table for GT discrete log. Built once per fixed base
    // (base = e(g0, g2_base) is constant across the whole decrypt), reused for
    // every cell; supports negative exponents via a bidirectional search.
    struct BSGSTable {
        long long m = 0;        // step size = ceil(sqrt(bound))
        long long bound = 0;    // |exponent| search bound
        mcl::bn::GT giant_inv;  // base^{-m}
        std::unordered_map<std::string, long long> baby;  // base^j -> j, j in [0,m)
        bool built = false;
    };
    BSGSTable bsgs_;
    mcl::bn::GT base_gt_;       // cached e(g0, g2_base), set by build_bsgs

public:
    PC_MCFE_Server(int K, const SecLoRA_PP& global_pp);

    // Build / reuse the BSGS table for the given base and search bound.
    void build_bsgs(
        const mcl::bn::GT& base,
        long long bound,
        const std::function<void(long long, long long)>& progress = {});
    long long bsgs_bound() const { return bsgs_.bound; }
    long long bsgs_step() const { return bsgs_.m; }
    long long bsgs_table_entries() const { return (long long)bsgs_.baby.size(); }
    long long bsgs_table_bytes_estimate() const;
    // Solve base^e == target for e in [-bound, bound]; sets found.
    long long bsgs_search(const mcl::bn::GT& target, bool& found) const;

    // Decrypt a single ΔW cell (u,v) using the baseline math (no projection / no
    // batching). Requires build_bsgs() to have been called first. Used by the
    // standalone skeleton-decryption test.
    int decrypt_one_cell(
        const std::vector<std::vector<A_Ciphertext_Slot>>& all_A_cts,
        const std::vector<std::vector<B_SecretKey_Slot>>& all_B_sks,
        const std::vector<mcl::bn::Fr>& p_weights,
        const std::pair<std::vector<mcl::bn::Fr>, mcl::bn::Fr>& dk_d,
        int layer_id, int pos_y, int round_q, int u, int v, bool& found);
    mcl::bn::GT eval_one_cell_group(
        const std::vector<std::vector<A_Ciphertext_Slot>>& all_A_cts,
        const std::vector<std::vector<B_SecretKey_Slot>>& all_B_sks,
        const std::vector<mcl::bn::Fr>& p_weights,
        const std::pair<std::vector<mcl::bn::Fr>, mcl::bn::Fr>& dk_d,
        int layer_id, int pos_y, int round_q, int u, int v);
    mcl::bn::GT eval_one_cell_group_refs(
        const std::vector<const std::vector<A_Ciphertext_Slot>*>& all_A_cts,
        const std::vector<const std::vector<B_SecretKey_Slot>*>& all_B_sks,
        const std::vector<mcl::bn::Fr>& p_weights,
        const std::pair<std::vector<mcl::bn::Fr>, mcl::bn::Fr>& dk_d,
        int layer_id, int pos_y, int round_q, int u, int v);
    bool group_encodes(
        const mcl::bn::GT& encoded,
        const mcl::bn::Fr& exponent) const;
    int decrypt_one_cell_timed(
        const std::vector<std::vector<A_Ciphertext_Slot>>& all_A_cts,
        const std::vector<std::vector<B_SecretKey_Slot>>& all_B_sks,
        const std::vector<mcl::bn::Fr>& p_weights,
        const std::pair<std::vector<mcl::bn::Fr>, mcl::bn::Fr>& dk_d,
        int layer_id, int pos_y, int round_q, int u, int v, bool& found,
        double& bsgs_sec);

    std::pair<std::vector<mcl::bn::Fr>, mcl::bn::Fr> AggregateKeys(
        const std::vector<std::pair<std::vector<mcl::bn::Fr>, mcl::bn::Fr>>& client_shares);

    bool VerifyKey(const std::vector<mcl::bn::Fr>& p_weights,
                   const std::vector<mcl::bn::Fr>& dk_d, const mcl::bn::Fr& r_d);
};

#endif  // PC_MCFE_SERVER_H

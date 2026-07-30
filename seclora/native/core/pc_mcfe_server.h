#ifndef PC_MCFE_SERVER_H
#define PC_MCFE_SERVER_H

#include <array>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <mcl/bn.hpp>

#include "seclora_types.h"

// PC-DMCFE server: combines decentralized key shares and decrypts weighted
// sums of the clients' B*A products without seeing any individual matrix.
class PC_MCFE_Server {
private:
    int K_clients;
    const SecLoRA_PP& pp;
    const DmcfePublicParams2& dfe_pp;
    std::vector<std::array<mcl::bn::G1, 2>> dfe_mask_cache_;
    std::vector<char> dfe_mask_ready_;

    struct BSGSTable {
        long long m = 0;
        long long bound = 0;
        mcl::bn::GT giant_inv;
        std::unordered_map<std::string, long long> baby;
        bool built = false;
    };
    BSGSTable bsgs_;
    mcl::bn::GT base_gt_;

public:
    PC_MCFE_Server(int K, const SecLoRA_PP& global_pp,
                   const DmcfePublicParams2& global_dfe_pp);

    void build_bsgs(
        const mcl::bn::GT& base,
        long long bound,
        const std::function<void(long long, long long)>& progress = {});
    long long bsgs_bound() const { return bsgs_.bound; }
    long long bsgs_step() const { return bsgs_.m; }
    long long bsgs_table_entries() const {
        return static_cast<long long>(bsgs_.baby.size());
    }
    long long bsgs_table_bytes_estimate() const;
    long long bsgs_search(
        const mcl::bn::GT& target, bool& found) const;

    DmcfeFunctionalKey2 DKeyComb(
        const std::vector<DmcfeKeyShare2>& client_shares) const;

    // The end-to-end session stores each client's layer payload separately.
    // Reference views avoid copying all ciphertext slots before aggregation.
    double PrepareDfeMaskCacheRefs(
        const std::vector<const std::vector<A_Ciphertext_Slot>*>& all_A_cts,
        const std::vector<mcl::bn::Fr>& p_weights,
        const DmcfeFunctionalKey2& key,
        const std::vector<int>& columns,
        int threads,
        double& worker_thread_sum_sec);

    mcl::bn::GT eval_one_cell_group_refs(
        const std::vector<const std::vector<A_Ciphertext_Slot>*>& all_A_cts,
        const std::vector<const std::vector<B_SecretKey_Slot>*>& all_B_sks,
        const std::vector<mcl::bn::Fr>& p_weights,
        int layer_id, int pos_y, int round_q, int u, int v);
};

#endif  // PC_MCFE_SERVER_H

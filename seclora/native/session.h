#ifndef SECLORA_NATIVE_SESSION_H
#define SECLORA_NATIVE_SESSION_H

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/pc_mcfe_client.h"
#include "core/pc_mcfe_server.h"

struct FloatLayerInput {
    int layer_id = 0;
    std::string name;
    int rows = 0;
    int cols = 0;
    std::vector<float> a;  // rank x cols
    std::vector<float> b;  // rows x rank
};

struct NativeLayerUpload {
    int layer_id = 0;
    std::string name;
    int rows = 0;
    int cols = 0;
    int encrypted_b_rows = 0;
    int encrypted_a_cols = 0;
    std::vector<int> candidate_rows;
    std::vector<int> candidate_cols;

    // S_P package: only the clear slices are present.
    std::vector<std::vector<long long>> plain_b;
    std::vector<std::vector<long long>> plain_a;

    // S_D package: unselected slots remain unused.
    std::vector<A_Ciphertext_Slot> encrypted_a;
    std::vector<B_SecretKey_Slot> encrypted_b;

    // One compressed PC-DMCFE pair for beta^T * DeltaW * alpha.
    std::vector<A_Ciphertext_Slot> projection_a;
    std::vector<B_SecretKey_Slot> projection_b;
    std::size_t serialized_size_bytes = 0;
};

struct NativeClientUpdate {
    int client_id = 0;
    int round_id = 0;
    std::vector<NativeLayerUpload> layers;
    std::size_t serialized_size_bytes = 0;
};

struct NativeLayerSkeleton {
    int layer_id = 0;
    int rows = 0;
    int cols = 0;
    std::vector<std::vector<long long>> c;
    std::vector<std::vector<long long>> m;
    std::vector<std::vector<long long>> s;
    int selected_rank = 0;
    int projection_checks = 0;
    std::size_t decrypted_cells = 0;
};

class SelectiveTwoServerSession {
public:
    SelectiveTwoServerSession(int num_clients, int rank, double ratio,
                              int sfp, double xmax, int threads);

    std::shared_ptr<NativeClientUpdate> encrypt_client(
        int client_id, int round_id, const std::vector<FloatLayerInput>& layers);

    std::vector<NativeLayerSkeleton> aggregate_round(
        int round_id,
        const std::vector<std::shared_ptr<NativeClientUpdate>>& updates);

    void close();

private:
    int num_clients_;
    int rank_;
    double ratio_;
    int sfp_;
    long long scale_;
    double xmax_;
    long long encoded_bound_;
    long long bsgs_bound_;
    int threads_;
    bool closed_ = false;

    SecLoRA_PP pp_;
    std::vector<std::unique_ptr<PC_MCFE_Client>> clients_;
    std::unique_ptr<PC_MCFE_Server> server_;
    std::vector<mcl::bn::Fr> weights_;
    std::pair<std::vector<mcl::bn::Fr>, mcl::bn::Fr> aggregate_key_;

    struct ProjectionChallenge {
        int rows = 0;
        int cols = 0;
        std::vector<mcl::bn::Fr> alpha;
        std::vector<mcl::bn::Fr> beta;
    };
    std::map<std::pair<int, int>, ProjectionChallenge> projection_challenges_;

    void require_open() const;
    std::vector<std::vector<long long>> quantize_a(
        const FloatLayerInput& layer) const;
    std::vector<std::vector<long long>> quantize_b(
        const FloatLayerInput& layer) const;
};

#endif  // SECLORA_NATIVE_SESSION_H

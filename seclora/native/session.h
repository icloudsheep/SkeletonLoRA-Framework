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
    std::size_t sp_plain_bytes = 0;
    std::size_t sd_cipher_bytes = 0;
    std::size_t serialized_size_bytes = 0;
    double quantize_pack_wall_sec = 0.0;
    double precompute_wall_sec = 0.0;
    double online_crypto_wall_sec = 0.0;
    double serialize_wall_sec = 0.0;
};

struct NativeClientUpdate {
    int client_id = 0;
    int round_id = 0;
    std::vector<NativeLayerUpload> layers;
    std::size_t sp_plain_bytes = 0;
    std::size_t sd_cipher_bytes = 0;
    std::size_t serialized_size_bytes = 0;
    std::size_t protected_b_labels = 0;
    std::size_t protected_a_labels = 0;
    std::size_t candidate_b_labels = 0;
    std::size_t candidate_a_labels = 0;
    double binding_input_copy_wall_sec = 0.0;
    double quantize_pack_wall_sec = 0.0;
    double precompute_wall_sec = 0.0;
    double online_crypto_wall_sec = 0.0;
    double serialize_wall_sec = 0.0;
};

struct NativeLayerSkeleton {
    int layer_id = 0;
    int rows = 0;
    int cols = 0;
    std::vector<std::vector<long long>> c;
    std::vector<std::vector<long long>> m;
    std::vector<std::vector<long long>> s;
    int selected_rank = 0;
    std::vector<int> pivot_rows;
    std::vector<int> pivot_cols;
    int baseline_checks = 0;
    double baseline_relative_error = 0.0;
    std::size_t decrypted_cells = 0;
    std::size_t pivot_candidate_cells = 0;
    std::size_t download_c_bytes = 0;
    std::size_t download_m_bytes = 0;
    std::size_t download_s_bytes = 0;
};

struct NativeRoundMetrics {
    std::string mode;
    double sp_wall_sec = 0.0;
    double sd_wall_sec = 0.0;
    double sd_dfe_mask_wall_sec = 0.0;
    double sd_fe_eval_wall_sec = 0.0;
    double sd_bsgs_search_wall_sec = 0.0;
    double sd_control_wall_sec = 0.0;
    double cur_skeleton_wall_sec = 0.0;
    double cur_reconstruct_wall_sec = 0.0;
    double experiment_verify_wall_sec = 0.0;
    double server_common_control_wall_sec = 0.0;
    double observed_serial_server_wall_sec = 0.0;
    std::size_t protected_skeleton_cells = 0;
    std::size_t pivot_candidate_cells = 0;
    std::size_t download_c_bytes_per_client = 0;
    std::size_t download_m_bytes_per_client = 0;
    std::size_t download_s_bytes_per_client = 0;
    std::size_t download_bytes_per_client = 0;
};

class SelectiveTwoServerSession {
public:
    SelectiveTwoServerSession(int num_clients, int rank, double ratio,
                              int sfp, double xmax, int threads,
                              const std::string& mode = "sel-2s");

    std::shared_ptr<NativeClientUpdate> encrypt_client(
        int client_id, int round_id, const std::vector<FloatLayerInput>& layers);

    std::vector<NativeLayerSkeleton> aggregate_round(
        int round_id,
        const std::vector<std::shared_ptr<NativeClientUpdate>>& updates);

    const NativeRoundMetrics& last_round_metrics() const {
        return last_round_metrics_;
    }

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
    std::string mode_;
    bool full_sk_ = false;
    bool closed_ = false;
    NativeRoundMetrics last_round_metrics_;

    SecLoRA_PP pp_;
    DmcfePublicParams2 dfe_pp_;
    std::vector<std::unique_ptr<PC_MCFE_Client>> clients_;
    std::unique_ptr<PC_MCFE_Server> server_;
    std::vector<mcl::bn::Fr> weights_;
    DmcfeFunctionalKey2 aggregate_key_;

    struct PlaintextOracleLayer {
        int rows = 0;
        int cols = 0;
        std::vector<unsigned char> present;
        std::vector<std::vector<std::vector<long long>>> client_a;
        std::vector<std::vector<std::vector<long long>>> client_b;
    };
    std::map<std::pair<int, int>, PlaintextOracleLayer> plaintext_oracles_;

    void require_open() const;
    std::vector<std::vector<long long>> quantize_a(
        const FloatLayerInput& layer) const;
    std::vector<std::vector<long long>> quantize_b(
        const FloatLayerInput& layer) const;
};

#endif  // SECLORA_NATIVE_SESSION_H

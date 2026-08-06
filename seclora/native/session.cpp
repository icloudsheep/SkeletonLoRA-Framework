#include "session.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

using namespace mcl::bn;
using IntMat = std::vector<std::vector<long long>>;
using FrMat = std::vector<std::vector<Fr>>;
using Real = long double;
using RealMat = std::vector<std::vector<Real>>;

namespace {

constexpr Real kBaselineRelativeTolerance = 1e-8L;

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point& started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

template <class F>
void parallel_for(int count, int threads, const F& fn) {
    if (threads <= 1 || count <= 1) {
        for (int i = 0; i < count; ++i) fn(i);
        return;
    }
    const int worker_count = std::min(count, threads);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker]() {
            for (int i = worker; i < count; i += worker_count) fn(i);
        });
    }
    for (auto& worker : workers) worker.join();
}

Fr ll_to_fr(long long value) {
    if (value >= 0) return Fr(static_cast<int64_t>(value));
    Fr result(static_cast<int64_t>(-value));
    Fr::neg(result, result);
    return result;
}

template <class Cell>
std::pair<std::vector<int>, std::vector<int>> select_square_pivots(
    const std::vector<int>& row_candidates,
    const std::vector<int>& col_candidates,
    int rank_cap,
    const Cell& cell) {
    const int rows = static_cast<int>(row_candidates.size());
    const int cols = static_cast<int>(col_candidates.size());
    // Complete pivoting keeps every prefix invertible over Fr while preferring
    // a numerically stable real-valued skeleton core.
    FrMat field_work(rows, std::vector<Fr>(cols));
    RealMat real_work(rows, std::vector<Real>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const long long value =
                cell(row_candidates[i], col_candidates[j]);
            field_work[i][j] = ll_to_fr(value);
            real_work[i][j] = static_cast<Real>(value);
        }
    }

    std::vector<int> row_ids = row_candidates;
    std::vector<int> col_ids = col_candidates;
    const int limit = std::min(rank_cap, std::min(rows, cols));
    int rank = 0;
    while (rank < limit) {
        int selected_row = -1;
        int selected_col = -1;
        Real selected_magnitude = -1;
        for (int i = rank; i < rows; ++i) {
            for (int j = rank; j < cols; ++j) {
                if (!field_work[i][j].isZero() &&
                    std::fabs(real_work[i][j]) > selected_magnitude) {
                    selected_row = i;
                    selected_col = j;
                    selected_magnitude = std::fabs(real_work[i][j]);
                }
            }
        }
        if (selected_row < 0) break;

        std::swap(field_work[rank], field_work[selected_row]);
        std::swap(real_work[rank], real_work[selected_row]);
        std::swap(row_ids[rank], row_ids[selected_row]);
        for (int i = 0; i < rows; ++i) {
            std::swap(field_work[i][rank], field_work[i][selected_col]);
            std::swap(real_work[i][rank], real_work[i][selected_col]);
        }
        std::swap(col_ids[rank], col_ids[selected_col]);

        Fr field_inverse;
        Fr::inv(field_inverse, field_work[rank][rank]);
        const Real real_pivot = real_work[rank][rank];
        for (int i = rank + 1; i < rows; ++i) {
            if (!field_work[i][rank].isZero()) {
                const Fr factor = field_work[i][rank] * field_inverse;
                for (int j = rank; j < cols; ++j) {
                    field_work[i][j] -=
                        factor * field_work[rank][j];
                }
            }
            if (real_pivot == 0 || real_work[i][rank] == 0) continue;
            const Real factor = real_work[i][rank] / real_pivot;
            for (int j = rank; j < cols; ++j) {
                real_work[i][j] -= factor * real_work[rank][j];
            }
        }
        ++rank;
    }

    row_ids.resize(rank);
    col_ids.resize(rank);
    return std::make_pair(std::move(row_ids), std::move(col_ids));
}

RealMat solve_nonsingular_real(
    const IntMat& matrix_values,
    const IntMat& rhs_values,
    int rank,
    int cols) {
    RealMat matrix(rank, std::vector<Real>(rank));
    RealMat rhs(rank, std::vector<Real>(cols));
    for (int row = 0; row < rank; ++row) {
        for (int col = 0; col < rank; ++col) {
            matrix[row][col] =
                static_cast<Real>(matrix_values[row][col]);
        }
        for (int col = 0; col < cols; ++col) {
            rhs[row][col] =
                static_cast<Real>(rhs_values[row][col]);
        }
    }

    for (int col = 0; col < rank; ++col) {
        int pivot = col;
        for (int row = col + 1; row < rank; ++row) {
            if (std::fabs(matrix[row][col]) >
                std::fabs(matrix[pivot][col])) {
                pivot = row;
            }
        }
        if (matrix[pivot][col] == 0) {
            throw std::runtime_error(
                "selected skeleton core is singular over the reals");
        }
        if (pivot != col) {
            std::swap(matrix[pivot], matrix[col]);
            std::swap(rhs[pivot], rhs[col]);
        }

        const Real inverse = 1 / matrix[col][col];
        for (int next = col; next < rank; ++next) {
            matrix[col][next] *= inverse;
        }
        for (int rhs_col = 0; rhs_col < cols; ++rhs_col) {
            rhs[col][rhs_col] *= inverse;
        }

        for (int row = 0; row < rank; ++row) {
            if (row == col || matrix[row][col] == 0) continue;
            const Real factor = matrix[row][col];
            for (int next = col; next < rank; ++next) {
                matrix[row][next] -= factor * matrix[col][next];
            }
            for (int rhs_col = 0; rhs_col < cols; ++rhs_col) {
                rhs[row][rhs_col] -= factor * rhs[col][rhs_col];
            }
        }
    }
    return rhs;
}

Real gram_product_sum(const RealMat& left, const RealMat& right) {
    Real result = 0;
    for (std::size_t row = 0; row < left.size(); ++row) {
        for (std::size_t col = 0; col < left[row].size(); ++col) {
            result += left[row][col] * right[row][col];
        }
    }
    return result;
}

uint64_t mix_index(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::vector<int> public_candidates(
    int begin, int end, int count, uint64_t seed) {
    std::vector<int> result;
    for (int index = begin; index < end; ++index) result.push_back(index);
    std::sort(result.begin(), result.end(), [seed](int left, int right) {
        const uint64_t left_hash = mix_index(
            static_cast<uint64_t>(static_cast<unsigned>(left)) + seed);
        const uint64_t right_hash = mix_index(
            static_cast<uint64_t>(static_cast<unsigned>(right)) + seed);
        return left_hash == right_hash ? left < right : left_hash < right_hash;
    });
    if (static_cast<int>(result.size()) > count) result.resize(count);
    std::sort(result.begin(), result.end());
    return result;
}

void append_unique(std::vector<int>& values, int value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

std::size_t a_slot_bytes(const A_Ciphertext_Slot& slot) {
    std::size_t bytes =
        slot.ife_c1.getStr(mcl::IoSerialize).size();
    for (const auto& value : slot.ife_c2) {
        bytes += value.getStr(mcl::IoSerialize).size();
    }
    bytes += static_cast<std::size_t>(
        ABG19DmcfeMask2::SerializedBytes(slot.dfe_ct));
    return bytes;
}

std::size_t b_slot_bytes(const B_SecretKey_Slot& slot) {
    std::size_t bytes = slot.ife_k1.getStr(mcl::IoSerialize).size();
    for (const auto& value : slot.ife_k2) {
        bytes += value.getStr(mcl::IoSerialize).size();
    }
    return bytes;
}

void append_wire_i64(std::vector<unsigned char>& payload, long long value) {
    const uint64_t encoded = static_cast<uint64_t>(value);
    for (int shift = 56; shift >= 0; shift -= 8) {
        payload.push_back(
            static_cast<unsigned char>((encoded >> shift) & 0xffU));
    }
}

std::size_t serialize_plain_slices(
    const IntMat& plain_b, const IntMat& plain_a) {
    std::size_t values = 0;
    for (const auto& row : plain_b) values += row.size();
    for (const auto& row : plain_a) values += row.size();
    std::vector<unsigned char> payload;
    payload.reserve(values * 8);
    for (const auto& row : plain_b) {
        for (long long value : row) append_wire_i64(payload, value);
    }
    for (const auto& row : plain_a) {
        for (long long value : row) append_wire_i64(payload, value);
    }
    return payload.size();
}

void serialize_skeleton_parts(NativeLayerSkeleton& skeleton) {
    std::vector<unsigned char> c_payload;
    std::vector<unsigned char> m_payload;
    std::vector<unsigned char> s_payload;
    const int rank = skeleton.selected_rank;
    for (int row = 0; row < skeleton.rows; ++row) {
        if (std::find(
                skeleton.pivot_rows.begin(), skeleton.pivot_rows.end(), row) !=
            skeleton.pivot_rows.end()) {
            continue;
        }
        for (int col = 0; col < rank; ++col) {
            append_wire_i64(c_payload, skeleton.c[row][col]);
        }
    }
    for (int row = 0; row < rank; ++row) {
        for (int col = 0; col < rank; ++col) {
            append_wire_i64(m_payload, skeleton.m[row][col]);
        }
    }
    for (int row = 0; row < rank; ++row) {
        for (int col = 0; col < skeleton.cols; ++col) {
            if (std::find(
                    skeleton.pivot_cols.begin(), skeleton.pivot_cols.end(), col) !=
                skeleton.pivot_cols.end()) {
                continue;
            }
            append_wire_i64(s_payload, skeleton.s[row][col]);
        }
    }
    skeleton.download_c_bytes = c_payload.size();
    skeleton.download_m_bytes = m_payload.size();
    skeleton.download_s_bytes = s_payload.size();
}

long long checked_int64(__int128 value, const char* context) {
    if (value < std::numeric_limits<long long>::min() ||
        value > std::numeric_limits<long long>::max()) {
        throw std::overflow_error(std::string(context) + " exceeds int64");
    }
    return static_cast<long long>(value);
}

}  // namespace

SelectiveTwoServerSession::SelectiveTwoServerSession(
    int num_clients, int rank, double ratio, int sfp, double xmax, int threads,
    const std::string& mode)
    : num_clients_(num_clients),
      rank_(rank),
      ratio_(ratio),
      sfp_(sfp),
      scale_(sfp >= 0 && sfp < 63 ? (1LL << sfp) : 0),
      xmax_(xmax),
      encoded_bound_(0),
      bsgs_bound_(0),
      threads_(threads),
      mode_(mode),
      full_sk_(mode == "full-sk") {
    if (num_clients_ <= 0 || rank_ <= 0) {
        throw std::invalid_argument("num_clients and rank must be positive");
    }
    if (mode_ != "sel-2s" && mode_ != "full-sk") {
        throw std::invalid_argument("mode must be sel-2s or full-sk");
    }
    if (!full_sk_ && !(ratio_ >= 0.0 && ratio_ < 1.0)) {
        throw std::invalid_argument("SEL-2S ratio must be in [0, 1)");
    }
    if (full_sk_) ratio_ = 1.0;
    if (sfp_ < 1 || sfp_ > 30 || threads_ <= 0) {
        throw std::invalid_argument("sfp must be in [1, 30] and threads positive");
    }
    if (!std::isfinite(xmax_) || xmax_ <= 0.0) {
        throw std::invalid_argument("xmax must be finite and positive");
    }

    static std::once_flag pairing_once;
    std::call_once(pairing_once, []() { mcl::bn::initPairing(mcl::BN254); });

    const long double encoded =
        std::ceil(static_cast<long double>(scale_) * xmax_);
    if (encoded < 1 ||
        encoded > static_cast<long double>(std::numeric_limits<long long>::max())) {
        throw std::overflow_error("encoded clipping bound exceeds int64");
    }
    encoded_bound_ = static_cast<long long>(encoded);
    const __int128 wide_bound =
        static_cast<__int128>(encoded_bound_) * encoded_bound_ *
        num_clients_ * rank_;
    bsgs_bound_ = checked_int64(wide_bound, "public BSGS bound");

    hashAndMapToG1(pp_.g0, "base_g0", 7);
    hashAndMapToG2(pp_.g2_base, "base_g2", 7);

    std::vector<DmcfeClientSecret2> dfe_secrets;
    ABG19DmcfeMask2::Setup(
        num_clients_, pp_.g0, dfe_pp_, dfe_secrets);
    clients_.reserve(num_clients_);
    for (int client_id = 0; client_id < num_clients_; ++client_id) {
        clients_.emplace_back(
            new PC_MCFE_Client(
                client_id, rank_, pp_, dfe_pp_,
                dfe_secrets[static_cast<size_t>(client_id)]));
    }
    server_.reset(new PC_MCFE_Server(num_clients_, pp_, dfe_pp_));

    weights_.assign(num_clients_, Fr(1));
    std::vector<DmcfeKeyShare2> shares(num_clients_);
    for (int client_id = 0; client_id < num_clients_; ++client_id) {
        shares[client_id] = clients_[client_id]->KeyGenShare(weights_);
    }
    aggregate_key_ = server_->DKeyComb(shares);

    GT base;
    pairing(base, pp_.g0, pp_.g2_base);
    const auto bsgs_started = std::chrono::steady_clock::now();
    std::fprintf(
        stderr,
        "[SecLoRA] building BSGS: bound=%lld, baby_steps~%.0f\n",
        bsgs_bound_, std::ceil(std::sqrt(static_cast<double>(bsgs_bound_))));
    server_->build_bsgs(
        base, bsgs_bound_,
        [](long long completed, long long total) {
            const int percent = total > 0
                ? static_cast<int>(100 * completed / total)
                : 100;
            std::fprintf(
                stderr, "\r[SecLoRA] BSGS precompute %3d%% (%lld/%lld)",
                percent, completed, total);
            std::fflush(stderr);
        });
    const double bsgs_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - bsgs_started).count();
    std::fprintf(
        stderr, "\n[SecLoRA] BSGS ready in %.3fs, estimated table=%lld bytes\n",
        bsgs_seconds, server_->bsgs_table_bytes_estimate());
}

std::shared_ptr<NativeClientUpdate>
SelectiveTwoServerSession::encrypt_client(
    int client_id, int round_id, const std::vector<FloatLayerInput>& layers) {
    require_open();
    if (client_id < 0 || client_id >= num_clients_) {
        throw std::out_of_range("client_id is outside the configured range");
    }
    if (round_id < 0 || layers.empty()) {
        throw std::invalid_argument("round_id must be nonnegative and layers nonempty");
    }

    std::shared_ptr<NativeClientUpdate> update(new NativeClientUpdate());
    update->client_id = client_id;
    update->round_id = round_id;
    update->layers.reserve(layers.size());

    for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const auto& layer = layers[layer_index];
        std::fprintf(
            stderr,
            "\r[SecLoRA] round %d client %d encrypt layer %zu/%zu",
            round_id, client_id, layer_index + 1, layers.size());
        std::fflush(stderr);
        if (layer.rows <= 0 || layer.cols <= 0 ||
            layer.a.size() != static_cast<std::size_t>(rank_ * layer.cols) ||
            layer.b.size() != static_cast<std::size_t>(layer.rows * rank_)) {
            throw std::invalid_argument("invalid LoRA layer dimensions");
        }
        const auto quantize_pack_started = Clock::now();
        IntMat a = quantize_a(layer);
        IntMat b = quantize_b(layer);
        const int eb = full_sk_ ? layer.rows : static_cast<int>(
            std::ceil(ratio_ * static_cast<double>(layer.rows)));
        const int ea = full_sk_ ? layer.cols : static_cast<int>(
            std::ceil(ratio_ * static_cast<double>(layer.cols)));
        const int candidate_count = 2 * num_clients_ * rank_;
        const auto oracle_key =
            std::make_pair(round_id, layer.layer_id);
        auto oracle_it = plaintext_oracles_.find(oracle_key);
        if (oracle_it == plaintext_oracles_.end()) {
            PlaintextOracleLayer oracle;
            oracle.rows = layer.rows;
            oracle.cols = layer.cols;
            oracle.present.assign(num_clients_, 0);
            oracle.client_a.resize(num_clients_);
            oracle.client_b.resize(num_clients_);
            oracle_it = plaintext_oracles_
                .emplace(oracle_key, std::move(oracle)).first;
        } else if (
            oracle_it->second.rows != layer.rows ||
            oracle_it->second.cols != layer.cols) {
            throw std::invalid_argument(
                "layer id reused with different oracle dimensions");
        }
        PlaintextOracleLayer& oracle = oracle_it->second;
        if (oracle.present[client_id]) {
            throw std::invalid_argument(
                "client submitted the same layer twice in one round");
        }

        NativeLayerUpload payload;
        payload.layer_id = layer.layer_id;
        payload.name = layer.name;
        payload.rows = layer.rows;
        payload.cols = layer.cols;
        payload.encrypted_b_rows = eb;
        payload.encrypted_a_cols = ea;
        payload.candidate_rows = public_candidates(
            full_sk_ ? 0 : eb, layer.rows, candidate_count,
            0x425f43414e44ULL);
        payload.candidate_cols = public_candidates(
            full_sk_ ? 0 : ea, layer.cols, candidate_count,
            0x415f43414e44ULL);

        if (!full_sk_) {
            payload.plain_b.assign(
                layer.rows - eb, std::vector<long long>(rank_));
            for (int row = eb; row < layer.rows; ++row) {
                payload.plain_b[row - eb] = b[row];
            }
            payload.plain_a.assign(
                rank_, std::vector<long long>(layer.cols - ea));
            for (int k = 0; k < rank_; ++k) {
                std::copy(
                    a[k].begin() + ea, a[k].end(),
                    payload.plain_a[k].begin());
            }
        }

        std::vector<int> encrypted_rows;
        std::vector<int> encrypted_cols;
        for (int row = 0; row < eb; ++row) encrypted_rows.push_back(row);
        for (int col = 0; col < ea; ++col) encrypted_cols.push_back(col);
        if (ea > 0) {
            for (int row : payload.candidate_rows) {
                append_unique(encrypted_rows, row);
            }
        }
        if (eb > 0) {
            for (int col : payload.candidate_cols) {
                append_unique(encrypted_cols, col);
            }
        }
        std::sort(encrypted_rows.begin(), encrypted_rows.end());
        std::sort(encrypted_cols.begin(), encrypted_cols.end());

        PC_MCFE_Client& client = *clients_[client_id];
        client.SetLoraMatrices(a, b);
        payload.quantize_pack_wall_sec =
            elapsed_seconds(quantize_pack_started);
        const auto precompute_started = Clock::now();
        client.u_setup();
        try {
            client.precompute_encA_indices_mt(
                layer.layer_id, 0, round_id, layer.cols,
                encrypted_cols, threads_);
            client.precompute_encB_indices_mt(
                layer.layer_id, 0, round_id, layer.rows,
                encrypted_rows, threads_);
            payload.precompute_wall_sec = elapsed_seconds(precompute_started);

            const auto online_started = Clock::now();
            payload.encrypted_a = client.encA_indices_mt(
                layer.layer_id, 0, round_id, layer.cols,
                encrypted_cols, threads_);
            payload.encrypted_b = client.encB_indices_mt(
                layer.layer_id, 0, round_id, layer.rows,
                encrypted_rows, threads_);
            payload.online_crypto_wall_sec = elapsed_seconds(online_started);
        } catch (...) {
            client.ClearEncryptionPrecompute();
            throw;
        }
        const auto precompute_cleanup_started = Clock::now();
        client.ClearEncryptionPrecompute();
        payload.precompute_wall_sec +=
            elapsed_seconds(precompute_cleanup_started);

        oracle.client_a[client_id] = a;
        oracle.client_b[client_id] = b;
        oracle.present[client_id] = 1;

        const auto serialize_started = Clock::now();
        payload.sp_plain_bytes = serialize_plain_slices(
            payload.plain_b, payload.plain_a);
        for (int col : encrypted_cols) {
            payload.sd_cipher_bytes += a_slot_bytes(payload.encrypted_a[col]);
        }
        for (int row : encrypted_rows) {
            payload.sd_cipher_bytes += b_slot_bytes(payload.encrypted_b[row]);
        }
        payload.serialized_size_bytes =
            payload.sp_plain_bytes + payload.sd_cipher_bytes;
        payload.serialize_wall_sec = elapsed_seconds(serialize_started);

        update->sp_plain_bytes += payload.sp_plain_bytes;
        update->sd_cipher_bytes += payload.sd_cipher_bytes;
        update->serialized_size_bytes += payload.serialized_size_bytes;
        update->protected_b_labels += static_cast<std::size_t>(eb);
        update->protected_a_labels += static_cast<std::size_t>(ea);
        if (!full_sk_) {
            if (ea > 0) {
                update->candidate_b_labels += payload.candidate_rows.size();
            }
            if (eb > 0) {
                update->candidate_a_labels += payload.candidate_cols.size();
            }
        }
        update->quantize_pack_wall_sec += payload.quantize_pack_wall_sec;
        update->precompute_wall_sec += payload.precompute_wall_sec;
        update->online_crypto_wall_sec += payload.online_crypto_wall_sec;
        update->serialize_wall_sec += payload.serialize_wall_sec;
        update->layers.push_back(std::move(payload));
    }
    std::fprintf(stderr, "\n");
    return update;
}

std::vector<NativeLayerSkeleton>
SelectiveTwoServerSession::aggregate_round(
    int round_id,
    const std::vector<std::shared_ptr<NativeClientUpdate>>& updates) {
    require_open();
    const auto aggregate_started = Clock::now();
    last_round_metrics_ = NativeRoundMetrics();
    last_round_metrics_.mode = mode_;
    if (static_cast<int>(updates.size()) != num_clients_) {
        throw std::invalid_argument("aggregate_round requires every configured client");
    }

    std::vector<const NativeClientUpdate*> ordered(num_clients_, nullptr);
    for (const auto& update : updates) {
        if (!update || update->round_id != round_id ||
            update->client_id < 0 || update->client_id >= num_clients_) {
            throw std::invalid_argument("invalid client update metadata");
        }
        if (ordered[update->client_id] != nullptr) {
            throw std::invalid_argument("duplicate client update");
        }
        ordered[update->client_id] = update.get();
    }
    const std::size_t layer_count = ordered.front()->layers.size();
    if (layer_count == 0) throw std::invalid_argument("client update has no layers");
    for (const auto* update : ordered) {
        if (update == nullptr || update->layers.size() != layer_count) {
            throw std::invalid_argument("client layer manifests do not match");
        }
    }

    std::vector<NativeLayerSkeleton> output;
    output.reserve(layer_count);
    for (std::size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        std::fprintf(
            stderr,
            "\r[SecLoRA] round %d aggregate layer %zu/%zu",
            round_id, layer_index + 1, layer_count);
        std::fflush(stderr);
        const NativeLayerUpload& first = ordered.front()->layers[layer_index];
        const int rows = first.rows;
        const int cols = first.cols;
        const int eb = first.encrypted_b_rows;
        const int ea = first.encrypted_a_cols;
        for (const auto* update : ordered) {
            const NativeLayerUpload& layer = update->layers[layer_index];
            if (layer.layer_id != first.layer_id ||
                layer.rows != rows || layer.cols != cols ||
                layer.encrypted_b_rows != eb ||
                layer.encrypted_a_cols != ea ||
                layer.candidate_rows != first.candidate_rows ||
                layer.candidate_cols != first.candidate_cols) {
                throw std::invalid_argument("client layer metadata does not match");
            }
        }

        auto sp_cell = [&](int row, int col) {
            if (full_sk_) {
                throw std::logic_error("FULL+SK has no plaintext server cells");
            }
            __int128 sum = 0;
            for (const auto* update : ordered) {
                const NativeLayerUpload& layer = update->layers[layer_index];
                for (int k = 0; k < rank_; ++k) {
                    sum += static_cast<__int128>(layer.plain_b[row - eb][k]) *
                           layer.plain_a[k][col - ea];
                }
            }
            return checked_int64(sum, "plaintext aggregate cell");
        };

        const int aggregate_rank_cap =
            std::min(std::min(rows, cols), num_clients_ * rank_);
        if (!full_sk_ && (eb >= rows || ea >= cols)) {
            throw std::runtime_error(
                "SEL-2S needs a nonempty plaintext row and column region");
        }

        const auto oracle_key =
            std::make_pair(round_id, first.layer_id);
        const auto oracle_it = plaintext_oracles_.find(oracle_key);
        if (oracle_it == plaintext_oracles_.end()) {
            throw std::runtime_error(
                "plaintext evaluation oracle is unavailable for this layer");
        }
        const PlaintextOracleLayer& oracle = oracle_it->second;
        if (std::find(
                oracle.present.begin(), oracle.present.end(),
                static_cast<unsigned char>(0)) != oracle.present.end()) {
            throw std::runtime_error(
                "plaintext evaluation oracle is missing a client update");
        }

        std::vector<const std::vector<A_Ciphertext_Slot>*> a_refs;
        std::vector<const std::vector<B_SecretKey_Slot>*> b_refs;
        a_refs.reserve(num_clients_);
        b_refs.reserve(num_clients_);
        for (const auto* update : ordered) {
            const NativeLayerUpload& layer = update->layers[layer_index];
            a_refs.push_back(&layer.encrypted_a);
            b_refs.push_back(&layer.encrypted_b);
        }
        std::vector<int> encrypted_cols;
        for (int col = 0; col < ea; ++col) {
            encrypted_cols.push_back(col);
        }
        if (!full_sk_ && eb > 0) {
            for (int col : first.candidate_cols) {
                append_unique(encrypted_cols, col);
            }
        }
        std::sort(encrypted_cols.begin(), encrypted_cols.end());
        double dfe_worker_seconds = 0.0;
        const auto dfe_started = Clock::now();
        server_->PrepareDfeMaskCacheRefs(
            a_refs, weights_, aggregate_key_, encrypted_cols,
            threads_, dfe_worker_seconds);
        last_round_metrics_.sd_dfe_mask_wall_sec +=
            elapsed_seconds(dfe_started);

        auto decrypt_batch = [&](const std::vector<std::pair<int, int>>& cells) {
            const auto batch_started = Clock::now();
            std::vector<GT> groups(cells.size());
            const auto fe_started = Clock::now();
            parallel_for(
                static_cast<int>(cells.size()), threads_, [&](int index) {
                    groups[index] = server_->eval_one_cell_group_refs(
                        a_refs, b_refs, weights_, first.layer_id, 0,
                        round_id, cells[index].first, cells[index].second);
                });
            const double fe_seconds = elapsed_seconds(fe_started);
            last_round_metrics_.sd_fe_eval_wall_sec += fe_seconds;

            std::vector<long long> values(cells.size(), 0);
            std::vector<unsigned char> found(cells.size(), 0);
            const auto bsgs_started = Clock::now();
            parallel_for(
                static_cast<int>(cells.size()), threads_, [&](int index) {
                    bool cell_found = false;
                    values[index] = server_->bsgs_search(
                        groups[index], cell_found);
                    found[index] = cell_found ? 1 : 0;
                });
            const double bsgs_seconds = elapsed_seconds(bsgs_started);
            last_round_metrics_.sd_bsgs_search_wall_sec += bsgs_seconds;
            for (unsigned char value : found) {
                if (!value) {
                    throw std::runtime_error(
                        "BSGS failed for an encrypted aggregate cell; "
                        "check sfp, xmax, and the public bound");
                }
            }
            last_round_metrics_.sd_control_wall_sec += std::max(
                0.0,
                elapsed_seconds(batch_started) - fe_seconds - bsgs_seconds);
            return values;
        };

        std::pair<std::vector<int>, std::vector<int>> candidate_pivots;
        std::map<std::pair<int, int>, long long> encrypted_cell_cache;
        if (full_sk_) {
            const auto candidate_prepare_started = Clock::now();
            std::vector<std::pair<int, int>> candidate_cells;
            candidate_cells.reserve(
                first.candidate_rows.size() * first.candidate_cols.size());
            for (int row : first.candidate_rows) {
                for (int col : first.candidate_cols) {
                    candidate_cells.emplace_back(row, col);
                }
            }
            last_round_metrics_.cur_skeleton_wall_sec +=
                elapsed_seconds(candidate_prepare_started);
            const std::vector<long long> candidate_values =
                decrypt_batch(candidate_cells);
            const auto candidate_control_started = Clock::now();
            for (std::size_t index = 0; index < candidate_cells.size(); ++index) {
                encrypted_cell_cache[candidate_cells[index]] =
                    candidate_values[index];
            }
            last_round_metrics_.pivot_candidate_cells +=
                candidate_cells.size();
            candidate_pivots = select_square_pivots(
                first.candidate_rows, first.candidate_cols,
                aggregate_rank_cap,
                [&](int row, int col) {
                    return encrypted_cell_cache.at(std::make_pair(row, col));
                });
            last_round_metrics_.cur_skeleton_wall_sec +=
                elapsed_seconds(candidate_control_started);
        } else {
            std::map<std::pair<int, int>, long long> candidate_values;
            const auto sp_candidate_started = Clock::now();
            for (int row : first.candidate_rows) {
                for (int col : first.candidate_cols) {
                    candidate_values.emplace(
                        std::make_pair(row, col), sp_cell(row, col));
                }
            }
            last_round_metrics_.sp_wall_sec +=
                elapsed_seconds(sp_candidate_started);
            const auto pivot_select_started = Clock::now();
            candidate_pivots = select_square_pivots(
                first.candidate_rows, first.candidate_cols,
                aggregate_rank_cap,
                [&](int row, int col) {
                    return candidate_values.at(std::make_pair(row, col));
                });
            last_round_metrics_.cur_skeleton_wall_sec +=
                elapsed_seconds(pivot_select_started);
        }
        if (candidate_pivots.first.size() < static_cast<std::size_t>(rank_)) {
            throw std::runtime_error(
                "public pivot candidate pool cannot form the initial "
                "rank-R nonsingular skeleton");
        }

        const std::vector<int>& pivot_rows = candidate_pivots.first;
        const std::vector<int>& pivot_cols = candidate_pivots.second;
        const int available_rank = static_cast<int>(pivot_rows.size());

        const int factor_width = num_clients_ * rank_;
        auto oracle_b = [&](int row, int component) -> Real {
            const int client = component / rank_;
            const int k = component % rank_;
            return static_cast<Real>(oracle.client_b[client][row][k]);
        };
        auto oracle_a = [&](int component, int col) -> Real {
            const int client = component / rank_;
            const int k = component % rank_;
            return static_cast<Real>(oracle.client_a[client][k][col]);
        };

        const auto baseline_setup_started = Clock::now();
        RealMat btb(
            factor_width, std::vector<Real>(factor_width, 0));
        RealMat aat(
            factor_width, std::vector<Real>(factor_width, 0));
        for (int row = 0; row < rows; ++row) {
            for (int left = 0; left < factor_width; ++left) {
                const Real value = oracle_b(row, left);
                for (int right = 0; right < factor_width; ++right) {
                    btb[left][right] +=
                        value * oracle_b(row, right);
                }
            }
        }
        for (int col = 0; col < cols; ++col) {
            for (int left = 0; left < factor_width; ++left) {
                const Real value = oracle_a(left, col);
                for (int right = 0; right < factor_width; ++right) {
                    aat[left][right] +=
                        value * oracle_a(right, col);
                }
            }
        }
        const Real baseline_norm_sq = gram_product_sum(btb, aat);
        if (!(baseline_norm_sq > 0)) {
            throw std::runtime_error(
                "plaintext fixed-point baseline has zero Frobenius norm");
        }
        last_round_metrics_.experiment_verify_wall_sec +=
            elapsed_seconds(baseline_setup_started);

        const auto skeleton_cache_started = Clock::now();
        IntMat cached_c(
            rows, std::vector<long long>(available_rank, 0));
        IntMat cached_m(
            available_rank, std::vector<long long>(available_rank, 0));
        IntMat cached_s(
            available_rank, std::vector<long long>(cols, 0));
        last_round_metrics_.cur_skeleton_wall_sec +=
            elapsed_seconds(skeleton_cache_started);

        struct DecryptCell {
            int row;
            int col;
            int skeleton_row;
            int skeleton_col;
            bool writes_c;
        };

        NativeLayerSkeleton skeleton;
        bool verified = false;
        int previous_rank = 0;
        int baseline_checks = 0;
        std::size_t decrypted_cells = 0;
        std::fprintf(stderr, "\n");
        for (int current_rank = rank_;
             current_rank <= available_rank;
             ++current_rank) {
            const int added_begin =
                previous_rank == 0 ? 0 : current_rank - 1;
            if (!full_sk_) {
                const auto sp_skeleton_started = Clock::now();
                for (int t = added_begin; t < current_rank; ++t) {
                    for (int row = eb; row < rows; ++row) {
                        cached_c[row][t] = sp_cell(row, pivot_cols[t]);
                    }
                    for (int col = ea; col < cols; ++col) {
                        cached_s[t][col] = sp_cell(pivot_rows[t], col);
                    }
                    for (int j = 0; j <= t; ++j) {
                        cached_m[t][j] =
                            sp_cell(pivot_rows[t], pivot_cols[j]);
                        cached_m[j][t] =
                            sp_cell(pivot_rows[j], pivot_cols[t]);
                    }
                }
                last_round_metrics_.sp_wall_sec +=
                    elapsed_seconds(sp_skeleton_started);
            }

            const auto sd_prepare_started = Clock::now();
            std::vector<DecryptCell> work;
            const int encrypted_row_limit = full_sk_ ? rows : eb;
            const int encrypted_col_limit = full_sk_ ? cols : ea;
            work.reserve(
                static_cast<std::size_t>(
                    encrypted_row_limit + encrypted_col_limit) *
                (current_rank - added_begin));
            for (int t = added_begin; t < current_rank; ++t) {
                for (int row = 0; row < encrypted_row_limit; ++row) {
                    work.push_back({row, pivot_cols[t], row, t, true});
                }
                for (int col = 0; col < encrypted_col_limit; ++col) {
                    work.push_back({pivot_rows[t], col, t, col, false});
                }
            }

            std::vector<std::pair<int, int>> missing_cells;
            for (const DecryptCell& cell : work) {
                const auto coordinate = std::make_pair(cell.row, cell.col);
                if (encrypted_cell_cache.find(coordinate) ==
                    encrypted_cell_cache.end()) {
                    encrypted_cell_cache.emplace(
                        coordinate, std::numeric_limits<long long>::min());
                    missing_cells.push_back(coordinate);
                }
            }
            last_round_metrics_.cur_skeleton_wall_sec +=
                elapsed_seconds(sd_prepare_started);
            const std::vector<long long> missing_values =
                decrypt_batch(missing_cells);
            const auto sd_assign_started = Clock::now();
            for (std::size_t index = 0; index < missing_cells.size(); ++index) {
                encrypted_cell_cache[missing_cells[index]] =
                    missing_values[index];
            }
            for (const DecryptCell& cell : work) {
                const long long value = encrypted_cell_cache.at(
                    std::make_pair(cell.row, cell.col));
                if (cell.writes_c) {
                    cached_c[cell.skeleton_row][cell.skeleton_col] =
                        value;
                } else {
                    cached_s[cell.skeleton_row][cell.skeleton_col] =
                        value;
                }
            }
            decrypted_cells += missing_cells.size();

            if (full_sk_) {
                for (int t = added_begin; t < current_rank; ++t) {
                    for (int j = 0; j <= t; ++j) {
                        cached_m[t][j] = cached_c[pivot_rows[t]][j];
                        cached_m[j][t] = cached_c[pivot_rows[j]][t];
                    }
                }
            }
            last_round_metrics_.cur_skeleton_wall_sec +=
                elapsed_seconds(sd_assign_started);

            const auto cur_started = Clock::now();
            const RealMat solved = solve_nonsingular_real(
                cached_m, cached_s, current_rank, cols);
            const double cur_solve_seconds = elapsed_seconds(cur_started);
            last_round_metrics_.cur_reconstruct_wall_sec += cur_solve_seconds;
            last_round_metrics_.cur_skeleton_wall_sec += cur_solve_seconds;
            const auto verify_started = Clock::now();
            RealMat ctc(
                current_rank, std::vector<Real>(current_rank, 0));
            RealMat xxt(
                current_rank, std::vector<Real>(current_rank, 0));
            RealMat btc(
                factor_width, std::vector<Real>(current_rank, 0));
            RealMat axt(
                factor_width, std::vector<Real>(current_rank, 0));
            for (int row = 0; row < rows; ++row) {
                for (int left = 0; left < current_rank; ++left) {
                    const Real c_value =
                        static_cast<Real>(cached_c[row][left]);
                    for (int right = 0; right < current_rank; ++right) {
                        ctc[left][right] += c_value *
                            static_cast<Real>(cached_c[row][right]);
                    }
                    for (int component = 0;
                         component < factor_width;
                         ++component) {
                        btc[component][left] +=
                            oracle_b(row, component) * c_value;
                    }
                }
            }
            for (int col = 0; col < cols; ++col) {
                for (int left = 0; left < current_rank; ++left) {
                    const Real x_value = solved[left][col];
                    for (int right = 0; right < current_rank; ++right) {
                        xxt[left][right] +=
                            x_value * solved[right][col];
                    }
                    for (int component = 0;
                         component < factor_width;
                         ++component) {
                        axt[component][left] +=
                            oracle_a(component, col) * x_value;
                    }
                }
            }

            const Real reconstructed_norm_sq =
                gram_product_sum(ctc, xxt);
            Real cross_inner_product = 0;
            for (int component = 0;
                 component < factor_width;
                 ++component) {
                for (int t = 0; t < current_rank; ++t) {
                    cross_inner_product +=
                        btc[component][t] * axt[component][t];
                }
            }
            Real error_norm_sq =
                baseline_norm_sq + reconstructed_norm_sq -
                2 * cross_inner_product;
            const Real cancellation_scale = std::max(
                static_cast<Real>(1),
                std::max(
                    baseline_norm_sq,
                    std::max(
                        reconstructed_norm_sq,
                        std::fabs(2 * cross_inner_product))));
            if (std::fabs(error_norm_sq) <=
                1e-15L * cancellation_scale) {
                error_norm_sq = 0;
            }
            error_norm_sq = std::max(static_cast<Real>(0), error_norm_sq);
            const Real relative_error =
                std::sqrt(error_norm_sq / baseline_norm_sq);
            ++baseline_checks;
            verified = relative_error <= kBaselineRelativeTolerance;
            last_round_metrics_.experiment_verify_wall_sec +=
                elapsed_seconds(verify_started);
            std::fprintf(
                stderr,
                "[SecLoRA] layer %d rank %d baseline error %.3Le %s\n",
                first.layer_id, current_rank,
                relative_error,
                verified ? "PASS" : "FAIL");
            if (verified) {
                const auto skeleton_output_started = Clock::now();
                skeleton.layer_id = first.layer_id;
                skeleton.rows = rows;
                skeleton.cols = cols;
                skeleton.selected_rank = current_rank;
                skeleton.pivot_rows.assign(
                    pivot_rows.begin(), pivot_rows.begin() + current_rank);
                skeleton.pivot_cols.assign(
                    pivot_cols.begin(), pivot_cols.begin() + current_rank);
                skeleton.baseline_checks = baseline_checks;
                skeleton.baseline_relative_error =
                    static_cast<double>(relative_error);
                skeleton.decrypted_cells = full_sk_
                    ? encrypted_cell_cache.size()
                    : decrypted_cells;
                skeleton.pivot_candidate_cells = full_sk_
                    ? first.candidate_rows.size() * first.candidate_cols.size()
                    : 0;
                skeleton.c.assign(
                    rows, std::vector<long long>(current_rank));
                for (int row = 0; row < rows; ++row) {
                    std::copy_n(
                        cached_c[row].begin(), current_rank,
                        skeleton.c[row].begin());
                }
                skeleton.m.assign(
                    current_rank, std::vector<long long>(current_rank));
                for (int t = 0; t < current_rank; ++t) {
                    std::copy_n(
                        cached_m[t].begin(), current_rank,
                        skeleton.m[t].begin());
                }
                skeleton.s.assign(
                    cached_s.begin(), cached_s.begin() + current_rank);
                serialize_skeleton_parts(skeleton);
                const double skeleton_output_seconds =
                    elapsed_seconds(skeleton_output_started);
                last_round_metrics_.cur_reconstruct_wall_sec +=
                    skeleton_output_seconds;
                last_round_metrics_.cur_skeleton_wall_sec +=
                    skeleton_output_seconds;
                last_round_metrics_.protected_skeleton_cells += full_sk_
                    ? static_cast<std::size_t>(current_rank) *
                          (rows + cols - current_rank)
                    : static_cast<std::size_t>(current_rank) * (eb + ea);
                last_round_metrics_.download_c_bytes_per_client +=
                    skeleton.download_c_bytes;
                last_round_metrics_.download_m_bytes_per_client +=
                    skeleton.download_m_bytes;
                last_round_metrics_.download_s_bytes_per_client +=
                    skeleton.download_s_bytes;
                break;
            }
            previous_rank = current_rank;
        }
        if (!verified) {
            throw std::runtime_error(
                "no plaintext-baseline-verified skeleton was found from R through "
                "the public candidate pool; enlarge or change the public "
                "pivot candidates");
        }
        plaintext_oracles_.erase(oracle_key);
        output.push_back(std::move(skeleton));
    }
    std::fprintf(stderr, "\n");
    last_round_metrics_.download_bytes_per_client =
        last_round_metrics_.download_c_bytes_per_client +
        last_round_metrics_.download_m_bytes_per_client +
        last_round_metrics_.download_s_bytes_per_client;
    last_round_metrics_.sd_wall_sec =
        last_round_metrics_.sd_dfe_mask_wall_sec +
        last_round_metrics_.sd_fe_eval_wall_sec +
        last_round_metrics_.sd_bsgs_search_wall_sec +
        last_round_metrics_.sd_control_wall_sec;
    last_round_metrics_.observed_serial_server_wall_sec =
        elapsed_seconds(aggregate_started);
    const double attributed_wall_sec =
        last_round_metrics_.sp_wall_sec +
        last_round_metrics_.sd_wall_sec +
        last_round_metrics_.cur_skeleton_wall_sec +
        last_round_metrics_.experiment_verify_wall_sec;
    last_round_metrics_.server_common_control_wall_sec = std::max(
        0.0,
        last_round_metrics_.observed_serial_server_wall_sec -
            attributed_wall_sec);
    return output;
}

void SelectiveTwoServerSession::close() {
    closed_ = true;
    plaintext_oracles_.clear();
    clients_.clear();
    server_.reset();
}

void SelectiveTwoServerSession::require_open() const {
    if (closed_) throw std::runtime_error("SecLoRA native session is closed");
}

IntMat SelectiveTwoServerSession::quantize_a(
    const FloatLayerInput& layer) const {
    IntMat result(rank_, std::vector<long long>(layer.cols));
    for (int k = 0; k < rank_; ++k) {
        for (int col = 0; col < layer.cols; ++col) {
            const double value = layer.a[k * layer.cols + col];
            if (!std::isfinite(value)) {
                throw std::invalid_argument("LoRA A contains a non-finite value");
            }
            const double clipped = std::max(-xmax_, std::min(xmax_, value));
            result[k][col] = std::llround(clipped * scale_);
        }
    }
    return result;
}

IntMat SelectiveTwoServerSession::quantize_b(
    const FloatLayerInput& layer) const {
    IntMat result(layer.rows, std::vector<long long>(rank_));
    for (int row = 0; row < layer.rows; ++row) {
        for (int k = 0; k < rank_; ++k) {
            const double value = layer.b[row * rank_ + k];
            if (!std::isfinite(value)) {
                throw std::invalid_argument("LoRA B contains a non-finite value");
            }
            const double clipped = std::max(-xmax_, std::min(xmax_, value));
            result[row][k] = std::llround(clipped * scale_);
        }
    }
    return result;
}

#include "session.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {

FloatLayerInput make_layer(int client_id) {
    FloatLayerInput layer;
    layer.layer_id = 0;
    layer.name = "selftest";
    layer.rows = 12;
    layer.cols = 12;
    layer.a.resize(2 * layer.cols);
    layer.b.resize(layer.rows * 2);
    if (client_id == 1) return layer;

    for (int col = 0; col < layer.cols; ++col) {
        layer.a[col] = static_cast<float>(col + 1) / 16.0f;
        layer.a[layer.cols + col] =
            static_cast<float>((col % 5) - 2) / 16.0f;
    }
    for (int row = 0; row < layer.rows; ++row) {
        layer.b[row * 2] =
            static_cast<float>((row % 4) + 1) / 16.0f;
        layer.b[row * 2 + 1] =
            static_cast<float>((row % 7) - 3) / 16.0f;
    }
    return layer;
}

}  // namespace

int main() {
    SelectiveTwoServerSession session(2, 2, 0.25, 4, 8.0, 2);
    std::vector<std::shared_ptr<NativeClientUpdate>> updates;
    for (int client = 0; client < 2; ++client) {
        updates.push_back(
            session.encrypt_client(client, 1, {make_layer(client)}));
    }
    const auto result = session.aggregate_round(1, updates);
    const bool passed =
        result.size() == 1 &&
        result[0].selected_rank == 2 &&
        result[0].baseline_relative_error <= 1e-8;
    std::printf(
        "SEL-2S PC-DMCFE session: %s (rank=%d, error=%.3e)\n",
        passed ? "PASS" : "FAIL",
        result.empty() ? -1 : result[0].selected_rank,
        result.empty() ? -1.0 : result[0].baseline_relative_error);
    return passed ? 0 : 1;
}

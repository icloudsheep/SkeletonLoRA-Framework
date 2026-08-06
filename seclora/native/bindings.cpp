#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "session.h"

namespace py = pybind11;

namespace {

FloatLayerInput parse_layer(const py::dict& value) {
    FloatLayerInput layer;
    layer.layer_id = py::cast<int>(value["layer_id"]);
    if (value.contains(py::str("name"))) {
        layer.name = py::cast<std::string>(value["name"]);
    }

    using FloatArray =
        py::array_t<float, py::array::c_style | py::array::forcecast>;
    FloatArray a = FloatArray::ensure(value["a"]);
    FloatArray b = FloatArray::ensure(value["b"]);
    if (!a || !b || a.ndim() != 2 || b.ndim() != 2 ||
        a.shape(0) != b.shape(1)) {
        throw std::invalid_argument(
            "native layer expects A=(rank, cols), B=(rows, rank)");
    }

    layer.rows = static_cast<int>(b.shape(0));
    layer.cols = static_cast<int>(a.shape(1));
    const float* a_data = a.data();
    const float* b_data = b.data();
    layer.a.assign(a_data, a_data + a.size());
    layer.b.assign(b_data, b_data + b.size());
    return layer;
}

py::array_t<long long> matrix_array(
    const std::vector<std::vector<long long>>& matrix,
    std::size_t empty_cols = 0) {
    const py::ssize_t rows = static_cast<py::ssize_t>(matrix.size());
    const py::ssize_t cols = rows > 0
        ? static_cast<py::ssize_t>(matrix.front().size())
        : static_cast<py::ssize_t>(empty_cols);
    py::array_t<long long> result({rows, cols});
    auto view = result.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < rows; ++row) {
        if (static_cast<py::ssize_t>(matrix[row].size()) != cols) {
            throw std::runtime_error("native matrix is ragged");
        }
        for (py::ssize_t col = 0; col < cols; ++col) {
            view(row, col) = matrix[row][col];
        }
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_seclora_native, module) {
    module.doc() = "Persistent PC-DMCFE SecLoRA backend for SkeletonLoRA";

    py::class_<NativeClientUpdate, std::shared_ptr<NativeClientUpdate>>(
        module, "NativeClientUpdate")
        .def_property_readonly(
            "client_id", [](const NativeClientUpdate& value) {
                return value.client_id;
            })
        .def_property_readonly(
            "round_id", [](const NativeClientUpdate& value) {
                return value.round_id;
            })
        .def_property_readonly(
            "serialized_size_bytes", [](const NativeClientUpdate& value) {
                return value.serialized_size_bytes;
            })
        .def_readonly("sp_plain_bytes", &NativeClientUpdate::sp_plain_bytes)
        .def_readonly("sd_cipher_bytes", &NativeClientUpdate::sd_cipher_bytes)
        .def_readonly("protected_b_labels", &NativeClientUpdate::protected_b_labels)
        .def_readonly("protected_a_labels", &NativeClientUpdate::protected_a_labels)
        .def_readonly("candidate_b_labels", &NativeClientUpdate::candidate_b_labels)
        .def_readonly("candidate_a_labels", &NativeClientUpdate::candidate_a_labels)
        .def_readonly(
            "quantize_pack_wall_sec",
            &NativeClientUpdate::quantize_pack_wall_sec)
        .def_readonly(
            "binding_input_copy_wall_sec",
            &NativeClientUpdate::binding_input_copy_wall_sec)
        .def_readonly("precompute_wall_sec", &NativeClientUpdate::precompute_wall_sec)
        .def_readonly(
            "online_crypto_wall_sec",
            &NativeClientUpdate::online_crypto_wall_sec)
        .def_readonly("serialize_wall_sec", &NativeClientUpdate::serialize_wall_sec);

    py::class_<NativeLayerSkeleton>(module, "NativeLayerSkeleton")
        .def_property_readonly(
            "layer_id", [](const NativeLayerSkeleton& value) {
                return value.layer_id;
            })
        .def_property_readonly(
            "c", [](const NativeLayerSkeleton& value) {
                const std::size_t rank =
                    value.m.empty() ? 0 : value.m.front().size();
                return matrix_array(value.c, rank);
            })
        .def_property_readonly(
            "m", [](const NativeLayerSkeleton& value) {
                return matrix_array(value.m, value.m.size());
            })
        .def_property_readonly(
            "s", [](const NativeLayerSkeleton& value) {
                return matrix_array(value.s, value.cols);
            })
        .def_readonly(
            "selected_rank", &NativeLayerSkeleton::selected_rank)
        .def_readonly("pivot_rows", &NativeLayerSkeleton::pivot_rows)
        .def_readonly("pivot_cols", &NativeLayerSkeleton::pivot_cols)
        .def_readonly(
            "baseline_checks", &NativeLayerSkeleton::baseline_checks)
        .def_readonly(
            "baseline_relative_error",
            &NativeLayerSkeleton::baseline_relative_error)
        .def_readonly(
            "decrypted_cells", &NativeLayerSkeleton::decrypted_cells)
        .def_readonly(
            "pivot_candidate_cells",
            &NativeLayerSkeleton::pivot_candidate_cells)
        .def_readonly(
            "download_c_bytes", &NativeLayerSkeleton::download_c_bytes)
        .def_readonly(
            "download_m_bytes", &NativeLayerSkeleton::download_m_bytes)
        .def_readonly(
            "download_s_bytes", &NativeLayerSkeleton::download_s_bytes);

    py::class_<NativeRoundMetrics>(module, "NativeRoundMetrics")
        .def_readonly("mode", &NativeRoundMetrics::mode)
        .def_readonly("sp_wall_sec", &NativeRoundMetrics::sp_wall_sec)
        .def_readonly("sd_wall_sec", &NativeRoundMetrics::sd_wall_sec)
        .def_readonly(
            "sd_dfe_mask_wall_sec",
            &NativeRoundMetrics::sd_dfe_mask_wall_sec)
        .def_readonly(
            "sd_fe_eval_wall_sec",
            &NativeRoundMetrics::sd_fe_eval_wall_sec)
        .def_readonly(
            "sd_bsgs_search_wall_sec",
            &NativeRoundMetrics::sd_bsgs_search_wall_sec)
        .def_readonly(
            "sd_control_wall_sec",
            &NativeRoundMetrics::sd_control_wall_sec)
        .def_readonly(
            "cur_skeleton_wall_sec",
            &NativeRoundMetrics::cur_skeleton_wall_sec)
        .def_readonly(
            "cur_reconstruct_wall_sec",
            &NativeRoundMetrics::cur_reconstruct_wall_sec)
        .def_readonly(
            "experiment_verify_wall_sec",
            &NativeRoundMetrics::experiment_verify_wall_sec)
        .def_readonly(
            "server_common_control_wall_sec",
            &NativeRoundMetrics::server_common_control_wall_sec)
        .def_readonly(
            "observed_serial_server_wall_sec",
            &NativeRoundMetrics::observed_serial_server_wall_sec)
        .def_readonly(
            "protected_skeleton_cells",
            &NativeRoundMetrics::protected_skeleton_cells)
        .def_readonly(
            "pivot_candidate_cells",
            &NativeRoundMetrics::pivot_candidate_cells)
        .def_readonly(
            "download_c_bytes_per_client",
            &NativeRoundMetrics::download_c_bytes_per_client)
        .def_readonly(
            "download_m_bytes_per_client",
            &NativeRoundMetrics::download_m_bytes_per_client)
        .def_readonly(
            "download_s_bytes_per_client",
            &NativeRoundMetrics::download_s_bytes_per_client)
        .def_readonly(
            "download_bytes_per_client",
            &NativeRoundMetrics::download_bytes_per_client);

    py::class_<SelectiveTwoServerSession>(
        module, "SelectiveTwoServerSession")
        .def(
            py::init([](int num_clients, int rank, double ratio,
                        int sfp, double xmax, int threads,
                        const std::string& mode) {
                py::gil_scoped_release release;
                return std::unique_ptr<SelectiveTwoServerSession>(
                    new SelectiveTwoServerSession(
                        num_clients, rank, ratio, sfp, xmax, threads, mode));
            }),
            py::arg("num_clients"),
            py::arg("rank"),
            py::arg("ratio"),
            py::arg("sfp"),
            py::arg("xmax"),
            py::arg("threads"),
            py::arg("mode") = "sel-2s")
        .def(
            "encrypt_client",
            [](SelectiveTwoServerSession& session,
               int client_id, int round_id, const py::list& layers) {
                const auto parse_started = std::chrono::steady_clock::now();
                std::vector<FloatLayerInput> parsed;
                parsed.reserve(layers.size());
                for (const py::handle& item : layers) {
                    parsed.push_back(parse_layer(py::cast<py::dict>(item)));
                }
                const double parse_wall_sec = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - parse_started).count();
                std::shared_ptr<NativeClientUpdate> result;
                {
                    py::gil_scoped_release release;
                    result = session.encrypt_client(client_id, round_id, parsed);
                }
                result->binding_input_copy_wall_sec = parse_wall_sec;
                return result;
            },
            py::arg("client_id"),
            py::arg("round_id"),
            py::arg("layers"))
        .def(
            "aggregate_round",
            [](SelectiveTwoServerSession& session,
               int round_id,
               const std::vector<std::shared_ptr<NativeClientUpdate>>& updates) {
                py::gil_scoped_release release;
                return session.aggregate_round(round_id, updates);
            },
            py::arg("round_id"),
            py::arg("updates"))
        .def_property_readonly(
            "last_round_metrics",
            &SelectiveTwoServerSession::last_round_metrics,
            py::return_value_policy::reference_internal)
        .def("close", &SelectiveTwoServerSession::close);

    module.def(
        "create_session",
        [](int num_clients, int rank, double ratio,
           int sfp, double xmax, int threads, const std::string& mode) {
            py::gil_scoped_release release;
            return std::unique_ptr<SelectiveTwoServerSession>(
                new SelectiveTwoServerSession(
                    num_clients, rank, ratio, sfp, xmax, threads, mode));
        },
        py::arg("num_clients"),
        py::arg("rank"),
        py::arg("ratio"),
        py::arg("sfp"),
        py::arg("xmax"),
        py::arg("threads"),
        py::arg("mode") = "sel-2s");
}

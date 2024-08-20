// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"
// clang-format on

/**
 * @brief Main with support Unicode paths, wide strings
 */
int tmain(int argc, tchar* argv[]) {
    try {
        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (argc != 3) {
            slog::info << "Usage : " << TSTRING2STRING(argv[0]) << " <path_to_model> <device_name>" << slog::endl;
            return EXIT_FAILURE;
        }

        const std::string args = TSTRING2STRING(argv[0]);
        const std::string model_path = TSTRING2STRING(argv[1]);
        const std::string device_name = TSTRING2STRING(argv[2]);

        // -------- Step 1. Initialize OpenVINO Runtime Core --------
        ov::Core core;

        // -------- Step 2. Read a model --------
        slog::info << "Loading model files: " << model_path << slog::endl;

        // -------- Step 5. Loading a model to the device --------
        std::ifstream modelStream(model_path, std::ios_base::binary | std::ios_base::in);
        if (!modelStream.is_open()) {
            throw std::runtime_error("Cannot open model file " + model_path);
        }
        ov::CompiledModel compiled_model = core.import_model(modelStream, device_name, {{"LOG_LEVEL", "LOG_DEBUG"}});

        // -------- Step 6. Create an infer request --------
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        for (auto input : compiled_model.inputs()) {
            infer_request.get_tensor(input);
        }
        for (auto output : compiled_model.outputs()) {
            infer_request.get_tensor(output);
        }

        // -----------------------------------------------------------------------------------------------------

        // -------- Step 8. Do inference synchronously --------
        infer_request.infer();
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

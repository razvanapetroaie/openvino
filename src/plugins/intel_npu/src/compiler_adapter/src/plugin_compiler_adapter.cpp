// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_compiler_adapter.hpp"

#include <memory>
#include <string>

#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/compiler.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_result.hpp"
#include "openvino/core/model.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "plugin_graph.hpp"

namespace {
std::shared_ptr<void> loadLibrary(const std::string& libpath) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    return ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#else
    return ov::util::load_shared_object(libpath.c_str());
#endif
}

std::shared_ptr<intel_npu::ICompiler> getCompiler(std::shared_ptr<void> so) {
    static constexpr auto CreateFuncName = "CreateNPUCompiler";
    auto symbol = ov::util::get_symbol(so, CreateFuncName);

    using CreateFuncT = void (*)(std::shared_ptr<intel_npu::ICompiler>&);
    const auto createFunc = reinterpret_cast<CreateFuncT>(symbol);

    std::shared_ptr<intel_npu::ICompiler> compilerPtr;
    createFunc(compilerPtr);
    return compilerPtr;
}

ov::SoPtr<intel_npu::ICompiler> loadCompiler(const std::string& libpath) {
    auto compilerSO = loadLibrary(libpath);
    auto compiler = getCompiler(compilerSO);

    return ov::SoPtr<intel_npu::ICompiler>(compiler, compilerSO);
}
}  // namespace

namespace intel_npu {

PluginCompilerAdapter::PluginCompilerAdapter(const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct)
    : _zeroInitStruct(zeroInitStruct),
      _logger("PluginCompilerAdapter", Logger::global().level()) {
    _logger.debug("initialize PluginCompilerAdapter start");

    _logger.info("MLIR compiler will be used.");
    std::string baseName = "npu_mlir_compiler";
    auto libPath = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), baseName + OV_BUILD_POSTFIX);
    _compiler = loadCompiler(libPath);

    if (_zeroInitStruct == nullptr) {
        return;
    }

    uint32_t graphExtVersion = _zeroInitStruct->getGraphDdiTable().version();

    _logger.info("PluginCompilerAdapter creating adapter using graphExtVersion");

    _zeGraphExt = std::make_shared<ZeGraphExtWrappers>(_zeroInitStruct);

    _logger.info("initialize PluginCompilerAdapter complete, using graphExtVersion: %d.%d",
                 ZE_MAJOR_VERSION(graphExtVersion),
                 ZE_MINOR_VERSION(graphExtVersion));
}

std::shared_ptr<IGraph> PluginCompilerAdapter::compile(const std::shared_ptr<const ov::Model>& model,
                                                       const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "compile");

    _logger.debug("compile start");
    auto networkDesc = _compiler->compile(model, config);
    _logger.debug("compile end");

    ze_graph_handle_t graphHandle = nullptr;

    if (_zeGraphExt) {
        // Depending on the config, we may get an error when trying to get the graph handle from the compiled network
        try {
            graphHandle = _zeGraphExt->getGraphHandle(networkDesc.compiledNetwork);
        } catch (...) {
            _logger.info("Failed to obtain the level zero graph handle. Inference requests for this model are not "
                         "allowed. Only exports are available");
        }
    }

    return std::make_shared<PluginGraph>(_zeGraphExt,
                                         _compiler,
                                         _zeroInitStruct,
                                         graphHandle,
                                         std::move(networkDesc.metadata),
                                         std::move(networkDesc.compiledNetwork),
                                         config);
}

std::vector<std::shared_ptr<IGraph>> PluginCompilerAdapter::compileWS(const std::shared_ptr<ov::Model>& model,
                                                                      const Config& config) const {
    OV_ITT_TASK_CHAIN(COMPILE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "compileWS");

    auto compileNetBegin = std::chrono::steady_clock::now();

    std::shared_ptr<NetworkDescription> initNetworkDescription;
    std::shared_ptr<NetworkDescription> mainNetworkDescription;

    _logger.debug("compile start");

    const auto starts_with = [](const std::string& str, const std::string& prefix) {
        return str.substr(0, prefix.size()) == prefix;
    };
    const auto isInit = [&](std::string name) {
        return starts_with(name, "init");
    };

    const auto isMain = [&](std::string name) {
        return starts_with(name, "main");
    };

    std::vector<std::shared_ptr<IGraph>> results;

    switch (config.get<SEPARATE_WEIGHTS_VERSION>()) {
    case 1: {
        const std::vector<std::shared_ptr<NetworkDescription>> initMainNetworkDescriptions =
            _compiler->compileWS_v1(model, config);

        OPENVINO_ASSERT(isMain(initMainNetworkDescriptions.back()->metadata.name),
                        "Unexpected network name for main:",
                        initMainNetworkDescriptions.back()->metadata.name);

        for (auto& networkDesc : initMainNetworkDescriptions) {
            ze_graph_handle_t graphHandle = nullptr;
            if (_zeGraphExt) {
                // Depending on the config, we may get an error when trying to
                // get the graph handle from the compiled network
                try {
                    graphHandle = _zeGraphExt->getGraphHandle(networkDesc->compiledNetwork);
                } catch (...) {
                    _logger.info(
                        "Failed to obtain the level zero graph handle. Inference requests for this model are not "
                        "allowed. Only exports are available");
                }
            }

            results.push_back(std::make_shared<PluginGraph>(_zeGraphExt,
                                                            _compiler,
                                                            _zeroInitStruct,
                                                            graphHandle,
                                                            std::move(networkDesc->metadata),
                                                            std::move(networkDesc->compiledNetwork),
                                                            config));
        }

        return results;
    } break;
    case 2: {
        std::vector<std::shared_ptr<NetworkDescription>> initDscrs;
        while (auto networkDescription = _compiler->compileWS_v2(model, config)) {
            if (isInit(networkDescription->metadata.name)) {
                initDscrs.push_back(networkDescription);
                continue;
            }
            OPENVINO_ASSERT(isMain(networkDescription->metadata.name),
                            "Unexpected network name: ",
                            networkDescription->metadata.name);

            mainNetworkDescription = std::move(networkDescription);
            break;
        }

        // FIXME
        initNetworkDescription = std::move(initDscrs[0]);
    } break;
    case 3: {
        std::vector<std::shared_ptr<NetworkDescription>> initDscrs;
        const std::shared_ptr<ov::Model> originalModel = model->clone();
        std::shared_ptr<ov::Model> targetModel = model;
        size_t i = 0;
        while (auto networkDescription = _compiler->compileWS_v3(targetModel, config, i++)) {
            if (isInit(networkDescription->metadata.name)) {
                initDscrs.push_back(networkDescription);
                targetModel = originalModel->clone();
                continue;
            }
            OPENVINO_ASSERT(isMain(networkDescription->metadata.name),
                            "Unexpected network name: ",
                            networkDescription->metadata.name);

            mainNetworkDescription = std::move(networkDescription);
            break;
        }

        // FIXME
        initNetworkDescription = std::move(initDscrs[0]);
    } break;
    default:
        OPENVINO_THROW("Invalid \"SEPARATE_WEIGHTS_VERSION\" value found within the \"compileWS\" call");
        break;
    }

    _logger.debug("compile end");

    auto compileNetEnd = std::chrono::steady_clock::now();
    std::cout << "Compile net time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(compileNetEnd - compileNetBegin).count() << " ms"
              << std::endl;

    ze_graph_handle_t initGraphHandle = nullptr;
    ze_graph_handle_t mainGraphHandle = nullptr;

    if (_zeGraphExt) {
        // Depending on the config, we may get an error when trying to get the graph handle from the compiled network
        try {
            initGraphHandle = _zeGraphExt->getGraphHandle(initNetworkDescription->compiledNetwork);
            mainGraphHandle = _zeGraphExt->getGraphHandle(mainNetworkDescription->compiledNetwork);
        } catch (...) {
            _logger.info("Failed to obtain the level zero graph handle. Inference requests for this model are not "
                         "allowed. Only exports are available");
        }
    }

    auto initPluginGraph = std::make_shared<PluginGraph>(_zeGraphExt,
                                                         _compiler,
                                                         _zeroInitStruct,
                                                         initGraphHandle,
                                                         std::move(initNetworkDescription->metadata),
                                                         std::move(initNetworkDescription->compiledNetwork),
                                                         config);
    auto mainPluginGraph = std::make_shared<PluginGraph>(_zeGraphExt,
                                                         _compiler,
                                                         _zeroInitStruct,
                                                         mainGraphHandle,
                                                         std::move(mainNetworkDescription->metadata),
                                                         std::move(mainNetworkDescription->compiledNetwork),
                                                         config);

    return {initPluginGraph, mainPluginGraph};
}

std::shared_ptr<IGraph> PluginCompilerAdapter::parse(std::vector<uint8_t> network, const Config& config) const {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "parse");

    _logger.debug("parse start");
    auto networkMeta = _compiler->parse(network, config);
    _logger.debug("parse end");

    ze_graph_handle_t graphHandle = nullptr;

    if (_zeGraphExt) {
        graphHandle = _zeGraphExt->getGraphHandle(network);
    }

    return std::make_shared<PluginGraph>(_zeGraphExt,
                                         _compiler,
                                         _zeroInitStruct,
                                         graphHandle,
                                         std::move(networkMeta),
                                         std::move(network),
                                         config);
}

ov::SupportedOpsMap PluginCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                 const Config& config) const {
    OV_ITT_TASK_CHAIN(QUERY_BLOB, itt::domains::NPUPlugin, "PluginCompilerAdapter", "query");

    return _compiler->query(model, config);
}

uint32_t PluginCompilerAdapter::get_version() const {
    return _compiler->get_version();
}

}  // namespace intel_npu

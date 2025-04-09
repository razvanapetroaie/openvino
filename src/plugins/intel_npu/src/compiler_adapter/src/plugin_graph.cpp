// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_graph.hpp"

#include <numeric>

#include "intel_npu/config/common.hpp"
#include "intel_npu/config/runtime.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"

namespace intel_npu {

PluginGraph::PluginGraph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
                         const ov::SoPtr<ICompiler>& compiler,
                         const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                         ze_graph_handle_t graphHandle,
                         NetworkMetadata metadata,
                         std::unique_ptr<BlobContainer> blobPtr,
                         const Config& config)
    : IGraph(graphHandle, std::move(metadata), config, std::move(blobPtr)),
      _zeGraphExt(zeGraphExt),
      _zeroInitStruct(zeroInitStruct),
      _compiler(compiler),
      _logger("PluginGraph", config.get<LOG_LEVEL>()) {
    if (!config.get<CREATE_EXECUTOR>() || config.get<DEFER_WEIGHTS_LOAD>()) {
        _logger.info("Graph initialize is deferred from the \"Graph\" constructor");
        return;
    }

    initialize(config);
}

void PluginGraph::custom_export(std::ostream& stream,
                                const std::shared_ptr<IGraph> initGraph,
                                const std::shared_ptr<ov::Model> initModel) const {
    std::stringstream xmlContent;
    std::stringstream binContent;

    ov::pass::Manager manager("SaveModel");
    manager.register_pass<ov::pass::Serialize>(xmlContent, binContent);
    manager.run_passes(initModel);

    xmlContent.seekg(0, std::ios::end);
    uint32_t xmlSize = static_cast<uint32_t>(xmlContent.tellp());
    xmlContent.seekg(0, std::ios::beg);
    binContent.seekg(0, std::ios::end);
    uint32_t binSize = static_cast<uint32_t>(binContent.tellp());
    binContent.seekg(0, std::ios::beg);

    stream << xmlSize;
    stream << xmlContent.rdbuf();

    stream << binSize;
    stream << binContent.rdbuf();

    uint32_t mainBlobSize = static_cast<uint32_t>(_blobPtr->size());
    stream << mainBlobSize;
    stream.write(reinterpret_cast<const char*>(_blobPtr->get_ptr()), _blobPtr->size());

    const auto& initBlob = initGraph->get_blob();
    uint32_t initBlobSize = static_cast<uint32_t>(initBlob->size());
    stream << initBlobSize;
    stream.write(reinterpret_cast<const char*>(initBlob->get_ptr()), initBlob->size());

    if (!stream) {
        _logger.error("Write blob to stream failed. Blob is broken!");
    } else {
        if (_logger.level() >= ov::log::Level::INFO) {
            std::stringstream str;
            str << "Blob size: " << _blobPtr->size() + initBlob->size() << std::endl;
            str << "Blob size with weights: "
                << _blobPtr->size() + initBlob->size() + 4 * sizeof(uint32_t) + xmlSize + binSize << std::endl;
            _logger.info(str.str().c_str());
        }
        _logger.info("Write blob to stream successfully.");
    }
}

void PluginGraph::custom_export_split_init(std::ostream& stream,
                                           const std::vector<std::shared_ptr<IGraph>>& initGraphs,
                                           const std::shared_ptr<ov::Model>& initModel) const {
    std::stringstream xmlContent;
    std::stringstream binContent;

    ov::pass::Manager manager("SaveModelSplitInit");
    manager.register_pass<ov::pass::Serialize>(xmlContent, binContent);
    manager.run_passes(initModel);

    xmlContent.seekg(0, std::ios::end);
    uint32_t xmlSize = static_cast<uint32_t>(xmlContent.tellp());
    xmlContent.seekg(0, std::ios::beg);
    binContent.seekg(0, std::ios::end);
    uint32_t binSize = static_cast<uint32_t>(binContent.tellp());
    binContent.seekg(0, std::ios::beg);

    stream << xmlSize;
    stream << xmlContent.rdbuf();

    stream << binSize;
    stream << binContent.rdbuf();

    uint32_t mainBlobSize = static_cast<uint32_t>(_blobPtr->size());
    stream << mainBlobSize;
    stream.write(reinterpret_cast<const char*>(_blobPtr->get_ptr()), _blobPtr->size());

    uint32_t initCount = static_cast<uint32_t>(initGraphs.size());
    stream << initCount;
    // TODO: delimiter required to separate init count from first init blob
    // size, otherwise the stream squashes two uint32_t values together.
    // alternatively, we could write blobs directly (no count before) and then
    // have a delimiter *at the end* to signify "end of init blobs" (slightly
    // better stream-length-wise?), but having explicit count means we could
    // resize vector<blob> (slightly better memory-wise?).
    stream << ':';  // random character

    for (const auto& initGraph : initGraphs) {
        const auto& initBlob = initGraph->get_blob();
        uint32_t initBlobSize = static_cast<uint32_t>(initBlob->size());
        stream << initBlobSize;
        stream.write(reinterpret_cast<const char*>(initBlob->get_ptr()), initBlob->size());

        if (!stream) {
            _logger.error("Write blob to stream failed. Blob is broken!");
            return;
        }
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        const size_t totalInitBlobSize =
            std::accumulate(initGraphs.begin(), initGraphs.end(), size_t(0), [](size_t i, const auto& graph) {
                return i + graph->get_blob()->size();
            });
        std::stringstream str;
        str << "Blob size: " << _blobPtr->size() + totalInitBlobSize << std::endl;
        str << "Blob size with weights: "
            << _blobPtr->size() + totalInitBlobSize + (size_t(4) + initGraphs.size()) * sizeof(uint32_t) + sizeof(char) +
                   xmlSize + binSize
            << std::endl;
        _logger.info(str.str().c_str());
        _logger.info("Write blob to stream successfully.");
    }
}

size_t PluginGraph::export_blob(std::ostream& stream) const {
    stream.write(reinterpret_cast<const char*>(_blobPtr->get_ptr()), _blobPtr->size());

    if (!stream) {
        _logger.error("Write blob to stream failed. Blob is broken!");
        return 0;
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        std::uint32_t result = 1171117u;
        for (const uint8_t* it = reinterpret_cast<const uint8_t*>(_blobPtr->get_ptr());
             it != reinterpret_cast<const uint8_t*>(_blobPtr->get_ptr()) + _blobPtr->size();
             ++it) {
            result = ((result << 7) + result) + static_cast<uint32_t>(*it);
        }

        std::stringstream str;
        str << "Blob size: " << _blobPtr->size() << ", hash: " << std::hex << result;
        _logger.info(str.str().c_str());
    }
    _logger.info("Write blob to stream successfully.");
    return _blobPtr->size();
}

std::vector<ov::ProfilingInfo> PluginGraph::process_profiling_output(const std::vector<uint8_t>& profData,
                                                                     const Config& config) const {
    std::vector<uint8_t> blob(_blobPtr->size());
    blob.assign(reinterpret_cast<const uint8_t*>(_blobPtr->get_ptr()),
                reinterpret_cast<const uint8_t*>(_blobPtr->get_ptr()) + _blobPtr->size());
    return _compiler->process_profiling_output(profData, blob, config);
}

void PluginGraph::set_argument_value(uint32_t argi, const void* argv) const {
    if (_zeGraphExt == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }
    _zeGraphExt->setGraphArgumentValue(_handle, argi, argv);
}

void PluginGraph::initialize(const Config& config) {
    if (_zeGraphExt == nullptr || _handle == nullptr) {
        return;
    }

    _logger.debug("Graph initialize start");

    _logger.debug("performing pfnGetProperties");
    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    auto result = _zeroInitStruct->getGraphDdiTable().pfnGetProperties(_handle, &props);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetProperties", result, _zeroInitStruct->getGraphDdiTable());

    _logger.debug("performing pfnGetArgumentProperties3");
    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        auto result = _zeroInitStruct->getGraphDdiTable().pfnGetArgumentProperties3(_handle, index, &arg3);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _zeroInitStruct->getGraphDdiTable());

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            _input_descriptors.push_back(ArgumentDescriptor{arg3, index});
        } else {
            _output_descriptors.push_back(ArgumentDescriptor{arg3, index});
        }
    }

    _input_descriptors.shrink_to_fit();
    _output_descriptors.shrink_to_fit();

    _command_queue_group_ordinal =
        zeroUtils::findCommandQueueGroupOrdinal(_zeroInitStruct->getDevice(),
                                                ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

    bool turbo = false;
    if (config.has<TURBO>()) {
        turbo = config.get<TURBO>();
    }

    _command_queue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                    zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                    _command_queue_group_ordinal,
                                                    turbo);

    if (config.has<WORKLOAD_TYPE>()) {
        set_workload_type(config.get<WORKLOAD_TYPE>());
    }

    _zeGraphExt->initializeGraph(_handle, _command_queue_group_ordinal);

    if (config.get<BATCH_MODE>() != ov::intel_npu::BatchMode::COMPILER) {
        _batch_size = get_batch_size(_metadata);
    }

    if (config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        auto number_of_command_lists = _batch_size.has_value() ? *_batch_size : 1;

        _last_submitted_event.resize(number_of_command_lists);
    }

    _logger.debug("Graph initialize finish");
}

PluginGraph::~PluginGraph() {
    // make sure all the context-dependent components are destroyed before the zero context is destroyed
    if (_handle != nullptr) {
        auto result = _zeGraphExt->destroyGraph(_handle);

        if (ZE_RESULT_SUCCESS == result) {
            _handle = nullptr;
        }
    }

    if (_last_submitted_event.size()) {
        _last_submitted_event.clear();
    }

    if (_command_queue != nullptr) {
        _command_queue.reset();
    }
}

}  // namespace intel_npu

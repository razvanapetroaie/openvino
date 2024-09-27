// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_pipeline.hpp"

#include <ze_api.h>
#include <ze_graph_ext.h>

#include "intel_npu/al/itt.hpp"
#include "intel_npu/al/prefix.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "zero_types.hpp"

namespace intel_npu {

Pipeline::Pipeline(const Config& config,
                   const std::shared_ptr<const IExecutor>& executorPtr,
                   zeroProfiling::ProfilingPool& profiling_pool,
                   zeroProfiling::ProfilingQuery& profiling_query,
                   std::shared_ptr<zeroProfiling::NpuInferProfiling> npu_profiling,
                   const std::vector<std::optional<TensorData>>& inputTensorsData,
                   const std::vector<std::optional<TensorData>>& outputTensorsData,
                   const size_t numberOfCommandLists)
    : _config(config),
      _executor(static_cast<const ZeroExecutor*>(executorPtr.get())),
      _command_queue(*_executor->getCommandQueue()),
      _event_pool{_executor->getInitStructs()->getDevice(),
                  _executor->getInitStructs()->getContext(),
                  numberOfCommandLists ? static_cast<uint32_t>(numberOfCommandLists) : 1,
                  _config},
      _npu_profiling(std::move(npu_profiling)),
      _logger("IntegratedPipeline", _config.get<LOG_LEVEL>()) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Zero_infer_request::IntegratedPipeline::IntegratedPipeline");
    _logger.debug("IntegratedPipeline - initialize started");

    if (profiling_pool.create()) {
        profiling_query.create(profiling_pool._handle);
    }

    _command_lists.reserve(numberOfCommandLists);
    _events.reserve(numberOfCommandLists);
    _fences.reserve(numberOfCommandLists);
    _logger.debug("IntegratedPipeline - emplace_back _event_pool and _command_queue");
    for (size_t i = 0; i < numberOfCommandLists; i++) {
        _command_lists.emplace_back(
            std::make_unique<CommandList>(_executor->getInitStructs()->getDevice(),
                                          _executor->getInitStructs()->getContext(),
                                          _executor->getInitStructs()->getGraphDdiTable(),
                                          _config,
                                          _executor->get_group_ordinal(),
                                          _executor->getInitStructs()->getMutableCommandListVersion() ? true : false));
        _events.emplace_back(std::make_unique<Event>(_event_pool.handle(), static_cast<uint32_t>(i), _config));
        _fences.emplace_back(std::make_unique<Fence>(_command_queue, _config));
    }

    for (size_t i = 0; i < numberOfCommandLists; i++) {
        size_t ioIndex = 0;
        for (const auto& desc : _executor->get_input_descriptors()) {
            _executor->setArgumentValue(desc.idx,
                                        static_cast<const unsigned char*>(inputTensorsData.at(ioIndex)->mem) +
                                            (i * inputTensorsData.at(ioIndex)->size) / numberOfCommandLists);
            ++ioIndex;
        }

        ioIndex = 0;
        for (const auto& desc : _executor->get_output_descriptors()) {
            _executor->setArgumentValue(desc.idx,
                                        static_cast<const unsigned char*>(outputTensorsData.at(ioIndex)->mem) +
                                            (i * outputTensorsData.at(ioIndex)->size) / numberOfCommandLists);
            ++ioIndex;
        }

        /// append timestamp command if feature was activated
        if (_npu_profiling != nullptr) {
            _command_lists.at(i)->appendBarrier();
            _command_lists.at(i)->appendNpuTimestamp(reinterpret_cast<uint64_t*>(_npu_profiling->npu_ts_infer_start));
        }

        _command_lists.at(i)->appendGraphExecute(_executor->graph(), profiling_query.getHandle());

        /// append timestamp command if feature was activated
        if (_npu_profiling != nullptr) {
            _command_lists.at(i)->appendBarrier();
            _command_lists.at(i)->appendNpuTimestamp(reinterpret_cast<uint64_t*>(_npu_profiling->npu_ts_infer_end));
        }

        // appendBarrier used in L0 as well
        if (!sync_output_with_fences_) {
            _command_lists.at(i)->appendBarrier();
            _events.at(i)->AppendSignalEvent(*_command_lists.at(i));
        }
        _command_lists.at(i)->close();
    }
    _logger.debug("IntegratedPipeline - initialize completed");
}

void Pipeline::push() {
    _logger.debug("IntegratedPipeline - push() started");

    for (size_t i = 0; i < _command_lists.size(); ++i) {
        OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_PUSH, itt::domains::LevelZeroBackend, "IntegratedPipeline", "push");
        if (sync_output_with_fences_) {
            _command_queue.executeCommandList(*_command_lists.at(i), *_fences.at(i));
        } else {
            _command_queue.executeCommandList(*_command_lists.at(i));
        }
    }

    _logger.debug("IntegratedPipeline - push() completed");
};

void Pipeline::pull() {
    _logger.debug("IntegratedPipeline - pull() started");
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_PULL, itt::domains::LevelZeroBackend, "IntegratedPipeline", "pull");

    for (size_t i = 0; i < _command_lists.size(); ++i) {
        if (sync_output_with_fences_) {
            _fences.at(i)->hostSynchronize();
        } else {
            _events.at(i)->hostSynchronize();
        }
        /// sample npu timestamps if feature was activated
        if (_npu_profiling != nullptr) {
            _npu_profiling->sampleNpuTimestamps();
        }
    }

    _logger.debug("IntegratedPipeline - pull() completed");
};

void Pipeline::reset() const {
    _logger.debug("IntegratedPipeline - rest() started");

    for (size_t i = 0; i < _command_lists.size(); ++i) {
        if (sync_output_with_fences_) {
            _fences.at(i)->reset();
        } else {
            _events.at(i)->reset();
        }
    }

    _logger.debug("IntegratedPipeline - rest() completed");
};

void Pipeline::updateCommandList(const TensorData& tensorsData, const uint32_t index) {
    OV_ITT_TASK_CHAIN(ZERO_EXECUTOR_IP_UMCL, itt::domains::LevelZeroBackend, "IntegratedPipeline", "updateCommandList");
    const size_t numberOfCommandLists = _command_lists.size();

    for (size_t i = 0; i < numberOfCommandLists; i++) {
        _command_lists.at(i)->updateMutableCommandList(
            index,
            static_cast<const unsigned char*>(tensorsData.mem) + (i * tensorsData.size) / numberOfCommandLists);
        _command_lists.at(i)->close();
    }
};

}  // namespace intel_npu

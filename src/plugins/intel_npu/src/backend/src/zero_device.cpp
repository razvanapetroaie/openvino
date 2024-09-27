// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_device.hpp"

#include "intel_npu/al/itt.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "zero_executor.hpp"
#include "zero_host_tensor.hpp"
#include "zero_infer_request.hpp"
#include "zero_remote_tensor.hpp"
#include "zero_utils.hpp"

using namespace intel_npu;

namespace {
struct ArgumentDescriptor {
    ze_graph_argument_properties_3_t info;
    uint32_t idx;
};
}  // namespace

ZeroDevice::ZeroDevice(const std::shared_ptr<ZeroInitStructsHolder>& initStructs)
    : _initStructs(initStructs),
      _graph_ddi_table_ext(_initStructs->getGraphDdiTable()),
      log("ZeroDevice", Logger::global().level()) {
    log.debug("ZeroDevice::ZeroDevice init");
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    zeroUtils::throwOnFail("zeDeviceGetProperties",
                           zeDeviceGetProperties(_initStructs->getDevice(), &device_properties));

    // Query PCI information
    // Older drivers do not have this implementend. Linux driver returns NOT_IMPLEMENTED, while windows driver returns
    // zero values. If this is detected, we populate only device with ID from device_properties for backwards
    // compatibility. For any other error, we just fall-back to device ID to assure backwards compatibilty with even
    // older drivers
    pci_properties.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    ze_result_t retpci = zeDevicePciGetPropertiesExt(_initStructs->getDevice(), &pci_properties);
    if (ZE_RESULT_SUCCESS == retpci) {
        // windows driver specific backwards compatibility
        if (pci_properties.address.device == 0) {
            log.warning("PCI information not available in driver. Falling back to deviceId");
            pci_properties.address.device = device_properties.deviceId;
        }
    } else {
        // general backwards compatibility
        log.warning("PCI information not available in driver. Falling back to deviceId");
        pci_properties.address.device = device_properties.deviceId;
    }

    /// Calculate and store device GOPS with formula: frequency * number of tiles * ops per tile
    /// cross-OS backwards compatibilty: only calculate gops if driver supports it (version>x)
    uint32_t gops_support_drv_version = UINT32_MAX;
#if defined(_WIN32) || defined(__CYGWIN__)
    gops_support_drv_version = 2465;  /// Windows driver version which supports Gops calculations
#else                                 // _WIN32 || __CYGWIN__
    gops_support_drv_version = 1715354569;  /// Linux driver version which supports Gops calculations
#endif                                // _WIN32 || __CYGWIN__
    if (_initStructs->getDriverVersion() >= gops_support_drv_version) {
        float gops = (device_properties.coreClockRate / powf(1000, 3)) * device_properties.numSlices *
                     device_properties.physicalEUSimdWidth;
        device_gops[ov::element::f32] = 0;
        device_gops[ov::element::u8] = gops;
        device_gops[ov::element::i8] = gops;
        device_gops[ov::element::f16] = 0.5f * gops;
    }

    std::vector<ze_command_queue_group_properties_t> command_group_properties;
    uint32_t command_queue_group_count = 0;
    // Discover all command queue groups
    zeroUtils::throwOnFail(
        "zeDeviceGetCommandQueueGroupProperties",
        zeDeviceGetCommandQueueGroupProperties(_initStructs->getDevice(), &command_queue_group_count, nullptr));

    log.debug("ZeroDevice::ZeroDevice - resize command_queue_group_count");
    command_group_properties.resize(command_queue_group_count);

    for (auto& prop : command_group_properties) {
        prop.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        prop.pNext = nullptr;
    }

    zeroUtils::throwOnFail("zeDeviceGetCommandQueueGroupProperties",
                           zeDeviceGetCommandQueueGroupProperties(_initStructs->getDevice(),
                                                                  &command_queue_group_count,
                                                                  command_group_properties.data()));

    // Find the corresponding command queue group.
    log.debug("ZeroDevice::ZeroDevice - findGroupOrdinal");
    _group_ordinal = zeroUtils::findGroupOrdinal(command_group_properties, device_properties);
    log.debug("ZeroDevice::ZeroDevice - init completed");
}

std::shared_ptr<IExecutor> ZeroDevice::createExecutor(
    const std::shared_ptr<const NetworkDescription>& networkDescription,
    const Config& config) {
    OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "Device::createExecutor");
    return std::make_shared<ZeroExecutor>(_initStructs, networkDescription, config, _group_ordinal);
}

void ZeroDevice::runInit(const std::shared_ptr<NetworkDescription>& initNetworkDescription,
                         const std::shared_ptr<NetworkDescription>& mainNetworkDescription,
                         const Config& config) {
    // TODO logger
    // OV_ITT_SCOPED_TASK(itt::domains::LevelZeroBackend, "ZeroDevice::runInit");

    // auto graph_ddi_table_ext = _initStructs->getGraphDdiTable();
    // auto _command_queues = std::make_shared<CommandQueue>(_initStructs->getDevice(),
    //                                                       _initStructs->getContext(),
    //                                                       zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
    //                                                       _initStructs->getCommandQueueDdiTable(),
    //                                                       config,
    //                                                       _group_ordinal);

    // CommandList graph_command_list(_initStructs->getDevice(),
    //                                _initStructs->getContext(),
    //                                _initStructs->getGraphDdiTable(),
    //                                config,
    //                                _group_ordinal);
    // CommandQueue graph_command_queue(_initStructs->getDevice(),
    //                                  _initStructs->getContext(),
    //                                  ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
    //                                  _initStructs->getCommandQueueDdiTable(),
    //                                  config,
    //                                  _group_ordinal);
    // Fence fence(graph_command_queue, config);
    // const bool useGraphHandle = initNetworkDescription->metadata.graphHandle;
    // ze_graph_handle_t init_graph = nullptr;
    // ze_graph_handle_t main_graph = nullptr;
    // ze_graph_properties_t init_props{};
    // ze_graph_properties_t main_props{};
    // std::vector<ArgumentDescriptor> init_input_descriptors;
    // std::vector<ArgumentDescriptor> main_input_descriptors;
    // std::vector<ArgumentDescriptor> init_output_descriptors;
    // std::vector<ArgumentDescriptor> main_output_descriptors;

    // // _graph is a nullptr for CIP path, a new handle will be obtained from the driver based on the given
    // // compiledNetwork _graph gets (reuses) graphHandle from the compiler for CID path
    // if (!useGraphHandle) {
    //     ze_graph_desc_t init_desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
    //                               nullptr,
    //                               ZE_GRAPH_FORMAT_NATIVE,
    //                               initNetworkDescription->compiledNetwork.size(),
    //                               initNetworkDescription->compiledNetwork.data(),
    //                               nullptr};
    //     ze_graph_desc_t main_desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
    //                               nullptr,
    //                               ZE_GRAPH_FORMAT_NATIVE,
    //                               mainNetworkDescription->compiledNetwork.size(),
    //                               mainNetworkDescription->compiledNetwork.data(),
    //                               nullptr};

    //     zeroUtils::throwOnFail("pfnCreate",
    //                            _graph_ddi_table_ext->pfnCreate(_initStructs->getContext(),
    //                                                            _initStructs->getDevice(),
    //                                                            &init_desc,
    //                                                            &init_graph));
    //     zeroUtils::throwOnFail("pfnCreate",
    //                            _graph_ddi_table_ext->pfnCreate(_initStructs->getContext(),
    //                                                            _initStructs->getDevice(),
    //                                                            &main_desc,
    //                                                            &main_graph));

    // } else {
    //     init_graph = static_cast<ze_graph_handle_t>(initNetworkDescription->metadata.graphHandle);
    //     main_graph = static_cast<ze_graph_handle_t>(mainNetworkDescription->metadata.graphHandle);
    // }

    // zeroUtils::throwOnFail("pfnGetProperties", _graph_ddi_table_ext->pfnGetProperties(init_graph, &init_props));
    // zeroUtils::throwOnFail("pfnGetProperties", _graph_ddi_table_ext->pfnGetProperties(main_graph, &main_props));

    // auto targetDriverExtVersion = _initStructs->getDriverExtVersion();
    // if (targetDriverExtVersion <= ZE_GRAPH_EXT_VERSION_1_1) {
    //     OPENVINO_THROW("Incompatibility between the NPU plugin and driver! The driver version is too old, please "
    //                    "update the driver version");
    // }

    // for (uint32_t index = 0; index < init_props.numGraphArgs; ++index) {
    //     ze_graph_argument_properties_3_t arg3;
    //     zeroUtils::throwOnFail("pfnGetArgumentProperties3",
    //                            _graph_ddi_table_ext->pfnGetArgumentProperties3(init_graph, index, &arg3));

    //     if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
    //         init_input_descriptors.push_back(ArgumentDescriptor{arg3, index});
    //     } else {
    //         init_output_descriptors.push_back(ArgumentDescriptor{arg3, index});
    //     }
    // }
    // for (uint32_t index = 0; index < main_props.numGraphArgs; ++index) {
    //     ze_graph_argument_properties_3_t arg3;
    //     zeroUtils::throwOnFail("pfnGetArgumentProperties3",
    //                            _graph_ddi_table_ext->pfnGetArgumentProperties3(main_graph, index, &arg3));

    //     if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
    //         main_input_descriptors.push_back(ArgumentDescriptor{arg3, index});
    //     } else {
    //         main_output_descriptors.push_back(ArgumentDescriptor{arg3, index});
    //     }
    // }

    // graph_command_list.appendGraphInitialize(init_graph);
    // graph_command_list.close();

    // graph_command_queue.executeCommandList(graph_command_list, fence);
    // fence.hostSynchronize();
}

std::string ZeroDevice::getName() const {
//    KMD is setting usDeviceID from VpuFamilyID.h
#define NPU_3720_P_DEVICE_ID 0x7D1D
#define NPU_3720_S_DEVICE_ID 0xAD1D
#define NPU_4000_DEVICE_ID   0x643E

    std::string name;
    switch (device_properties.deviceId) {
    case NPU_3720_P_DEVICE_ID:
    case NPU_3720_S_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU3720;
        break;
    case NPU_4000_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU4000;
        break;
    default:
        name = ov::intel_npu::Platform::AUTO_DETECT;
    }

    return name;
}

std::string ZeroDevice::getFullDeviceName() const {
    return device_properties.name;
}

IDevice::Uuid ZeroDevice::getUuid() const {
    Uuid uuid{};
    static_assert(sizeof(device_properties.uuid.id) == uuid.uuid.size(),
                  "ze_device_uuid_t::id size doesn't match intel_npu::Uuid::uuid size");

    std::copy(std::begin(device_properties.uuid.id), std::end(device_properties.uuid.id), std::begin(uuid.uuid));

    return uuid;
}

uint32_t ZeroDevice::getSubDevId() const {
    return device_properties.subdeviceId;
}

uint32_t ZeroDevice::getMaxNumSlices() const {
    return device_properties.numSlices;
}

uint64_t ZeroDevice::getAllocMemSize() const {
    ze_graph_memory_query_t query{};
    zeroUtils::throwOnFail(
        "pfnQueryContextMemory",
        _graph_ddi_table_ext.pfnQueryContextMemory(_initStructs->getContext(), ZE_GRAPH_QUERY_MEMORY_DDR, &query));
    return query.allocated;
}

uint64_t ZeroDevice::getTotalMemSize() const {
    ze_graph_memory_query_t query{};
    zeroUtils::throwOnFail(
        "pfnQueryContextMemory",
        _graph_ddi_table_ext.pfnQueryContextMemory(_initStructs->getContext(), ZE_GRAPH_QUERY_MEMORY_DDR, &query));
    return query.total;
}

ov::device::PCIInfo ZeroDevice::getPciInfo() const {
    return ov::device::PCIInfo{pci_properties.address.domain,
                               pci_properties.address.bus,
                               pci_properties.address.device,
                               pci_properties.address.function};
}

std::map<ov::element::Type, float> ZeroDevice::getGops() const {
    return device_gops;
}

ov::device::Type ZeroDevice::getDeviceType() const {
    return ov::device::Type::INTEGRATED;
}

std::shared_ptr<SyncInferRequest> ZeroDevice::createInferRequest(
    const std::shared_ptr<const ICompiledModel>& compiledModel,
    const std::shared_ptr<IExecutor>& executor,
    const Config& config) {
    return std::make_shared<ZeroInferRequest>(_initStructs, compiledModel, executor, config);
}

ov::SoPtr<ov::IRemoteTensor> ZeroDevice::createRemoteTensor(std::shared_ptr<ov::IRemoteContext> context,
                                                            const ov::element::Type& element_type,
                                                            const ov::Shape& shape,
                                                            const Config& config,
                                                            ov::intel_npu::TensorType tensor_type,
                                                            ov::intel_npu::MemType mem_type,
                                                            void* mem) {
    return {std::make_shared<
        ZeroRemoteTensor>(context, _initStructs, element_type, shape, config, tensor_type, mem_type, mem)};
};

ov::SoPtr<ov::ITensor> ZeroDevice::createHostTensor(std::shared_ptr<ov::IRemoteContext> context,
                                                    const ov::element::Type& element_type,
                                                    const ov::Shape& shape,
                                                    const Config& config) {
    return {std::make_shared<ZeroHostTensor>(context, _initStructs, element_type, shape, config)};
};

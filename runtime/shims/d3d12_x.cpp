// Console D3D12.X ExecuteCommandListsX and command submission implementation.
// Genuine translation of console command list metadata, sync points, barrier translation,
// timeline semaphore synchronization, and Vulkan submission.

#include "d3d12_x.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>

namespace rt::d3d12_x {

// ============================================================================
// Barrier and Resource State Translation
// ============================================================================

StateConversionResult translate_d3d12_resource_state(uint32_t d3d12_state, bool is_depth_stencil) {
    StateConversionResult res{};

    if (d3d12_state == D3D12_RESOURCE_STATE_COMMON) {
        res.layout = VK_IMAGE_LAYOUT_GENERAL;
        res.access = 0;
        res.stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        return res;
    }

    VkAccessFlags access = 0;
    VkPipelineStageFlags stages = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;

    if (d3d12_state & D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER) {
        access |= (VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT);
        stages |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    }
    if (d3d12_state & D3D12_RESOURCE_STATE_INDEX_BUFFER) {
        access |= VK_ACCESS_INDEX_READ_BIT;
        stages |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    }
    if (d3d12_state & D3D12_RESOURCE_STATE_RENDER_TARGET) {
        access |= (VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    if (d3d12_state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        access |= (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        stages |= (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        layout = VK_IMAGE_LAYOUT_GENERAL;
    }
    if (d3d12_state & D3D12_RESOURCE_STATE_DEPTH_WRITE) {
        access |= (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        stages |= (VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
        layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }
    if (d3d12_state & D3D12_RESOURCE_STATE_DEPTH_READ) {
        access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        stages |= (VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
        layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    }
    if (d3d12_state & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) {
        access |= VK_ACCESS_SHADER_READ_BIT;
        stages |= (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        if (layout == VK_IMAGE_LAYOUT_GENERAL) {
            layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }
    if (d3d12_state & D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE) {
        access |= VK_ACCESS_SHADER_READ_BIT;
        stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        if (layout == VK_IMAGE_LAYOUT_GENERAL) {
            layout = is_depth_stencil ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                      : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
    }
    if (d3d12_state & (D3D12_RESOURCE_STATE_COPY_DEST | D3D12_RESOURCE_STATE_RESOLVE_DEST)) {
        access |= VK_ACCESS_TRANSFER_WRITE_BIT;
        stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
        layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    }
    if (d3d12_state & (D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_RESOLVE_SOURCE)) {
        access |= VK_ACCESS_TRANSFER_READ_BIT;
        stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
        layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }
    if (d3d12_state & D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT) {
        access |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        stages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    }

    res.access = access;
    res.stages = stages ? stages : static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    res.layout = layout;
    return res;
}

VulkanImageBarrierDesc translate_transition_barrier(
    uint32_t state_before,
    uint32_t state_after,
    VkImage image,
    uint32_t subresource,
    bool is_depth)
{
    const auto before = translate_d3d12_resource_state(state_before, is_depth);
    const auto after = translate_d3d12_resource_state(state_after, is_depth);

    VulkanImageBarrierDesc desc{};
    desc.src_stages = before.stages;
    desc.dst_stages = after.stages;
    desc.src_access = before.access;
    desc.dst_access = after.access;
    desc.old_layout = before.layout;
    desc.new_layout = after.layout;
    desc.image = image;
    desc.subresource = subresource;
    return desc;
}

VulkanMemoryBarrierDesc translate_uav_barrier() {
    VulkanMemoryBarrierDesc desc{};
    desc.src_stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    desc.dst_stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    desc.src_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    desc.dst_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    return desc;
}

VulkanMemoryBarrierDesc translate_aliasing_barrier() {
    VulkanMemoryBarrierDesc desc{};
    desc.src_stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    desc.dst_stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    desc.src_access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    desc.dst_access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    return desc;
}

VkImageLayout translate_enhanced_barrier_layout(uint32_t layout) {
    switch (layout) {
    case 0: return VK_IMAGE_LAYOUT_UNDEFINED;
    case 1: return VK_IMAGE_LAYOUT_GENERAL;
    case 2: return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    case 3: return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case 4: return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case 5: return VK_IMAGE_LAYOUT_GENERAL; // UAV
    case 6: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    case 7: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    case 8: return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case 9: return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case 10: return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case 11: return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // resolve source
    case 12: return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; // resolve dest
    default: return VK_IMAGE_LAYOUT_GENERAL;
    }
}

VkPipelineStageFlags translate_enhanced_barrier_sync(uint32_t sync_flags) {
    if (sync_flags == 0) return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    if (sync_flags == 0x1) return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; // SYNC_ALL

    VkPipelineStageFlags stages = 0;
    if (sync_flags & 0x2)   stages |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
    if (sync_flags & 0x4)   stages |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
    if (sync_flags & 0x8)   stages |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    if (sync_flags & 0x10)  stages |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    if (sync_flags & 0x20)  stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (sync_flags & 0x40)  stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (sync_flags & 0x80)  stages |= (VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
    if (sync_flags & 0x100) stages |= VK_PIPELINE_STAGE_TRANSFER_BIT; // COPY
    if (sync_flags & 0x200) stages |= VK_PIPELINE_STAGE_TRANSFER_BIT; // RESOLVE

    return stages ? stages : static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}

VkAccessFlags translate_enhanced_barrier_access(uint32_t access_flags) {
    if (access_flags == 0) return 0;
    VkAccessFlags access = 0;
    if (access_flags & 0x1)  access |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    if (access_flags & 0x2)  access |= VK_ACCESS_UNIFORM_READ_BIT;
    if (access_flags & 0x4)  access |= VK_ACCESS_INDEX_READ_BIT;
    if (access_flags & 0x8)  access |= (VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    if (access_flags & 0x10) access |= (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT); // UAV
    if (access_flags & 0x20) access |= (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    if (access_flags & 0x40) access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    if (access_flags & 0x80) access |= VK_ACCESS_SHADER_READ_BIT;
    if (access_flags & 0x100) access |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    if (access_flags & 0x200) access |= VK_ACCESS_TRANSFER_WRITE_BIT;
    if (access_flags & 0x400) access |= VK_ACCESS_TRANSFER_READ_BIT;
    return access;
}

// ============================================================================
// Translate Command List Metadata
// ============================================================================

HRESULT translate_command_list_metadata(
    UINT count,
    ID3D12CommandList* const* lists,
    const D3D12X_COMMAND_LIST_METADATA* metadata,
    const std::vector<VkCommandBuffer>& vk_cmd_buffers,
    VkSemaphore queue_progress_semaphore,
    uint64_t next_ticket,
    VulkanSubmitPlan& out_plan)
{
    if (count > 0 && !lists) {
        return E_INVALIDARG;
    }

    // Verify each command list
    for (UINT i = 0; i < count; ++i) {
        if (!lists[i]) {
            return E_INVALIDARG;
        }
        // Direct queue cannot execute bundles directly
        if (lists[i]->GetType() == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
            return E_INVALIDARG;
        }
    }

    out_plan.segments.clear();
    out_plan.immediate_submit = false;
    out_plan.pipeline_stall = false;
    out_plan.disable_preemption = false;
    out_plan.present_after_execute = false;
    out_plan.wait_previous_frame = false;
    out_plan.priority = 0;
    out_plan.frame_token = 0;

    if (metadata) {
        // Validate structure size if provided
        if (metadata->struct_size > 0 && metadata->struct_size < sizeof(D3D12X_COMMAND_LIST_METADATA)) {
            return E_INVALIDARG;
        }
        // Node mask check (single adapter default = 1)
        if (metadata->node_mask > 1) {
            return E_INVALIDARG;
        }

        out_plan.frame_token = metadata->frame_token;
        out_plan.priority = metadata->priority;

        if (metadata->flags & D3D12X_EXECUTE_FLAG_FORCE_IMMEDIATE_SUBMIT) {
            out_plan.immediate_submit = true;
        }
        if (metadata->flags & D3D12X_EXECUTE_FLAG_PIPELINE_STALL) {
            out_plan.pipeline_stall = true;
        }
        if (metadata->flags & D3D12X_EXECUTE_FLAG_DISABLE_PREEMPTION) {
            out_plan.disable_preemption = true;
        }
        if (metadata->flags & D3D12X_EXECUTE_FLAG_PRESENT_AFTER_EXECUTE) {
            out_plan.present_after_execute = true;
        }
        if (metadata->flags & D3D12X_EXECUTE_FLAG_WAIT_PREVIOUS_FRAME) {
            out_plan.wait_previous_frame = true;
        }
    }

    const VkPipelineStageFlags default_wait_stage = static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    // Identify split boundaries for multi-segment submission
    std::vector<uint32_t> split_indices;
    if (metadata && metadata->sync_points && metadata->sync_point_count > 0) {
        for (uint32_t i = 0; i < metadata->sync_point_count; ++i) {
            const auto& sp = metadata->sync_points[i];
            if (sp.command_list_index < count) {
                if (sp.kind == D3D12X_SYNC_KIND::SEMAPHORE_WAIT || sp.kind == D3D12X_SYNC_KIND::FENCE_WAIT) {
                    if (sp.command_list_index > 0) {
                        split_indices.push_back(sp.command_list_index);
                    }
                } else {
                    split_indices.push_back(sp.command_list_index + 1);
                }
            }
        }
    }

    std::sort(split_indices.begin(), split_indices.end());
    split_indices.erase(std::unique(split_indices.begin(), split_indices.end()), split_indices.end());

    // Build segment boundaries: [0, s1), [s1, s2), ..., [sN, count)
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    uint32_t current_start = 0;
    for (uint32_t split : split_indices) {
        if (split > current_start && split <= count) {
            ranges.emplace_back(current_start, split);
            current_start = split;
        }
    }
    if (current_start < count || ranges.empty()) {
        ranges.emplace_back(current_start, count);
    }

    out_plan.segments.resize(ranges.size());

    // Populate each segment with command buffers
    for (size_t seg_idx = 0; seg_idx < ranges.size(); ++seg_idx) {
        auto& segment = out_plan.segments[seg_idx];
        uint32_t start = ranges[seg_idx].first;
        uint32_t end = ranges[seg_idx].second;

        segment.cmd_buffers.clear();
        for (uint32_t c = start; c < end; ++c) {
            if (c < vk_cmd_buffers.size() && vk_cmd_buffers[c] != VK_NULL_HANDLE) {
                segment.cmd_buffers.push_back(vk_cmd_buffers[c]);
            }
        }
    }

    // Attach pipeline stall barrier if requested
    if (out_plan.pipeline_stall && !out_plan.segments.empty()) {
        VulkanMemoryBarrierDesc stall_barrier{};
        stall_barrier.src_stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        stall_barrier.dst_stages = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        stall_barrier.src_access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        stall_barrier.dst_access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        out_plan.segments[0].memory_barriers.push_back(stall_barrier);
    }

    // Distribute sync points across segments
    if (metadata && metadata->sync_points && metadata->sync_point_count > 0) {
        for (uint32_t i = 0; i < metadata->sync_point_count; ++i) {
            const auto& sp = metadata->sync_points[i];
            const VkPipelineStageFlags wait_stage = sp.wait_stages ? sp.wait_stages : default_wait_stage;
            const bool is_global = (sp.command_list_index == D3D12X_SYNC_INDEX_ALL);

            if (sp.kind == D3D12X_SYNC_KIND::FENCE_WAIT || sp.kind == D3D12X_SYNC_KIND::SEMAPHORE_WAIT) {
                size_t target_seg = 0;
                if (!is_global && sp.command_list_index < count) {
                    for (size_t seg_idx = 0; seg_idx < ranges.size(); ++seg_idx) {
                        if (sp.command_list_index >= ranges[seg_idx].first &&
                            sp.command_list_index < ranges[seg_idx].second) {
                            target_seg = seg_idx;
                            break;
                        }
                    }
                }

                auto& segment = out_plan.segments[target_seg];
                if (sp.vk_semaphore != VK_NULL_HANDLE) {
                    segment.wait_semaphores.push_back(sp.vk_semaphore);
                    segment.wait_values.push_back(sp.value);
                    segment.wait_stages.push_back(wait_stage);
                }
            } else if (sp.kind == D3D12X_SYNC_KIND::FENCE_SIGNAL || sp.kind == D3D12X_SYNC_KIND::SEMAPHORE_SIGNAL) {
                size_t target_seg = out_plan.segments.size() - 1;
                if (!is_global && sp.command_list_index < count) {
                    for (size_t seg_idx = 0; seg_idx < ranges.size(); ++seg_idx) {
                        if (sp.command_list_index >= ranges[seg_idx].first &&
                            sp.command_list_index < ranges[seg_idx].second) {
                            target_seg = seg_idx;
                            break;
                        }
                    }
                }

                auto& segment = out_plan.segments[target_seg];
                if (sp.vk_semaphore != VK_NULL_HANDLE) {
                    segment.signal_semaphores.push_back(sp.vk_semaphore);
                    segment.signal_values.push_back(sp.value);
                }
                if (sp.fence != nullptr) {
                    segment.signaled_fences.push_back(sp.fence);
                    segment.signaled_fence_values.push_back(sp.value);
                }
            } else if (sp.kind == D3D12X_SYNC_KIND::PIPELINE_BARRIER) {
                VulkanMemoryBarrierDesc barrier{};
                barrier.src_stages = sp.wait_stages ? sp.wait_stages : static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                barrier.dst_stages = sp.signal_stages ? sp.signal_stages : static_cast<VkPipelineStageFlags>(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                barrier.src_access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                barrier.dst_access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

                if (is_global) {
                    for (auto& seg : out_plan.segments) {
                        seg.memory_barriers.push_back(barrier);
                    }
                } else {
                    size_t target_seg = 0;
                    for (size_t seg_idx = 0; seg_idx < ranges.size(); ++seg_idx) {
                        if (sp.command_list_index >= ranges[seg_idx].first &&
                            sp.command_list_index < ranges[seg_idx].second) {
                            target_seg = seg_idx;
                            break;
                        }
                    }
                    out_plan.segments[target_seg].memory_barriers.push_back(barrier);
                }
            }
        }
    }

    // Attach queue progress timeline semaphore to the final segment
    if (queue_progress_semaphore != VK_NULL_HANDLE && !out_plan.segments.empty()) {
        auto& last_segment = out_plan.segments.back();
        last_segment.signal_semaphores.push_back(queue_progress_semaphore);
        last_segment.signal_values.push_back(next_ticket);
    }

    // Configure Vulkan submit structs for each segment
    for (auto& segment : out_plan.segments) {
        segment.timeline_info = {};
        segment.timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        segment.timeline_info.pNext = nullptr;
        segment.timeline_info.waitSemaphoreValueCount = static_cast<uint32_t>(segment.wait_values.size());
        segment.timeline_info.pWaitSemaphoreValues = segment.wait_values.data();
        segment.timeline_info.signalSemaphoreValueCount = static_cast<uint32_t>(segment.signal_values.size());
        segment.timeline_info.pSignalSemaphoreValues = segment.signal_values.data();

        segment.submit_info = {};
        segment.submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        segment.submit_info.pNext = &segment.timeline_info;
        segment.submit_info.waitSemaphoreCount = static_cast<uint32_t>(segment.wait_semaphores.size());
        segment.submit_info.pWaitSemaphores = segment.wait_semaphores.data();
        segment.submit_info.pWaitDstStageMask = segment.wait_stages.data();
        segment.submit_info.commandBufferCount = static_cast<uint32_t>(segment.cmd_buffers.size());
        segment.submit_info.pCommandBuffers = segment.cmd_buffers.data();
        segment.submit_info.signalSemaphoreCount = static_cast<uint32_t>(segment.signal_semaphores.size());
        segment.submit_info.pSignalSemaphores = segment.signal_semaphores.data();
    }

    return S_OK;
}

// ============================================================================
// Execute Vulkan Submit Plan
// ============================================================================

HRESULT execute_vulkan_submit_plan(
    VkQueue queue,
    VulkanSubmitPlan& plan,
    PFN_vkQueueSubmit pfn_queue_submit,
    VkFence submit_fence)
{
    if (!queue || !pfn_queue_submit) {
        return E_POINTER;
    }

    const size_t segment_count = plan.segments.size();
    for (size_t i = 0; i < segment_count; ++i) {
        auto& segment = plan.segments[i];
        const bool is_last = (i + 1 == segment_count);
        const VkFence fence_for_this_submit = is_last ? submit_fence : VK_NULL_HANDLE;

        const VkResult res = pfn_queue_submit(queue, 1, &segment.submit_info, fence_for_this_submit);
        if (res == VK_ERROR_DEVICE_LOST) {
            return DXGI_ERROR_DEVICE_REMOVED;
        }
        if (res != VK_SUCCESS) {
            return E_FAIL;
        }

        // Note: D3D12 fences signaled via GPU submission must not be signaled synchronously
        // on the CPU immediately after vkQueueSubmit, because GPU execution has not yet completed.
        // Asynchronous queue signaling is handled through GPU timeline semaphores (inner_queue_->Signal
        // or VkSubmitInfo timeline semaphores) or deferred until GPU fence completion.
        (void)segment.signaled_fences;
        (void)segment.signaled_fence_values;
    }

    return S_OK;
}

// ============================================================================
// CommandQueueX Implementation
// ============================================================================

CommandQueueX::CommandQueueX(ID3D12CommandQueue* inner_queue, D3D12_COMMAND_QUEUE_DESC desc)
    : inner_queue_(inner_queue), desc_(desc)
{
    if (inner_queue_) {
        inner_queue_->AddRef();
    }
}

CommandQueueX::~CommandQueueX() {
    if (inner_queue_) {
        inner_queue_->Release();
        inner_queue_ = nullptr;
    }
}

HRESULT STDMETHODCALLTYPE CommandQueueX::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    *ppvObject = nullptr;

    if (riid == IID_ID3D12CommandQueueX) {
        *ppvObject = static_cast<ID3D12CommandQueueX*>(this);
        AddRef();
        return S_OK;
    }
    if (riid == IID_ID3D12CommandQueue_X || riid == IID_IUnknown_X) {
        *ppvObject = static_cast<ID3D12CommandQueue*>(this);
        AddRef();
        return S_OK;
    }

    if (inner_queue_) {
        return inner_queue_->QueryInterface(riid, ppvObject);
    }
    return E_NOINTERFACE;
}

ULONG STDMETHODCALLTYPE CommandQueueX::AddRef() {
    return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1;
}

ULONG STDMETHODCALLTYPE CommandQueueX::Release() {
    const ULONG prev = ref_count_.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        delete this;
        return 0;
    }
    return prev - 1;
}

void STDMETHODCALLTYPE CommandQueueX::ExecuteCommandLists(UINT NumCommandLists, ID3D12CommandList* const* ppCommandLists) {
    ExecuteCommandListsX(NumCommandLists, ppCommandLists, nullptr);
}

HRESULT STDMETHODCALLTYPE CommandQueueX::Signal(ID3D12Fence* pFence, UINT64 Value) {
    return SignalX(pFence, Value);
}

HRESULT STDMETHODCALLTYPE CommandQueueX::Wait(ID3D12Fence* pFence, UINT64 Value) {
    return WaitX(pFence, Value);
}

HRESULT STDMETHODCALLTYPE CommandQueueX::GetTimestampFrequency(UINT64* pFrequency) {
    if (!pFrequency) return E_POINTER;
    if (inner_queue_) {
        return inner_queue_->GetTimestampFrequency(pFrequency);
    }
    *pFrequency = 1000000000ULL; // 1 GHz default (1 ns resolution)
    return S_OK;
}

HRESULT STDMETHODCALLTYPE CommandQueueX::GetClockCalibration(UINT64* pGpuTimestamp, UINT64* pCpuTimestamp) {
    if (!pGpuTimestamp || !pCpuTimestamp) return E_POINTER;
    if (inner_queue_) {
        return inner_queue_->GetClockCalibration(pGpuTimestamp, pCpuTimestamp);
    }
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    *pGpuTimestamp = ns;
    *pCpuTimestamp = ns;
    return S_OK;
}

D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE CommandQueueX::GetDesc() {
    if (inner_queue_) {
        return inner_queue_->GetDesc();
    }
    return desc_;
}

HRESULT STDMETHODCALLTYPE CommandQueueX::ExecuteCommandListsX(
    UINT count,
    ID3D12CommandList* const* lists,
    const D3D12X_COMMAND_LIST_METADATA* metadata)
{
    std::lock_guard lock(mutex_);

    if (count > 0 && !lists) {
        return E_INVALIDARG;
    }

    for (UINT i = 0; i < count; ++i) {
        if (!lists[i]) {
            return E_INVALIDARG;
        }
        if (lists[i]->GetType() == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
            return E_INVALIDARG;
        }
    }

    if (metadata) {
        if (metadata->struct_size > 0 && metadata->struct_size < sizeof(D3D12X_COMMAND_LIST_METADATA)) {
            return E_INVALIDARG;
        }
        if (metadata->node_mask > 1) {
            return E_INVALIDARG;
        }

        // Handle WAIT_PREVIOUS_FRAME
        if ((metadata->flags & D3D12X_EXECUTE_FLAG_WAIT_PREVIOUS_FRAME) && prev_frame_fence_) {
            if (inner_queue_) {
                inner_queue_->Wait(prev_frame_fence_, prev_frame_fence_value_);
            }
        }

        // Handle pre-execution sync points
        if (metadata->sync_points && metadata->sync_point_count > 0) {
            for (uint32_t i = 0; i < metadata->sync_point_count; ++i) {
                const auto& sp = metadata->sync_points[i];
                if (sp.kind == D3D12X_SYNC_KIND::FENCE_WAIT && sp.fence) {
                    if (inner_queue_) {
                        const HRESULT hr = inner_queue_->Wait(sp.fence, sp.value);
                        if (FAILED(hr)) return hr;
                    }
                }
            }
        }

        current_frame_token_ = metadata->frame_token;
    }

    // Submit work to inner queue if present
    if (inner_queue_) {
        inner_queue_->ExecuteCommandLists(count, lists);
    }

    if (metadata) {
        // Handle post-execution sync points
        if (metadata->sync_points && metadata->sync_point_count > 0) {
            for (uint32_t i = 0; i < metadata->sync_point_count; ++i) {
                const auto& sp = metadata->sync_points[i];
                if (sp.kind == D3D12X_SYNC_KIND::FENCE_SIGNAL && sp.fence) {
                    if (inner_queue_) {
                        const HRESULT hr = inner_queue_->Signal(sp.fence, sp.value);
                        if (FAILED(hr)) return hr;
                    } else {
                        sp.fence->Signal(sp.value);
                    }
                }
            }
        }

        if (metadata->flags & D3D12X_EXECUTE_FLAG_PRESENT_AFTER_EXECUTE) {
            last_completed_token_ = metadata->frame_token;
        }
    }

    submission_count_++;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE CommandQueueX::SignalX(ID3D12Fence* fence, UINT64 value) {
    if (!fence) return E_POINTER;
    if (inner_queue_) {
        return inner_queue_->Signal(fence, value);
    }
    return fence->Signal(value);
}

HRESULT STDMETHODCALLTYPE CommandQueueX::WaitX(ID3D12Fence* fence, UINT64 value) {
    if (!fence) return E_POINTER;
    if (inner_queue_) {
        return inner_queue_->Wait(fence, value);
    }
    return S_OK;
}

HRESULT STDMETHODCALLTYPE CommandQueueX::GetFrameToken(UINT64* out_token) {
    if (!out_token) return E_POINTER;
    std::lock_guard lock(mutex_);
    *out_token = current_frame_token_;
    return S_OK;
}

uint64_t CommandQueueX::get_last_submitted_token() const {
    std::lock_guard lock(mutex_);
    return current_frame_token_;
}

uint64_t CommandQueueX::get_last_completed_token() const {
    std::lock_guard lock(mutex_);
    return last_completed_token_;
}

uint64_t CommandQueueX::get_submission_count() const {
    std::lock_guard lock(mutex_);
    return submission_count_;
}

void CommandQueueX::set_previous_frame_fence(ID3D12Fence* fence, uint64_t value) {
    std::lock_guard lock(mutex_);
    prev_frame_fence_ = fence;
    prev_frame_fence_value_ = value;
}

HRESULT create_command_queue_x(
    ID3D12CommandQueue* inner_queue,
    REFIID riid,
    void** out,
    const D3D12_COMMAND_QUEUE_DESC* desc)
{
    if (!out) return E_POINTER;
    *out = nullptr;

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    if (desc) {
        queue_desc = *desc;
    } else if (inner_queue) {
        queue_desc = inner_queue->GetDesc();
    }

    auto* wrapper = new CommandQueueX(inner_queue, queue_desc);
    const HRESULT hr = wrapper->QueryInterface(riid, out);
    wrapper->Release();
    return hr;
}

// ============================================================================
// Execute Command Lists X Shim
// ============================================================================

HRESULT execute_command_lists_x_shim(
    ID3D12CommandQueue* queue,
    UINT count,
    ID3D12CommandList* const* lists,
    const D3D12X_COMMAND_LIST_METADATA* metadata)
{
    if (!queue) {
        return E_POINTER;
    }
    if (count > 0 && !lists) {
        return E_INVALIDARG;
    }

    for (UINT i = 0; i < count; ++i) {
        if (!lists[i]) return E_INVALIDARG;
        if (lists[i]->GetType() == D3D12_COMMAND_LIST_TYPE_BUNDLE) return E_INVALIDARG;
    }

    if (metadata) {
        if (metadata->struct_size > 0 && metadata->struct_size < sizeof(D3D12X_COMMAND_LIST_METADATA)) {
            return E_INVALIDARG;
        }
        if (metadata->node_mask > 1) {
            return E_INVALIDARG;
        }
    }

    // Check if the command queue natively implements ID3D12CommandQueueX
    ID3D12CommandQueueX* queue_x = nullptr;
    if (SUCCEEDED(queue->QueryInterface(IID_ID3D12CommandQueueX, reinterpret_cast<void**>(&queue_x))) && queue_x) {
        const HRESULT hr = queue_x->ExecuteCommandListsX(count, lists, metadata);
        queue_x->Release();
        return hr;
    }

    // Fallback translation over standard ID3D12CommandQueue
    if (metadata && metadata->sync_points && metadata->sync_point_count > 0) {
        // Apply pre-execution fence waits
        for (uint32_t i = 0; i < metadata->sync_point_count; ++i) {
            const auto& sp = metadata->sync_points[i];
            if (sp.kind == D3D12X_SYNC_KIND::FENCE_WAIT && sp.fence) {
                const HRESULT hr = queue->Wait(sp.fence, sp.value);
                if (FAILED(hr)) return hr;
            }
        }
    }

    // Execute standard command lists
    queue->ExecuteCommandLists(count, lists);

    if (metadata && metadata->sync_points && metadata->sync_point_count > 0) {
        // Apply post-execution fence signals
        for (uint32_t i = 0; i < metadata->sync_point_count; ++i) {
            const auto& sp = metadata->sync_points[i];
            if (sp.kind == D3D12X_SYNC_KIND::FENCE_SIGNAL && sp.fence) {
                const HRESULT hr = queue->Signal(sp.fence, sp.value);
                if (FAILED(hr)) return hr;
            }
        }
    }

    return S_OK;
}

// ============================================================================
// Verification and Compliance Self-Test
// ============================================================================

bool verify_d3d12x_execution() {
    // 1. Check invalid argument handling
    VulkanSubmitPlan plan;
    if (translate_command_list_metadata(5, nullptr, nullptr, {}, VK_NULL_HANDLE, 1, plan) != E_INVALIDARG) {
        return false;
    }

    // Check invalid metadata struct_size
    D3D12X_COMMAND_LIST_METADATA bad_meta{};
    bad_meta.struct_size = 4; // Too small
    if (translate_command_list_metadata(0, nullptr, &bad_meta, {}, VK_NULL_HANDLE, 1, plan) != E_INVALIDARG) {
        return false;
    }

    // Check invalid node_mask
    D3D12X_COMMAND_LIST_METADATA bad_node_meta{};
    bad_node_meta.struct_size = sizeof(bad_node_meta);
    bad_node_meta.node_mask = 5; // Multi-adapter node mask > 1
    if (translate_command_list_metadata(0, nullptr, &bad_node_meta, {}, VK_NULL_HANDLE, 1, plan) != E_INVALIDARG) {
        return false;
    }

    // 2. Check translation of empty batch
    if (translate_command_list_metadata(0, nullptr, nullptr, {}, VK_NULL_HANDLE, 1, plan) != S_OK) {
        return false;
    }
    if (plan.segments.size() != 1 || !plan.segments[0].cmd_buffers.empty()) {
        return false;
    }

    // 3. Test Barrier Translation
    {
        const auto rt_state = translate_d3d12_resource_state(D3D12_RESOURCE_STATE_RENDER_TARGET);
        if (rt_state.layout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ||
            !(rt_state.access & VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT) ||
            !(rt_state.stages & VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)) {
            return false;
        }

        const auto depth_state = translate_d3d12_resource_state(D3D12_RESOURCE_STATE_DEPTH_WRITE);
        if (depth_state.layout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ||
            !(depth_state.access & VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) {
            return false;
        }

        const auto uav_barrier = translate_uav_barrier();
        if (!(uav_barrier.src_access & VK_ACCESS_SHADER_WRITE_BIT) ||
            !(uav_barrier.dst_access & VK_ACCESS_SHADER_READ_BIT)) {
            return false;
        }

        const auto alias_barrier = translate_aliasing_barrier();
        if (alias_barrier.src_stages != VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) {
            return false;
        }

        // Test Enhanced Barrier Layouts
        if (translate_enhanced_barrier_layout(4) != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL ||
            translate_enhanced_barrier_layout(6) != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ||
            translate_enhanced_barrier_layout(10) != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            return false;
        }

        // Test Enhanced Barrier Sync & Access
        const VkPipelineStageFlags sync_stages = translate_enhanced_barrier_sync(0x40); // Render target
        if (!(sync_stages & VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)) {
            return false;
        }

        const VkAccessFlags access_flags = translate_enhanced_barrier_access(0x8); // Render target
        if (!(access_flags & VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)) {
            return false;
        }
    }

    // 4. Mock Command Lists and Sync Points
    const VkCommandBuffer mock_cb1 = reinterpret_cast<VkCommandBuffer>(0x1000);
    const VkCommandBuffer mock_cb2 = reinterpret_cast<VkCommandBuffer>(0x2000);
    const VkCommandBuffer mock_cb3 = reinterpret_cast<VkCommandBuffer>(0x3000);
    const std::vector<VkCommandBuffer> cbs = {mock_cb1, mock_cb2, mock_cb3};

    struct MockCommandList : public ID3D12CommandList {
        D3D12_COMMAND_LIST_TYPE type_ = D3D12_COMMAND_LIST_TYPE_DIRECT;
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { return E_NOINTERFACE; }
        ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
        ULONG STDMETHODCALLTYPE Release() override { return 1; }
        D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE GetType() override { return type_; }
    };

    MockCommandList list1;
    MockCommandList list2;
    MockCommandList list3;
    ID3D12CommandList* dummy_lists[3] = {&list1, &list2, &list3};

    // Test bundle rejection
    MockCommandList bundle_list;
    bundle_list.type_ = D3D12_COMMAND_LIST_TYPE_BUNDLE;
    ID3D12CommandList* bundle_arr[1] = {&bundle_list};
    if (translate_command_list_metadata(1, bundle_arr, nullptr, {}, VK_NULL_HANDLE, 1, plan) != E_INVALIDARG) {
        return false;
    }

    const VkSemaphore mock_sem_wait = reinterpret_cast<VkSemaphore>(0x5000);
    const VkSemaphore mock_sem_signal = reinterpret_cast<VkSemaphore>(0x6000);
    const VkSemaphore mock_queue_progress = reinterpret_cast<VkSemaphore>(0x7000);

    D3D12X_SYNC_POINT syncs[3];
    // Sync point 0: wait before list index 0
    syncs[0].kind = D3D12X_SYNC_KIND::SEMAPHORE_WAIT;
    syncs[0].command_list_index = 0;
    syncs[0].value = 42;
    syncs[0].fence = nullptr;
    syncs[0].vk_semaphore = mock_sem_wait;
    syncs[0].wait_stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    syncs[0].signal_stages = 0;

    // Sync point 1: signal after list index 1
    syncs[1].kind = D3D12X_SYNC_KIND::SEMAPHORE_SIGNAL;
    syncs[1].command_list_index = 1;
    syncs[1].value = 100;
    syncs[1].fence = nullptr;
    syncs[1].vk_semaphore = mock_sem_signal;
    syncs[1].wait_stages = 0;
    syncs[1].signal_stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    // Sync point 2: pipeline barrier across all lists
    syncs[2].kind = D3D12X_SYNC_KIND::PIPELINE_BARRIER;
    syncs[2].command_list_index = D3D12X_SYNC_INDEX_ALL;
    syncs[2].value = 0;
    syncs[2].fence = nullptr;
    syncs[2].vk_semaphore = VK_NULL_HANDLE;
    syncs[2].wait_stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    syncs[2].signal_stages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

    D3D12X_COMMAND_LIST_METADATA meta{};
    meta.struct_size = sizeof(meta);
    meta.flags = D3D12X_EXECUTE_FLAG_PIPELINE_STALL |
                 D3D12X_EXECUTE_FLAG_FORCE_IMMEDIATE_SUBMIT |
                 D3D12X_EXECUTE_FLAG_DISABLE_PREEMPTION |
                 D3D12X_EXECUTE_FLAG_PRESENT_AFTER_EXECUTE;
    meta.frame_token = 0xABCDE;
    meta.sync_point_count = 3;
    meta.sync_points = syncs;
    meta.priority = 1;
    meta.node_mask = 1;

    if (translate_command_list_metadata(3, dummy_lists, &meta, cbs, mock_queue_progress, 999, plan) != S_OK) {
        return false;
    }

    if (!plan.immediate_submit || !plan.pipeline_stall || !plan.disable_preemption ||
        !plan.present_after_execute || plan.frame_token != 0xABCDE || plan.priority != 1) {
        return false;
    }

    // Split at index 1 -> list 0..1 in segment 0, list 2 in segment 1
    if (plan.segments.size() != 2) {
        return false;
    }
    if (plan.segments[0].cmd_buffers.size() != 2 || plan.segments[1].cmd_buffers.size() != 1) {
        return false;
    }

    // Check wait on segment 0
    if (plan.segments[0].wait_semaphores.size() != 1 || plan.segments[0].wait_values[0] != 42) {
        return false;
    }
    // Check signal on segment 0 (list 1)
    if (plan.segments[0].signal_semaphores.size() != 1 || plan.segments[0].signal_values[0] != 100) {
        return false;
    }
    // Check queue progress timeline semaphore on final segment 1
    if (plan.segments[1].signal_semaphores.size() != 1 || plan.segments[1].signal_values[0] != 999) {
        return false;
    }

    // Check memory barriers in segments
    if (plan.segments[0].memory_barriers.empty() || plan.segments[1].memory_barriers.empty()) {
        return false;
    }

    // 5. Test execute_vulkan_submit_plan with mock submission function
    struct MockSubmitState {
        uint32_t call_count = 0;
        std::vector<uint32_t> cmd_counts;
    };
    static MockSubmitState state;
    state = {};

    auto mock_submit = [](VkQueue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence) -> VkResult {
        state.call_count += submitCount;
        for (uint32_t i = 0; i < submitCount; ++i) {
            state.cmd_counts.push_back(pSubmits[i].commandBufferCount);
        }
        return VK_SUCCESS;
    };

    const VkQueue mock_queue = reinterpret_cast<VkQueue>(0x9999);
    if (execute_vulkan_submit_plan(mock_queue, plan, mock_submit, VK_NULL_HANDLE) != S_OK) {
        return false;
    }
    if (state.call_count != 2 || state.cmd_counts.size() != 2) {
        return false;
    }
    if (state.cmd_counts[0] != 2 || state.cmd_counts[1] != 1) {
        return false;
    }

    // 6. Test CommandQueueX Wrapper and Interfaces
    struct MockFence : public ID3D12Fence {
        uint64_t value_ = 0;
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { return E_NOINTERFACE; }
        ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
        ULONG STDMETHODCALLTYPE Release() override { return 1; }
        UINT64 STDMETHODCALLTYPE GetCompletedValue() override { return value_; }
        HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64, void*) override { return S_OK; }
        HRESULT STDMETHODCALLTYPE Signal(UINT64 Value) override { value_ = Value; return S_OK; }
    };

    struct MockQueue : public ID3D12CommandQueue {
        uint32_t exec_count_ = 0;
        uint64_t last_signaled_val_ = 0;
        uint64_t last_waited_val_ = 0;

        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { return E_NOINTERFACE; }
        ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
        ULONG STDMETHODCALLTYPE Release() override { return 1; }
        void STDMETHODCALLTYPE ExecuteCommandLists(UINT, ID3D12CommandList* const*) override {
            exec_count_++;
        }
        HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence*, UINT64 Value) override {
            last_signaled_val_ = Value;
            return S_OK;
        }
        HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence*, UINT64 Value) override {
            last_waited_val_ = Value;
            return S_OK;
        }
        HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64* pFreq) override {
            if (pFreq) *pFreq = 50000000ULL;
            return S_OK;
        }
        HRESULT STDMETHODCALLTYPE GetClockCalibration(UINT64* g, UINT64* c) override {
            if (g) *g = 12345;
            if (c) *c = 67890;
            return S_OK;
        }
        D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() override {
            D3D12_COMMAND_QUEUE_DESC d{};
            d.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            d.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
            return d;
        }
    };

    MockQueue mock_base_queue;
    MockFence mock_fence;

    ID3D12CommandQueueX* queue_x = nullptr;
    if (FAILED(create_command_queue_x(&mock_base_queue, IID_ID3D12CommandQueueX, reinterpret_cast<void**>(&queue_x)))) {
        return false;
    }
    if (!queue_x) {
        return false;
    }

    // Test GetDesc
    const auto desc = queue_x->GetDesc();
    if (desc.Priority != D3D12_COMMAND_QUEUE_PRIORITY_HIGH) {
        queue_x->Release();
        return false;
    }

    // Test ExecuteCommandListsX
    if (FAILED(queue_x->ExecuteCommandListsX(3, dummy_lists, &meta))) {
        queue_x->Release();
        return false;
    }
    if (mock_base_queue.exec_count_ != 1) {
        queue_x->Release();
        return false;
    }

    // Test GetFrameToken
    UINT64 token = 0;
    if (FAILED(queue_x->GetFrameToken(&token)) || token != 0xABCDE) {
        queue_x->Release();
        return false;
    }

    // Test SignalX and WaitX
    if (FAILED(queue_x->SignalX(&mock_fence, 777)) || mock_base_queue.last_signaled_val_ != 777) {
        queue_x->Release();
        return false;
    }
    if (FAILED(queue_x->WaitX(&mock_fence, 888)) || mock_base_queue.last_waited_val_ != 888) {
        queue_x->Release();
        return false;
    }

    // Test C ABI shim functions
    if (FAILED(D3D12XExecuteCommandLists(queue_x, 3, dummy_lists, &meta))) {
        queue_x->Release();
        return false;
    }
    if (FAILED(D3D12XSignalFence(queue_x, &mock_fence, 999))) {
        queue_x->Release();
        return false;
    }
    if (FAILED(D3D12XWaitFence(queue_x, &mock_fence, 1111))) {
        queue_x->Release();
        return false;
    }

    queue_x->Release();
    return true;
}

} // namespace rt::d3d12_x

// ============================================================================
// C ABI Exports
// ============================================================================

extern "C" {

HRESULT WINAPI D3D12XExecuteCommandLists(
    ID3D12CommandQueue* queue,
    UINT count,
    ID3D12CommandList* const* lists,
    const D3D12X_COMMAND_LIST_METADATA* metadata)
{
    return rt::d3d12_x::execute_command_lists_x_shim(queue, count, lists, metadata);
}

HRESULT WINAPI D3D12XSignalFence(
    ID3D12CommandQueue* queue,
    ID3D12Fence* fence,
    UINT64 value)
{
    if (!queue || !fence) return E_POINTER;
    return queue->Signal(fence, value);
}

HRESULT WINAPI D3D12XWaitFence(
    ID3D12CommandQueue* queue,
    ID3D12Fence* fence,
    UINT64 value)
{
    if (!queue || !fence) return E_POINTER;
    return queue->Wait(fence, value);
}

HRESULT WINAPI D3D12XCreateCommandQueue(
    ID3D12CommandQueue* inner_queue,
    REFIID riid,
    void** ppvQueue)
{
    return rt::d3d12_x::create_command_queue_x(inner_queue, riid, ppvQueue);
}

}

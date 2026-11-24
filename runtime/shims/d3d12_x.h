// Console D3D12.X ExecuteCommandListsX and command submission shims.
// Translates Xbox Series D3D12.X command list metadata and synchronization points
// to Vulkan command buffer submissions (vkQueueSubmit) and fence signaling.

#ifndef LEONIDA_SHIMS_D3D12_X_H
#define LEONIDA_SHIMS_D3D12_X_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <span>
#include <memory>
#include <atomic>
#include <mutex>
#include <chrono>
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <windows.h>
#include <d3d12.h>
#else
// Portable COM & D3D12 definitions for non-Windows platforms
#ifndef WINAPI
#define WINAPI
#endif
#ifndef STDMETHODCALLTYPE
#define STDMETHODCALLTYPE
#endif
using UINT = uint32_t;
using UINT64 = uint64_t;
using INT = int32_t;
using HRESULT = int32_t;
using ULONG = uint32_t;

#ifndef S_OK
#define S_OK ((HRESULT)0L)
#endif
#ifndef S_FALSE
#define S_FALSE ((HRESULT)1L)
#endif
#ifndef E_INVALIDARG
#define E_INVALIDARG ((HRESULT)0x80070057L)
#endif
#ifndef E_POINTER
#define E_POINTER ((HRESULT)0x80004003L)
#endif
#ifndef E_FAIL
#define E_FAIL ((HRESULT)0x80004005L)
#endif
#ifndef E_NOTIMPL
#define E_NOTIMPL ((HRESULT)0x80004001L)
#endif
#ifndef E_NOINTERFACE
#define E_NOINTERFACE ((HRESULT)0x80004002L)
#endif
#ifndef DXGI_ERROR_DEVICE_REMOVED
#define DXGI_ERROR_DEVICE_REMOVED ((HRESULT)0x887A0005L)
#endif
#ifndef SUCCEEDED
#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)
#endif
#ifndef FAILED
#define FAILED(hr) (((HRESULT)(hr)) < 0)
#endif

struct GUID {
    uint32_t Data1;
    uint16_t Data2;
    uint16_t Data3;
    uint8_t  Data4[8];

    constexpr bool operator==(const GUID& o) const {
        return Data1 == o.Data1 && Data2 == o.Data2 && Data3 == o.Data3 &&
               Data4[0] == o.Data4[0] && Data4[1] == o.Data4[1] &&
               Data4[2] == o.Data4[2] && Data4[3] == o.Data4[3] &&
               Data4[4] == o.Data4[4] && Data4[5] == o.Data4[5] &&
               Data4[6] == o.Data4[6] && Data4[7] == o.Data4[7];
    }
};
using REFIID = const GUID&;

struct IUnknown {
    virtual HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) = 0;
    virtual ULONG STDMETHODCALLTYPE AddRef() = 0;
    virtual ULONG STDMETHODCALLTYPE Release() = 0;
    virtual ~IUnknown() = default;
};

struct ID3D12Object : public IUnknown {};
struct ID3D12DeviceChild : public ID3D12Object {};
struct ID3D12Pageable : public ID3D12DeviceChild {};

enum D3D12_COMMAND_LIST_TYPE {
    D3D12_COMMAND_LIST_TYPE_DIRECT = 0,
    D3D12_COMMAND_LIST_TYPE_BUNDLE = 1,
    D3D12_COMMAND_LIST_TYPE_COMPUTE = 2,
    D3D12_COMMAND_LIST_TYPE_COPY = 3,
    D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE = 4,
    D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS = 5,
    D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE = 6
};

struct ID3D12CommandList : public ID3D12DeviceChild {
    virtual D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE GetType() = 0;
};

struct ID3D12GraphicsCommandList : public ID3D12CommandList {
    virtual HRESULT STDMETHODCALLTYPE Close() = 0;
    virtual HRESULT STDMETHODCALLTYPE Reset(void* pAllocator, void* pInitialState) = 0;
};

struct ID3D12Fence : public ID3D12Pageable {
    virtual UINT64 STDMETHODCALLTYPE GetCompletedValue() = 0;
    virtual HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64 Value, void* hEvent) = 0;
    virtual HRESULT STDMETHODCALLTYPE Signal(UINT64 Value) = 0;
};

enum D3D12_COMMAND_QUEUE_FLAGS {
    D3D12_COMMAND_QUEUE_FLAG_NONE = 0,
    D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT = 0x1
};

enum D3D12_COMMAND_QUEUE_PRIORITY {
    D3D12_COMMAND_QUEUE_PRIORITY_NORMAL = 0,
    D3D12_COMMAND_QUEUE_PRIORITY_HIGH = 100,
    D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME = 10000
};

struct D3D12_COMMAND_QUEUE_DESC {
    D3D12_COMMAND_LIST_TYPE Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    INT Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    D3D12_COMMAND_QUEUE_FLAGS Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    UINT NodeMask = 0;
};

struct ID3D12CommandQueue : public ID3D12Pageable {
    virtual void STDMETHODCALLTYPE ExecuteCommandLists(UINT NumCommandLists, ID3D12CommandList* const* ppCommandLists) = 0;
    virtual HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence* pFence, UINT64 Value) = 0;
    virtual HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence* pFence, UINT64 Value) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64* pFrequency) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetClockCalibration(UINT64* pGpuTimestamp, UINT64* pCpuTimestamp) = 0;
    virtual D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() = 0;
};

// D3D12 Resource States
enum D3D12_RESOURCE_STATES : uint32_t {
    D3D12_RESOURCE_STATE_COMMON = 0,
    D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER = 0x1,
    D3D12_RESOURCE_STATE_INDEX_BUFFER = 0x2,
    D3D12_RESOURCE_STATE_RENDER_TARGET = 0x4,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS = 0x8,
    D3D12_RESOURCE_STATE_DEPTH_WRITE = 0x10,
    D3D12_RESOURCE_STATE_DEPTH_READ = 0x20,
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE = 0x40,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE = 0x80,
    D3D12_RESOURCE_STATE_STREAM_OUT = 0x100,
    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT = 0x200,
    D3D12_RESOURCE_STATE_COPY_DEST = 0x400,
    D3D12_RESOURCE_STATE_COPY_SOURCE = 0x800,
    D3D12_RESOURCE_STATE_RESOLVE_DEST = 0x1000,
    D3D12_RESOURCE_STATE_RESOLVE_SOURCE = 0x2000,
    D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE = 0x400000,
    D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE = 0x1000000,
    D3D12_RESOURCE_STATE_GENERIC_READ = (0x1 | 0x2 | 0x40 | 0x80 | 0x200 | 0x800),
    D3D12_RESOURCE_STATE_PRESENT = 0,
    D3D12_RESOURCE_STATE_PREDICATION = 0x200,
    D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE = (0x40 | 0x80)
};

// D3D12 Barrier Types
enum D3D12_RESOURCE_BARRIER_TYPE {
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION = 0,
    D3D12_RESOURCE_BARRIER_TYPE_ALIASING = 1,
    D3D12_RESOURCE_BARRIER_TYPE_UAV = 2
};

enum D3D12_RESOURCE_BARRIER_FLAGS {
    D3D12_RESOURCE_BARRIER_FLAG_NONE = 0x0,
    D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY = 0x1,
    D3D12_RESOURCE_BARRIER_FLAG_END_ONLY = 0x2
};

struct ID3D12Resource : public ID3D12Pageable {};

struct D3D12_RESOURCE_TRANSITION_BARRIER {
    ID3D12Resource* pResource;
    UINT Subresource;
    D3D12_RESOURCE_STATES StateBefore;
    D3D12_RESOURCE_STATES StateAfter;
};

struct D3D12_RESOURCE_ALIASING_BARRIER {
    ID3D12Resource* pResourceBefore;
    ID3D12Resource* pResourceAfter;
};

struct D3D12_RESOURCE_UAV_BARRIER {
    ID3D12Resource* pResource;
};

struct D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER_TYPE Type;
    D3D12_RESOURCE_BARRIER_FLAGS Flags;
    union {
        D3D12_RESOURCE_TRANSITION_BARRIER Transition;
        D3D12_RESOURCE_ALIASING_BARRIER Aliasing;
        D3D12_RESOURCE_UAV_BARRIER UAV;
    };
};

#endif // !_WIN32

// Standard COM IIDs
inline constexpr GUID IID_IUnknown_X = {
    0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}
};
inline constexpr GUID IID_ID3D12CommandQueue_X = {
    0x0ec87025, 0x04bb, 0x4fc4, {0xaa, 0x44, 0x4d, 0x6e, 0x76, 0x6e, 0x5a, 0x01}
};

// ============================================================================
// Xbox Series D3D12.X Console Extensions
// ============================================================================

enum D3D12X_EXECUTE_FLAGS : uint32_t {
    D3D12X_EXECUTE_FLAG_NONE                    = 0x0,
    D3D12X_EXECUTE_FLAG_PIPELINE_STALL          = 0x1,
    D3D12X_EXECUTE_FLAG_SYNC_BARRIER            = 0x2,
    D3D12X_EXECUTE_FLAG_DISABLE_PREEMPTION      = 0x4,
    D3D12X_EXECUTE_FLAG_FORCE_IMMEDIATE_SUBMIT  = 0x8,
    D3D12X_EXECUTE_FLAG_PRESENT_AFTER_EXECUTE   = 0x10,
    D3D12X_EXECUTE_FLAG_WAIT_PREVIOUS_FRAME     = 0x20,
};

enum class D3D12X_SYNC_KIND : uint32_t {
    FENCE_SIGNAL        = 0,
    FENCE_WAIT          = 1,
    SEMAPHORE_SIGNAL    = 2,
    SEMAPHORE_WAIT      = 3,
    PIPELINE_BARRIER    = 4,
};

inline constexpr uint32_t D3D12X_SYNC_INDEX_ALL = 0xFFFFFFFFu;

struct D3D12X_SYNC_POINT {
    D3D12X_SYNC_KIND kind;
    uint32_t command_list_index;        // 0-based index or D3D12X_SYNC_INDEX_ALL
    uint64_t value;                     // Timeline fence/semaphore value
    ID3D12Fence* fence;                 // D3D12 fence interface pointer (optional)
    VkSemaphore vk_semaphore;           // Vulkan timeline semaphore (optional)
    VkPipelineStageFlags wait_stages;   // Pipeline stages to block before executing
    VkPipelineStageFlags signal_stages; // Pipeline stages signaling on completion
};

struct D3D12X_COMMAND_LIST_METADATA {
    uint32_t struct_size;                 // sizeof(D3D12X_COMMAND_LIST_METADATA)
    uint32_t flags;                       // Bitwise combination of D3D12X_EXECUTE_FLAGS
    uint64_t frame_token;                 // GDKX Frame pipeline token
    uint32_t sync_point_count;            // Number of sync points
    const D3D12X_SYNC_POINT* sync_points; // Array of sync point descriptors
    uint32_t priority;                    // Priority (0 = normal, 1 = high, 2 = realtime)
    uint32_t node_mask;                   // Multi-adapter node mask (1 for single GPU)
};

// Interface identifier for ID3D12CommandQueueX
// {865F2783-63A5-49DE-8378-CD7E80A03C12}
inline constexpr GUID IID_ID3D12CommandQueueX = {
    0x865f2783, 0x63a5, 0x49de, {0x83, 0x78, 0xcd, 0x7e, 0x80, 0xa0, 0x3c, 0x12}
};

struct ID3D12CommandQueueX : public ID3D12CommandQueue {
    virtual HRESULT STDMETHODCALLTYPE ExecuteCommandListsX(
        UINT count,
        ID3D12CommandList* const* lists,
        const D3D12X_COMMAND_LIST_METADATA* metadata = nullptr) = 0;

    virtual HRESULT STDMETHODCALLTYPE SignalX(ID3D12Fence* fence, UINT64 value) = 0;
    virtual HRESULT STDMETHODCALLTYPE WaitX(ID3D12Fence* fence, UINT64 value) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetFrameToken(UINT64* out_token) = 0;
};

// ============================================================================
// Vulkan Translation and Submission Layer
// ============================================================================

namespace rt::d3d12_x {

struct VulkanMemoryBarrierDesc {
    VkPipelineStageFlags src_stages = 0;
    VkPipelineStageFlags dst_stages = 0;
    VkAccessFlags src_access = 0;
    VkAccessFlags dst_access = 0;
};

struct VulkanImageBarrierDesc {
    VkPipelineStageFlags src_stages = 0;
    VkPipelineStageFlags dst_stages = 0;
    VkAccessFlags src_access = 0;
    VkAccessFlags dst_access = 0;
    VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    uint32_t subresource = 0;
};

struct VulkanSubmitSegment {
    VkSubmitInfo submit_info{};
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    std::vector<VkCommandBuffer> cmd_buffers;
    std::vector<VkSemaphore> wait_semaphores;
    std::vector<uint64_t> wait_values;
    std::vector<VkPipelineStageFlags> wait_stages;
    std::vector<VkSemaphore> signal_semaphores;
    std::vector<uint64_t> signal_values;
    std::vector<ID3D12Fence*> signaled_fences;
    std::vector<uint64_t> signaled_fence_values;
    std::vector<VulkanMemoryBarrierDesc> memory_barriers;
    std::vector<VulkanImageBarrierDesc> image_barriers;
};

struct VulkanSubmitPlan {
    std::vector<VulkanSubmitSegment> segments;
    bool immediate_submit = false;
    bool pipeline_stall = false;
    bool disable_preemption = false;
    bool present_after_execute = false;
    bool wait_previous_frame = false;
    uint32_t priority = 0;
    uint64_t frame_token = 0;
};

struct SubmissionContext {
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    PFN_vkQueueSubmit queue_submit = nullptr;
    PFN_vkGetSemaphoreCounterValue get_semaphore_counter = nullptr;
    PFN_vkWaitSemaphores wait_semaphores = nullptr;
    VkSemaphore queue_progress_semaphore = VK_NULL_HANDLE;
    uint64_t* current_timeline_value = nullptr;
};

// Barrier and Resource State translation
struct StateConversionResult {
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags access = 0;
    VkPipelineStageFlags stages = 0;
};

StateConversionResult translate_d3d12_resource_state(uint32_t d3d12_state, bool is_depth_stencil = false);

VulkanImageBarrierDesc translate_transition_barrier(
    uint32_t state_before,
    uint32_t state_after,
    VkImage image = VK_NULL_HANDLE,
    uint32_t subresource = 0,
    bool is_depth = false);

VulkanMemoryBarrierDesc translate_uav_barrier();
VulkanMemoryBarrierDesc translate_aliasing_barrier();

// Enhanced barrier layout translation
VkImageLayout translate_enhanced_barrier_layout(uint32_t layout);
VkPipelineStageFlags translate_enhanced_barrier_sync(uint32_t sync_flags);
VkAccessFlags translate_enhanced_barrier_access(uint32_t access_flags);

// Translates console command lists + metadata into structured Vulkan submit segments
HRESULT translate_command_list_metadata(
    UINT count,
    ID3D12CommandList* const* lists,
    const D3D12X_COMMAND_LIST_METADATA* metadata,
    const std::vector<VkCommandBuffer>& vk_cmd_buffers,
    VkSemaphore queue_progress_semaphore,
    uint64_t next_ticket,
    VulkanSubmitPlan& out_plan);

// Executes the planned submission on the Vulkan hardware queue
HRESULT execute_vulkan_submit_plan(
    VkQueue queue,
    VulkanSubmitPlan& plan,
    PFN_vkQueueSubmit pfn_queue_submit,
    VkFence submit_fence = VK_NULL_HANDLE);

// Executes command lists on a D3D12 command queue shim with console metadata
HRESULT execute_command_lists_x_shim(
    ID3D12CommandQueue* queue,
    UINT count,
    ID3D12CommandList* const* lists,
    const D3D12X_COMMAND_LIST_METADATA* metadata);

// CommandQueueX implementation wrapping an existing command queue
class CommandQueueX : public ID3D12CommandQueueX {
public:
    explicit CommandQueueX(ID3D12CommandQueue* inner_queue, D3D12_COMMAND_QUEUE_DESC desc = {});
    virtual ~CommandQueueX();

    // IUnknown
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;
    ULONG STDMETHODCALLTYPE AddRef() override;
    ULONG STDMETHODCALLTYPE Release() override;

    // ID3D12CommandQueue
    void STDMETHODCALLTYPE ExecuteCommandLists(UINT NumCommandLists, ID3D12CommandList* const* ppCommandLists) override;
    HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence* pFence, UINT64 Value) override;
    HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence* pFence, UINT64 Value) override;
    HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64* pFrequency) override;
    HRESULT STDMETHODCALLTYPE GetClockCalibration(UINT64* pGpuTimestamp, UINT64* pCpuTimestamp) override;
    D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() override;

    // ID3D12CommandQueueX
    HRESULT STDMETHODCALLTYPE ExecuteCommandListsX(
        UINT count,
        ID3D12CommandList* const* lists,
        const D3D12X_COMMAND_LIST_METADATA* metadata = nullptr) override;
    HRESULT STDMETHODCALLTYPE SignalX(ID3D12Fence* fence, UINT64 value) override;
    HRESULT STDMETHODCALLTYPE WaitX(ID3D12Fence* fence, UINT64 value) override;
    HRESULT STDMETHODCALLTYPE GetFrameToken(UINT64* out_token) override;

    // Queue state queries
    uint64_t get_last_submitted_token() const;
    uint64_t get_last_completed_token() const;
    uint64_t get_submission_count() const;
    void set_previous_frame_fence(ID3D12Fence* fence, uint64_t value);

private:
    std::atomic<ULONG> ref_count_{1};
    ID3D12CommandQueue* inner_queue_ = nullptr;
    D3D12_COMMAND_QUEUE_DESC desc_{};
    mutable std::mutex mutex_;
    uint64_t current_frame_token_ = 0;
    uint64_t last_completed_token_ = 0;
    uint64_t submission_count_ = 0;
    ID3D12Fence* prev_frame_fence_ = nullptr;
    uint64_t prev_frame_fence_value_ = 0;
};

// Factory function to create an ID3D12CommandQueueX wrapper
HRESULT create_command_queue_x(
    ID3D12CommandQueue* inner_queue,
    REFIID riid,
    void** out,
    const D3D12_COMMAND_QUEUE_DESC* desc = nullptr);

// Internal regression and compliance self-test
bool verify_d3d12x_execution();

} // namespace rt::d3d12_x

// C ABI Exports matching console entry points
extern "C" {
HRESULT WINAPI D3D12XExecuteCommandLists(
    ID3D12CommandQueue* queue,
    UINT count,
    ID3D12CommandList* const* lists,
    const D3D12X_COMMAND_LIST_METADATA* metadata);

HRESULT WINAPI D3D12XSignalFence(
    ID3D12CommandQueue* queue,
    ID3D12Fence* fence,
    UINT64 value);

HRESULT WINAPI D3D12XWaitFence(
    ID3D12CommandQueue* queue,
    ID3D12Fence* fence,
    UINT64 value);

HRESULT WINAPI D3D12XCreateCommandQueue(
    ID3D12CommandQueue* inner_queue,
    REFIID riid,
    void** ppvQueue);
}

#endif // LEONIDA_SHIMS_D3D12_X_H

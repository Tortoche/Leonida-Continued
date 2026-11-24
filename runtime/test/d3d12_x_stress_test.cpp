// Empirical Stress Test Suite for D3D12.X Command Submission and Vulkan Presentation (Track 3, R4).
// Tests command list recording, multiple command lists in ExecuteCommandListsX, queue flushes,
// frame boundaries, swapchain lifecycle, resize stress, and headless presentation fallback.

#include "../shims/d3d12_x.h"
#include "../src/command_list.h"
#include "../src/vulkan_device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define TEST_CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "[FAILED] %s:%d: %s (%s)\n", __FILE__, __LINE__, msg, #cond); \
        return false; \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    std::printf("[RUNNING] %s ...\n", #fn); \
    if (fn()) { \
        std::printf("[PASSED]  %s\n", #fn); \
        ++g_passed; \
    } else { \
        std::printf("[FAILED]  %s\n", #fn); \
        ++g_failed; \
    } \
} while (0)

static int g_passed = 0;
static int g_failed = 0;

// ============================================================================
// MOCK VULKAN IMPLEMENTATION FOR ISOLATED, DETERMINISTIC HARDWARE-INDEPENDENT TESTING
// ============================================================================

struct MockVulkanState {
    std::atomic<uint32_t> create_swapchain_calls{0};
    std::atomic<uint32_t> destroy_swapchain_calls{0};
    std::atomic<uint32_t> get_images_calls{0};
    std::atomic<uint32_t> acquire_image_calls{0};
    std::atomic<uint32_t> queue_present_calls{0};

    std::atomic<uint32_t> create_image_calls{0};
    std::atomic<uint32_t> destroy_image_calls{0};
    std::atomic<uint32_t> alloc_mem_calls{0};
    std::atomic<uint32_t> free_mem_calls{0};
    std::atomic<uint32_t> bind_image_mem_calls{0};
    std::atomic<uint32_t> create_image_view_calls{0};
    std::atomic<uint32_t> destroy_image_view_calls{0};

    std::atomic<uint32_t> create_fence_calls{0};
    std::atomic<uint32_t> destroy_fence_calls{0};
    std::atomic<uint32_t> create_semaphore_calls{0};
    std::atomic<uint32_t> destroy_semaphore_calls{0};

    std::atomic<uint32_t> queue_submit_calls{0};
    std::atomic<uint32_t> wait_for_fences_calls{0};
    std::atomic<uint32_t> reset_fences_calls{0};
    std::atomic<uint32_t> device_wait_idle_calls{0};

    std::atomic<uint32_t> create_cmd_pool_calls{0};
    std::atomic<uint32_t> destroy_cmd_pool_calls{0};
    std::atomic<uint32_t> alloc_cmd_bufs_calls{0};
    std::atomic<uint32_t> create_buffer_calls{0};
    std::atomic<uint32_t> destroy_buffer_calls{0};

    std::atomic<uint32_t> last_submit_cmd_count{0};
    std::vector<VkSubmitInfo> recorded_submits;

    bool surface_supported = true;
    bool create_swapchain_fail = false;
    uint32_t mock_swapchain_image_count = 2;

    void reset() {
        create_swapchain_calls = 0;
        destroy_swapchain_calls = 0;
        get_images_calls = 0;
        acquire_image_calls = 0;
        queue_present_calls = 0;
        create_image_calls = 0;
        destroy_image_calls = 0;
        alloc_mem_calls = 0;
        free_mem_calls = 0;
        bind_image_mem_calls = 0;
        create_image_view_calls = 0;
        destroy_image_view_calls = 0;
        create_fence_calls = 0;
        destroy_fence_calls = 0;
        create_semaphore_calls = 0;
        destroy_semaphore_calls = 0;
        queue_submit_calls = 0;
        wait_for_fences_calls = 0;
        reset_fences_calls = 0;
        device_wait_idle_calls = 0;
        create_cmd_pool_calls = 0;
        destroy_cmd_pool_calls = 0;
        alloc_cmd_bufs_calls = 0;
        create_buffer_calls = 0;
        destroy_buffer_calls = 0;
        last_submit_cmd_count = 0;
        recorded_submits.clear();
        surface_supported = true;
        create_swapchain_fail = false;
        mock_swapchain_image_count = 2;
    }
};

struct TestMockCommandList : public ID3D12CommandList {
    D3D12_COMMAND_LIST_TYPE type_ = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { return E_NOINTERFACE; }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE GetType() override { return type_; }
};

static MockVulkanState g_vk_mock;

// Mock dispatch functions
extern "C" {

static VkResult VKAPI_PTR mock_vkGetPhysicalDeviceSurfaceSupportKHR(
    VkPhysicalDevice, uint32_t, VkSurfaceKHR, VkBool32* pSupported) {
    if (pSupported) *pSupported = g_vk_mock.surface_supported ? VK_TRUE : VK_FALSE;
    return VK_SUCCESS;
}

static VkResult VKAPI_PTR mock_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
    VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR* pCaps) {
    if (!pCaps) return VK_ERROR_INITIALIZATION_FAILED;
    std::memset(pCaps, 0, sizeof(*pCaps));
    pCaps->minImageCount = 2;
    pCaps->maxImageCount = 4;
    pCaps->currentExtent = {1920, 1080};
    pCaps->minImageExtent = {320, 240};
    pCaps->maxImageExtent = {3840, 2160};
    pCaps->maxImageArrayLayers = 1;
    pCaps->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    pCaps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    pCaps->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    pCaps->supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return VK_SUCCESS;
}

static VkResult VKAPI_PTR mock_vkGetPhysicalDeviceSurfaceFormatsKHR(
    VkPhysicalDevice, VkSurfaceKHR, uint32_t* pCount, VkSurfaceFormatKHR* pFormats) {
    if (!pCount) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pFormats) {
        *pCount = 2;
        return VK_SUCCESS;
    }
    pFormats[0] = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    pFormats[1] = {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    *pCount = 2;
    return VK_SUCCESS;
}

static VkResult VKAPI_PTR mock_vkGetPhysicalDeviceSurfacePresentModesKHR(
    VkPhysicalDevice, VkSurfaceKHR, uint32_t* pCount, VkPresentModeKHR* pModes) {
    if (!pCount) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pModes) {
        *pCount = 2;
        return VK_SUCCESS;
    }
    pModes[0] = VK_PRESENT_MODE_FIFO_KHR;
    pModes[1] = VK_PRESENT_MODE_IMMEDIATE_KHR;
    *pCount = 2;
    return VK_SUCCESS;
}

static void VKAPI_PTR mock_vkDestroySurfaceKHR(VkInstance, VkSurfaceKHR, const VkAllocationCallbacks*) {}

static void VKAPI_PTR mock_vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* pProps) {
    if (!pProps) return;
    std::memset(pProps, 0, sizeof(*pProps));
    pProps->memoryTypeCount = 2;
    pProps->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    pProps->memoryTypes[0].heapIndex = 0;
    pProps->memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    pProps->memoryTypes[1].heapIndex = 1;
    pProps->memoryHeapCount = 2;
    pProps->memoryHeaps[0].size = 1024ULL * 1024 * 1024 * 8; // 8GB
    pProps->memoryHeaps[1].size = 1024ULL * 1024 * 1024 * 8;
}

static VkResult VKAPI_PTR mock_vkCreateSwapchainKHR(
    VkDevice, const VkSwapchainCreateInfoKHR*, const VkAllocationCallbacks*, VkSwapchainKHR* pSwapchain) {
    ++g_vk_mock.create_swapchain_calls;
    if (g_vk_mock.create_swapchain_fail) return VK_ERROR_INITIALIZATION_FAILED;
    if (pSwapchain) *pSwapchain = reinterpret_cast<VkSwapchainKHR>(0xCAFE0001);
    return VK_SUCCESS;
}

static void VKAPI_PTR mock_vkDestroySwapchainKHR(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks*) {
    ++g_vk_mock.destroy_swapchain_calls;
}

static VkResult VKAPI_PTR mock_vkGetSwapchainImagesKHR(
    VkDevice, VkSwapchainKHR, uint32_t* pCount, VkImage* pImages) {
    ++g_vk_mock.get_images_calls;
    if (!pCount) return VK_ERROR_INITIALIZATION_FAILED;
    if (!pImages) {
        *pCount = g_vk_mock.mock_swapchain_image_count;
        return VK_SUCCESS;
    }
    const uint32_t n = std::min(*pCount, g_vk_mock.mock_swapchain_image_count);
    for (uint32_t i = 0; i < n; ++i) {
        pImages[i] = reinterpret_cast<VkImage>(0x1000 + i);
    }
    *pCount = n;
    return VK_SUCCESS;
}

static VkResult VKAPI_PTR mock_vkAcquireNextImageKHR(
    VkDevice, VkSwapchainKHR, uint64_t, VkSemaphore, VkFence, uint32_t* pIndex) {
    ++g_vk_mock.acquire_image_calls;
    if (pIndex) *pIndex = (g_vk_mock.acquire_image_calls - 1) % g_vk_mock.mock_swapchain_image_count;
    return VK_SUCCESS;
}

static VkResult VKAPI_PTR mock_vkQueuePresentKHR(VkQueue, const VkPresentInfoKHR*) {
    ++g_vk_mock.queue_present_calls;
    return VK_SUCCESS;
}

static VkResult VKAPI_PTR mock_vkCreateImageView(
    VkDevice, const VkImageViewCreateInfo*, const VkAllocationCallbacks*, VkImageView* pView) {
    ++g_vk_mock.create_image_view_calls;
    if (pView) *pView = reinterpret_cast<VkImageView>(0x2000 + g_vk_mock.create_image_view_calls.load());
    return VK_SUCCESS;
}

static void VKAPI_PTR mock_vkDestroyImageView(VkDevice, VkImageView, const VkAllocationCallbacks*) {
    ++g_vk_mock.destroy_image_view_calls;
}

static VkResult VKAPI_PTR mock_vkCreateImage(
    VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*, VkImage* pImage) {
    ++g_vk_mock.create_image_calls;
    if (pImage) *pImage = reinterpret_cast<VkImage>(0x3000 + g_vk_mock.create_image_calls.load());
    return VK_SUCCESS;
}

static void VKAPI_PTR mock_vkDestroyImage(VkDevice, VkImage, const VkAllocationCallbacks*) {
    ++g_vk_mock.destroy_image_calls;
}

static void VKAPI_PTR mock_vkGetImageMemoryRequirements(
    VkDevice, VkImage, VkMemoryRequirements* pReq) {
    if (pReq) {
        pReq->size = 1920 * 1080 * 4;
        pReq->alignment = 256;
        pReq->memoryTypeBits = 0x3;
    }
}

static VkResult VKAPI_PTR mock_vkAllocateMemory(
    VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory* pMem) {
    ++g_vk_mock.alloc_mem_calls;
    if (pMem) *pMem = reinterpret_cast<VkDeviceMemory>(0x4000 + g_vk_mock.alloc_mem_calls.load());
    return VK_SUCCESS;
}

static void VKAPI_PTR mock_vkFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {
    ++g_vk_mock.free_mem_calls;
}

static VkResult VKAPI_PTR mock_vkBindImageMemory(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    ++g_vk_mock.bind_image_mem_calls;
    return VK_SUCCESS;
}

static VkResult VKAPI_PTR mock_vkCreateFence(
    VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence* pFence) {
    ++g_vk_mock.create_fence_calls;
    if (pFence) *pFence = reinterpret_cast<VkFence>(0x5000 + g_vk_mock.create_fence_calls.load());
    return VK_SUCCESS;
}

static void VKAPI_PTR mock_vkDestroyFence(VkDevice, VkFence, const VkAllocationCallbacks*) {
    ++g_vk_mock.destroy_fence_calls;
}

static VkResult VKAPI_PTR mock_vkWaitForFences(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t) {
    ++g_vk_mock.wait_for_fences_calls;
    return VK_SUCCESS;
}

static VkResult VKAPI_PTR mock_vkResetFences(VkDevice, uint32_t, const VkFence*) {
    ++g_vk_mock.reset_fences_calls;
    return VK_SUCCESS;
}

static VkResult VKAPI_PTR mock_vkCreateSemaphore(
    VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore* pSem) {
    ++g_vk_mock.create_semaphore_calls;
    if (pSem) *pSem = reinterpret_cast<VkSemaphore>(0x6000 + g_vk_mock.create_semaphore_calls.load());
    return VK_SUCCESS;
}

static void VKAPI_PTR mock_vkDestroySemaphore(VkDevice, VkSemaphore, const VkAllocationCallbacks*) {
    ++g_vk_mock.destroy_semaphore_calls;
}

static VkResult VKAPI_PTR mock_vkQueueSubmit(VkQueue, uint32_t count, const VkSubmitInfo* pSubmits, VkFence) {
    g_vk_mock.queue_submit_calls += count;
    for (uint32_t i = 0; i < count; ++i) {
        g_vk_mock.last_submit_cmd_count = pSubmits[i].commandBufferCount;
        g_vk_mock.recorded_submits.push_back(pSubmits[i]);
    }
    return VK_SUCCESS;
}

static VkResult VKAPI_PTR mock_vkDeviceWaitIdle(VkDevice) {
    ++g_vk_mock.device_wait_idle_calls;
    return VK_SUCCESS;
}

static VkResult VKAPI_PTR mock_vkCreateCommandPool(
    VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool* pPool) {
    ++g_vk_mock.create_cmd_pool_calls;
    if (pPool) *pPool = reinterpret_cast<VkCommandPool>(0x7000);
    return VK_SUCCESS;
}

static void VKAPI_PTR mock_vkDestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {
    ++g_vk_mock.destroy_cmd_pool_calls;
}

static VkResult VKAPI_PTR mock_vkAllocateCommandBuffers(
    VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer* pCmd) {
    ++g_vk_mock.alloc_cmd_bufs_calls;
    if (pCmd) *pCmd = reinterpret_cast<VkCommandBuffer>(0x8000);
    return VK_SUCCESS;
}

static void VKAPI_PTR mock_vkFreeCommandBuffers(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*) {}
static VkResult VKAPI_PTR mock_vkResetCommandBuffer(VkCommandBuffer, VkCommandBufferResetFlags) { return VK_SUCCESS; }
static VkResult VKAPI_PTR mock_vkBeginCommandBuffer(VkCommandBuffer, const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
static VkResult VKAPI_PTR mock_vkEndCommandBuffer(VkCommandBuffer) { return VK_SUCCESS; }
static void VKAPI_PTR mock_vkCmdPipelineBarrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
    VkDependencyFlags, uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {}
static void VKAPI_PTR mock_vkCmdCopyImage(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t, const VkImageCopy*) {}
static void VKAPI_PTR mock_vkCmdCopyImageToBuffer(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, uint32_t, const VkBufferImageCopy*) {}
static void VKAPI_PTR mock_vkCmdBlitImage(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t, const VkImageBlit*, VkFilter) {}
static void VKAPI_PTR mock_vkCmdClearColorImage(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*, uint32_t, const VkImageSubresourceRange*) {}

static VkResult VKAPI_PTR mock_vkCreateBuffer(
    VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer* pBuf) {
    ++g_vk_mock.create_buffer_calls;
    if (pBuf) *pBuf = reinterpret_cast<VkBuffer>(0x9000);
    return VK_SUCCESS;
}

static void VKAPI_PTR mock_vkDestroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) {
    ++g_vk_mock.destroy_buffer_calls;
}

static void VKAPI_PTR mock_vkGetBufferMemoryRequirements(VkDevice, VkBuffer, VkMemoryRequirements* pReq) {
    if (pReq) {
        pReq->size = 1920 * 1080 * 4;
        pReq->alignment = 256;
        pReq->memoryTypeBits = 0x2;
    }
}

static VkResult VKAPI_PTR mock_vkBindBufferMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }

static uint8_t g_mock_mapped_buffer[1920 * 1080 * 4];
static VkResult VKAPI_PTR mock_vkMapMemory(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void** ppData) {
    if (ppData) *ppData = g_mock_mapped_buffer;
    return VK_SUCCESS;
}

static void VKAPI_PTR mock_vkUnmapMemory(VkDevice, VkDeviceMemory) {}

static PFN_vkVoidFunction mock_proc_lookup(const char* name) {
    std::string s(name);
    if (s == "vkGetPhysicalDeviceSurfaceSupportKHR") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkGetPhysicalDeviceSurfaceSupportKHR);
    if (s == "vkGetPhysicalDeviceSurfaceCapabilitiesKHR") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    if (s == "vkGetPhysicalDeviceSurfaceFormatsKHR") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkGetPhysicalDeviceSurfaceFormatsKHR);
    if (s == "vkGetPhysicalDeviceSurfacePresentModesKHR") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkGetPhysicalDeviceSurfacePresentModesKHR);
    if (s == "vkDestroySurfaceKHR") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkDestroySurfaceKHR);
    if (s == "vkGetPhysicalDeviceMemoryProperties") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkGetPhysicalDeviceMemoryProperties);

    if (s == "vkCreateSwapchainKHR") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCreateSwapchainKHR);
    if (s == "vkDestroySwapchainKHR") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkDestroySwapchainKHR);
    if (s == "vkGetSwapchainImagesKHR") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkGetSwapchainImagesKHR);
    if (s == "vkAcquireNextImageKHR") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkAcquireNextImageKHR);
    if (s == "vkQueuePresentKHR") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkQueuePresentKHR);

    if (s == "vkCreateImageView") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCreateImageView);
    if (s == "vkDestroyImageView") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkDestroyImageView);
    if (s == "vkCreateImage") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCreateImage);
    if (s == "vkDestroyImage") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkDestroyImage);
    if (s == "vkGetImageMemoryRequirements") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkGetImageMemoryRequirements);
    if (s == "vkAllocateMemory") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkAllocateMemory);
    if (s == "vkFreeMemory") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkFreeMemory);
    if (s == "vkBindImageMemory") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkBindImageMemory);

    if (s == "vkCreateFence") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCreateFence);
    if (s == "vkDestroyFence") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkDestroyFence);
    if (s == "vkWaitForFences") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkWaitForFences);
    if (s == "vkResetFences") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkResetFences);
    if (s == "vkCreateSemaphore") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCreateSemaphore);
    if (s == "vkDestroySemaphore") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkDestroySemaphore);
    if (s == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkQueueSubmit);
    if (s == "vkDeviceWaitIdle") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkDeviceWaitIdle);

    if (s == "vkCreateCommandPool") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCreateCommandPool);
    if (s == "vkDestroyCommandPool") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkDestroyCommandPool);
    if (s == "vkAllocateCommandBuffers") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkAllocateCommandBuffers);
    if (s == "vkFreeCommandBuffers") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkFreeCommandBuffers);
    if (s == "vkResetCommandBuffer") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkResetCommandBuffer);
    if (s == "vkBeginCommandBuffer") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkBeginCommandBuffer);
    if (s == "vkEndCommandBuffer") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkEndCommandBuffer);
    if (s == "vkCmdPipelineBarrier") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCmdPipelineBarrier);
    if (s == "vkCmdCopyImage") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCmdCopyImage);
    if (s == "vkCmdCopyImageToBuffer") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCmdCopyImageToBuffer);
    if (s == "vkCmdBlitImage") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCmdBlitImage);
    if (s == "vkCmdClearColorImage") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCmdClearColorImage);

    if (s == "vkCreateBuffer") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkCreateBuffer);
    if (s == "vkDestroyBuffer") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkDestroyBuffer);
    if (s == "vkGetBufferMemoryRequirements") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkGetBufferMemoryRequirements);
    if (s == "vkBindBufferMemory") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkBindBufferMemory);
    if (s == "vkMapMemory") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkMapMemory);
    if (s == "vkUnmapMemory") return reinterpret_cast<PFN_vkVoidFunction>(mock_vkUnmapMemory);

    return nullptr;
}

static PFN_vkVoidFunction VKAPI_PTR mock_gipa(VkInstance, const char* name) {
    return mock_proc_lookup(name);
}

static PFN_vkVoidFunction VKAPI_PTR mock_gdpa(VkDevice, const char* name) {
    return mock_proc_lookup(name);
}

} // extern "C"

// ============================================================================
// TEST 1: COMMAND LIST RECORDING STRESS & BOUNDARIES
// ============================================================================

bool test_command_list_recording_stress() {
    using namespace leonida;
    namespace s = d3d12_state;

    CommandList list;
    TEST_CHECK(!list.closed(), "Fresh list must not be closed");

    // Stress: 10,000 rapid commands
    const int kIterations = 10000;
    TargetView tv{100, 0, 0, 1};
    ClearColor col{0.1f, 0.2f, 0.3f, 1.0f};

    for (int i = 0; i < kIterations; ++i) {
        tv.resource = 100 + (i % 32);
        TEST_CHECK(list.clear_render_target(tv, col), "ClearRenderTarget must succeed");
        TEST_CHECK(list.transition({tv.resource, all_subresources, s::render_target, s::copy_source}, barrier_end_only),
                   "Transition barrier must succeed");
        TEST_CHECK(list.uav_barrier({tv.resource}), "UAV barrier must succeed");
    }

    TEST_CHECK(!list.closed(), "List must remain open during recording");
    TEST_CHECK(list.commands().size() == static_cast<size_t>(kIterations * 3), "All 30,000 commands recorded");

    // Close and verify state lock
    TEST_CHECK(list.close(), "Closing command list must succeed");
    TEST_CHECK(list.closed(), "Command list must report closed");
    TEST_CHECK(!list.close(), "Second close must fail");

    // Re-recording on closed list must fail
    TEST_CHECK(!list.clear_render_target(tv, col), "Recording on closed list must fail");
    TEST_CHECK(!list.transition({100, all_subresources, s::common, s::render_target}), "Transition on closed list must fail");

    // Reset and verify clean state
    TEST_CHECK(list.reset(), "Reset must succeed");
    TEST_CHECK(!list.closed(), "Reset list must be open");
    TEST_CHECK(list.commands().empty(), "Reset list must have empty commands");

    // Record after reset
    TEST_CHECK(list.clear_render_target(tv, col), "Recording after reset must succeed");
    TEST_CHECK(list.close(), "Closing after reset must succeed");
    TEST_CHECK(list.commands().size() == 1, "Single command recorded after reset");

    return true;
}

// Multi-threaded recording stress: 8 concurrent threads recording command lists
bool test_command_list_concurrent_recording() {
    using namespace leonida;
    namespace s = d3d12_state;

    constexpr int kThreads = 8;
    constexpr int kCommandsPerThread = 2000;
    std::vector<std::thread> threads;
    std::atomic<bool> success{true};

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t, &success]() {
            CommandList list;
            TargetView view{static_cast<ResourceId>(1000 + t), 0, 0, 1};
            ClearColor color{0.5f, 0.5f, 0.5f, 1.0f};

            for (int i = 0; i < kCommandsPerThread; ++i) {
                if (!list.clear_render_target(view, color) ||
                    !list.transition({view.resource, all_subresources, s::common, s::render_target}, barrier_end_only)) {
                    success = false;
                    return;
                }
            }
            if (!list.close() || list.commands().size() != static_cast<size_t>(kCommandsPerThread * 2)) {
                success = false;
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    TEST_CHECK(success.load(), "Concurrent command list recording must succeed across all threads");
    return true;
}

// ============================================================================
// TEST 2: EXECUTECOMMANDLISTSX MULTIPLE COMMAND LISTS & SYNC POINTS
// ============================================================================

bool test_execute_command_lists_x_multiple_lists() {
    using namespace rt::d3d12_x;

    // 1. Invalid args: count > 0 with null lists
    VulkanSubmitPlan plan;
    TEST_CHECK(translate_command_list_metadata(4, nullptr, nullptr, {}, VK_NULL_HANDLE, 1, plan) == E_INVALIDARG,
               "Null lists with count > 0 must return E_INVALIDARG");

    // 2. Empty batch: count = 0
    TEST_CHECK(translate_command_list_metadata(0, nullptr, nullptr, {}, VK_NULL_HANDLE, 1, plan) == S_OK,
               "count = 0 must succeed");
    TEST_CHECK(plan.segments.size() == 1, "Empty batch must create exactly one segment");

    // 3. Stress multiple command lists: 64 command lists in single batch
    constexpr uint32_t kListCount = 64;
    std::vector<TestMockCommandList> mock_storage(kListCount);
    std::vector<ID3D12CommandList*> dummy_lists(kListCount);
    for (uint32_t i = 0; i < kListCount; ++i) {
        dummy_lists[i] = &mock_storage[i];
    }
    std::vector<VkCommandBuffer> vk_cmds(kListCount);
    for (uint32_t i = 0; i < kListCount; ++i) {
        vk_cmds[i] = reinterpret_cast<VkCommandBuffer>(static_cast<uintptr_t>(0x10000 + i));
    }

    // Define 4 sync points at list indices 15, 31, 47, 63
    std::array<D3D12X_SYNC_POINT, 4> sync_points{};
    sync_points[0] = {D3D12X_SYNC_KIND::SEMAPHORE_SIGNAL, 15, 10, nullptr, reinterpret_cast<VkSemaphore>(0xAA1), 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    sync_points[1] = {D3D12X_SYNC_KIND::SEMAPHORE_WAIT, 16, 10, nullptr, reinterpret_cast<VkSemaphore>(0xAA1), VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0};
    sync_points[2] = {D3D12X_SYNC_KIND::SEMAPHORE_SIGNAL, 47, 20, nullptr, reinterpret_cast<VkSemaphore>(0xBB1), 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    sync_points[3] = {D3D12X_SYNC_KIND::SEMAPHORE_WAIT, 48, 20, nullptr, reinterpret_cast<VkSemaphore>(0xBB1), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0};

    D3D12X_COMMAND_LIST_METADATA meta{};
    meta.struct_size = sizeof(meta);
    meta.flags = D3D12X_EXECUTE_FLAG_FORCE_IMMEDIATE_SUBMIT | D3D12X_EXECUTE_FLAG_PIPELINE_STALL;
    meta.frame_token = 0xFEEDBEEF;
    meta.sync_point_count = static_cast<uint32_t>(sync_points.size());
    meta.sync_points = sync_points.data();

    const VkSemaphore progress_sem = reinterpret_cast<VkSemaphore>(0x9999);
    const uint64_t progress_ticket = 777;

    const HRESULT hr = translate_command_list_metadata(
        kListCount, dummy_lists.data(), &meta, vk_cmds, progress_sem, progress_ticket, plan);

    TEST_CHECK(hr == S_OK, "translate_command_list_metadata with 64 lists must return S_OK");
    TEST_CHECK(plan.immediate_submit, "Immediate submit flag must be propagated");
    TEST_CHECK(plan.pipeline_stall, "Pipeline stall flag must be propagated");
    TEST_CHECK(plan.frame_token == 0xFEEDBEEF, "Frame token must match");

    // Count total command buffers across all segments
    size_t total_cmds = 0;
    for (const auto& seg : plan.segments) {
        total_cmds += seg.cmd_buffers.size();
    }
    TEST_CHECK(total_cmds == kListCount, "All 64 command buffers must be accounted for across segments");

    // Queue progress semaphore must be attached to the final segment
    TEST_CHECK(!plan.segments.empty(), "Segments must not be empty");
    const auto& last_seg = plan.segments.back();
    bool has_progress = false;
    for (size_t s = 0; s < last_seg.signal_semaphores.size(); ++s) {
        if (last_seg.signal_semaphores[s] == progress_sem && last_seg.signal_values[s] == progress_ticket) {
            has_progress = true;
            break;
        }
    }
    TEST_CHECK(has_progress, "Final segment must signal queue progress semaphore");

    // Mock execution of submit plan
    g_vk_mock.reset();
    const VkQueue mock_q = reinterpret_cast<VkQueue>(0x5555);
    const HRESULT exec_hr = execute_vulkan_submit_plan(mock_q, plan, mock_vkQueueSubmit, VK_NULL_HANDLE);
    TEST_CHECK(exec_hr == S_OK, "execute_vulkan_submit_plan must succeed");
    TEST_CHECK(g_vk_mock.queue_submit_calls == plan.segments.size(),
               "Queue submit call count must match segment count");

    return true;
}

// ============================================================================
// TEST 3: QUEUE FLUSHES & FRAME BOUNDARY PROGRESSION
// ============================================================================

bool test_queue_flushes_and_frame_boundaries() {
    using namespace rt::d3d12_x;

    constexpr int kNumFrames = 250;
    const VkQueue mock_q = reinterpret_cast<VkQueue>(0x5555);
    const VkSemaphore timeline_sem = reinterpret_cast<VkSemaphore>(0x7777);

    uint64_t current_ticket = 0;
    g_vk_mock.reset();

    for (int frame = 1; frame <= kNumFrames; ++frame) {
        // Each frame has 3 command lists: compute pre-pass, main render, UI
        const uint32_t count = 3;
    TestMockCommandList frame_mock_lists[3];
    ID3D12CommandList* dummy[3] = {
        &frame_mock_lists[0],
        &frame_mock_lists[1],
        &frame_mock_lists[2]
    };
        std::vector<VkCommandBuffer> vk_cmds = {
            reinterpret_cast<VkCommandBuffer>(0x101),
            reinterpret_cast<VkCommandBuffer>(0x102),
            reinterpret_cast<VkCommandBuffer>(0x103)
        };

        D3D12X_COMMAND_LIST_METADATA meta{};
        meta.struct_size = sizeof(meta);
        meta.frame_token = static_cast<uint64_t>(frame);
        meta.flags = (frame % 60 == 0) ? D3D12X_EXECUTE_FLAG_PIPELINE_STALL : D3D12X_EXECUTE_FLAG_NONE;

        ++current_ticket;
        VulkanSubmitPlan plan;
        const HRESULT hr = translate_command_list_metadata(
            count, dummy, &meta, vk_cmds, timeline_sem, current_ticket, plan);
        TEST_CHECK(hr == S_OK, "Frame submission metadata translation must succeed");

        const HRESULT exec_hr = execute_vulkan_submit_plan(mock_q, plan, mock_vkQueueSubmit);
        TEST_CHECK(exec_hr == S_OK, "execute_vulkan_submit_plan must succeed");
    }

    TEST_CHECK(g_vk_mock.queue_submit_calls >= kNumFrames, "Submit count must cover all frames");
    TEST_CHECK(current_ticket == kNumFrames, "Timeline ticket must strictly advance monotonically");

    return true;
}

// ============================================================================
// TEST 4: VULKAN PRESENTATION SWAPCHAIN & HEADLESS FALLBACK
// ============================================================================

bool test_headless_presentation_swapchain() {
    leonida::VulkanDevice dev;
    g_vk_mock.reset();

    const VkInstance inst = reinterpret_cast<VkInstance>(0x1111);
    const VkPhysicalDevice phys = reinterpret_cast<VkPhysicalDevice>(0x2222);
    const VkDevice device = reinterpret_cast<VkDevice>(0x3333);
    const VkQueue queue = reinterpret_cast<VkQueue>(0x4444);

    TEST_CHECK(dev.init(inst, phys, device, queue, 0, mock_gipa, mock_gdpa),
               "VulkanDevice::init with mock GIPA/GDPA must succeed");

    leonida::VulkanSwapchainConfig cfg{};
    cfg.width = 1920;
    cfg.height = 1080;
    cfg.buffer_count = 3;
    cfg.preferred_format = VK_FORMAT_B8G8R8A8_UNORM;
    cfg.force_headless = true;

    // 1. Initialize headless swapchain
    TEST_CHECK(dev.create_headless_swapchain(cfg), "create_headless_swapchain must succeed");
    TEST_CHECK(dev.is_headless(), "Swapchain must report headless");
    TEST_CHECK(dev.back_buffer_count() == 3, "Buffer count must be 3");
    TEST_CHECK(dev.back_buffer_extent().width == 1920 && dev.back_buffer_extent().height == 1080, "Extent matches");

    // 2. Headless presentation loop (30 frames)
    int callback_invocations = 0;
    dev.swapchain()->set_frame_callback([&callback_invocations](uint32_t, VkImage img) {
        if (img != VK_NULL_HANDLE) ++callback_invocations;
    });

    for (int f = 0; f < 30; ++f) {
        uint32_t idx = 0;
        TEST_CHECK(dev.acquire_next_back_buffer(idx), "acquire_next_back_buffer must succeed");
        TEST_CHECK(idx < 3, "Acquired index within range");
        const VkResult pres_res = dev.present_back_buffer(idx);
        TEST_CHECK(pres_res == VK_SUCCESS, "Headless present must succeed with VK_SUCCESS");
    }

    TEST_CHECK(dev.swapchain()->presented_count() == 30, "Presented count must reach 30");
    TEST_CHECK(callback_invocations == 30, "Callback must be invoked 30 times");

    // 3. Swapchain resize stress in headless mode
    const std::vector<std::pair<uint32_t, uint32_t>> resolutions = {
        {1280, 720},
        {1920, 1080},
        {2560, 1440},
        {3840, 2160},
        {800, 600},
        {1920, 1080}
    };

    for (const auto& res : resolutions) {
        TEST_CHECK(dev.resize_back_buffers(res.first, res.second, VK_FORMAT_B8G8R8A8_UNORM, 3),
                   "Resize back buffers must succeed");
        TEST_CHECK(dev.back_buffer_extent().width == res.first &&
                   dev.back_buffer_extent().height == res.second, "Extent must update after resize");

        uint32_t idx = 0;
        TEST_CHECK(dev.acquire_next_back_buffer(idx), "Acquire after resize must succeed");
        TEST_CHECK(dev.present_back_buffer(idx) == VK_SUCCESS, "Present after resize must succeed");
    }

    // 4. Resource binding validation
    auto binding = dev.swapchain()->resource_binding(0);
    TEST_CHECK(binding.width == 1920 && binding.height == 1080, "Binding dimensions match");
    TEST_CHECK(binding.color_attachment, "Color attachment bit set");

    return true;
}

// Test environment variable detection for headless fallback
bool test_headless_detection_environment() {
    // Save original env
    const char* orig_headless = std::getenv("LEONIDA_HEADLESS");
    std::string saved = orig_headless ? orig_headless : "";

    setenv("LEONIDA_HEADLESS", "1", 1);
    TEST_CHECK(!leonida::VulkanDevice::is_display_available(),
               "LEONIDA_HEADLESS=1 must force is_display_available() to false");

    setenv("LEONIDA_HEADLESS", "true", 1);
    TEST_CHECK(!leonida::VulkanDevice::is_display_available(),
               "LEONIDA_HEADLESS=true must force is_display_available() to false");

    // Restore env
    if (orig_headless) {
        setenv("LEONIDA_HEADLESS", saved.c_str(), 1);
    } else {
        unsetenv("LEONIDA_HEADLESS");
    }

    return true;
}

// ============================================================================
// TEST 5: EMPIRICAL VERIFICATION OF BUGS & VULNERABILITIES FOUND
// ============================================================================

// Bug 1: Verify sync point wait attachment bug
// In translate_command_list_metadata: when sp.command_list_index == 1,
// does the wait attach to cmd 0's segment or cmd 1's segment?
bool test_bug_sync_point_wait_placement() {
    using namespace rt::d3d12_x;

    TestMockCommandList sync_mock_lists[3];
    const std::vector<ID3D12CommandList*> dummy_lists = {
        &sync_mock_lists[0],
        &sync_mock_lists[1],
        &sync_mock_lists[2]
    };
    const std::vector<VkCommandBuffer> vk_cmds = {
        reinterpret_cast<VkCommandBuffer>(0x10),
        reinterpret_cast<VkCommandBuffer>(0x20),
        reinterpret_cast<VkCommandBuffer>(0x30)
    };

    // Caller wants SEMAPHORE_WAIT before command list 1
    const VkSemaphore wait_sem = reinterpret_cast<VkSemaphore>(0xFACE);
    D3D12X_SYNC_POINT sp{};
    sp.kind = D3D12X_SYNC_KIND::SEMAPHORE_WAIT;
    sp.command_list_index = 1; // Wait before list 1!
    sp.value = 50;
    sp.vk_semaphore = wait_sem;
    sp.wait_stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;

    D3D12X_COMMAND_LIST_METADATA meta{};
    meta.struct_size = sizeof(meta);
    meta.sync_point_count = 1;
    meta.sync_points = &sp;

    VulkanSubmitPlan plan;
    translate_command_list_metadata(3, dummy_lists.data(), &meta, vk_cmds, VK_NULL_HANDLE, 0, plan);

    // Check which segment the wait was attached to:
    // In buggy code: split_indices.push_back(1 + 1 = 2) -> ranges: [0, 2), [2, 3)
    // Segment 0 contains cmd 0 and cmd 1. Target_seg is 0.
    // So segment 0 has wait_semaphores! Cmd 0 is blocked by the semaphore wait!
    std::printf("  [Empirical Probe] Segments count: %zu\n", plan.segments.size());
    for (size_t i = 0; i < plan.segments.size(); ++i) {
        std::printf("  [Empirical Probe] Segment %zu: %zu cmd buffers, %zu wait semaphores\n",
                    i, plan.segments[i].cmd_buffers.size(), plan.segments[i].wait_semaphores.size());
    }

    bool wait_attached_to_seg0_with_cmd0 = false;
    if (!plan.segments.empty() && !plan.segments[0].wait_semaphores.empty()) {
        // Check if segment 0 contains command buffer 0
        if (!plan.segments[0].cmd_buffers.empty() && plan.segments[0].cmd_buffers[0] == vk_cmds[0]) {
            wait_attached_to_seg0_with_cmd0 = true;
        }
    }

    TEST_CHECK(!wait_attached_to_seg0_with_cmd0,
               "List 0 must not wait on list 1's semaphore (split boundary must precede wait list)");
    TEST_CHECK(plan.segments.size() >= 2, "Must create at least 2 segments for wait at list 1");
    TEST_CHECK(plan.segments[0].wait_semaphores.empty(), "Segment 0 (cmd 0) must have no wait semaphores");
    TEST_CHECK(!plan.segments[1].wait_semaphores.empty(), "Segment 1 (cmd 1) must have wait semaphore attached");
    TEST_CHECK(plan.segments[1].cmd_buffers[0] == vk_cmds[1], "Segment 1 must start with cmd 1");

    return true;
}

// Bug 2: Verify fence signaling timing in execute_vulkan_submit_plan
// Ensures D3D12 fence is NOT signaled synchronously on CPU immediately after queue_submit before GPU execution completes
bool test_bug_premature_cpu_fence_signal() {
    using namespace rt::d3d12_x;

    // Mock ID3D12Fence
    struct MockD3D12Fence : public ID3D12Fence {
        uint64_t completed_val = 0;
        bool signal_called = false;
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { return E_NOTIMPL; }
        ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
        ULONG STDMETHODCALLTYPE Release() override { return 1; }
        UINT64 STDMETHODCALLTYPE GetCompletedValue() override { return completed_val; }
        HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64, void*) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE Signal(UINT64 Value) override {
            completed_val = Value;
            signal_called = true;
            return S_OK;
        }
    };

    MockD3D12Fence mock_fence;

    VulkanSubmitPlan plan;
    plan.segments.resize(1);
    plan.segments[0].signaled_fences.push_back(&mock_fence);
    plan.segments[0].signaled_fence_values.push_back(100);

    const VkQueue mock_q = reinterpret_cast<VkQueue>(0x1234);
    execute_vulkan_submit_plan(mock_q, plan, mock_vkQueueSubmit);

    TEST_CHECK(!mock_fence.signal_called,
               "D3D12 fence must NOT be signaled prematurely on CPU before GPU execution completes");

    return true;
}

// Bug 3: Verify swapchain resize cleans up fences and semaphores without leaks
bool test_bug_swapchain_resize_resource_leak() {
    leonida::VulkanPresentationSwapchain sc;
    g_vk_mock.reset();

    const VkInstance inst = reinterpret_cast<VkInstance>(0x1111);
    const VkPhysicalDevice phys = reinterpret_cast<VkPhysicalDevice>(0x2222);
    const VkDevice device = reinterpret_cast<VkDevice>(0x3333);
    const VkQueue queue = reinterpret_cast<VkQueue>(0x4444);
    const VkSurfaceKHR surface = reinterpret_cast<VkSurfaceKHR>(0x9999);

    leonida::VulkanSwapchainConfig cfg{};
    cfg.width = 1920;
    cfg.height = 1080;
    cfg.buffer_count = 2;
    cfg.force_headless = false; // test surface swapchain

    // Allow surface support
    g_vk_mock.surface_supported = true;
    // Set DISPLAY to avoid headless fallback in is_display_available
    setenv("DISPLAY", ":0", 1);
    unsetenv("LEONIDA_HEADLESS");

    sc.init(inst, phys, device, queue, 0, surface, cfg, mock_gipa, mock_gdpa);

    const uint32_t fences_after_init = g_vk_mock.create_fence_calls.load();
    const uint32_t sems_after_init = g_vk_mock.create_semaphore_calls.load();
    const uint32_t destroyed_fences_init = g_vk_mock.destroy_fence_calls.load();
    const uint32_t destroyed_sems_init = g_vk_mock.destroy_semaphore_calls.load();

    std::printf("  [Empirical Probe] After init: %u fences created (%u destroyed), %u semaphores created (%u destroyed)\n",
                fences_after_init, destroyed_fences_init, sems_after_init, destroyed_sems_init);

    // Perform resize
    sc.resize(2560, 1440);

    const uint32_t fences_after_resize = g_vk_mock.create_fence_calls.load();
    const uint32_t sems_after_resize = g_vk_mock.create_semaphore_calls.load();
    const uint32_t destroyed_fences_resize = g_vk_mock.destroy_fence_calls.load();
    const uint32_t destroyed_sems_resize = g_vk_mock.destroy_semaphore_calls.load();

    std::printf("  [Empirical Probe] After resize: %u fences created (%u destroyed), %u semaphores created (%u destroyed)\n",
                fences_after_resize, destroyed_fences_resize, sems_after_resize, destroyed_sems_resize);

    const uint32_t leaked_fences = (fences_after_resize - fences_after_init) - (destroyed_fences_resize - destroyed_fences_init);
    const uint32_t leaked_sems = (sems_after_resize - sems_after_init) - (destroyed_sems_resize - destroyed_sems_init);

    TEST_CHECK(leaked_fences == 0, "No fences leaked on swapchain resize");
    TEST_CHECK(leaked_sems == 0, "No semaphores leaked on swapchain resize");

    sc.destroy();
    return true;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::printf("====================================================================\n");
    std::printf("Leonida D3D12.X and Vulkan Presentation Empirical Stress Test Suite\n");
    std::printf("====================================================================\n");

    RUN_TEST(test_command_list_recording_stress);
    RUN_TEST(test_command_list_concurrent_recording);
    RUN_TEST(test_execute_command_lists_x_multiple_lists);
    RUN_TEST(test_queue_flushes_and_frame_boundaries);
    RUN_TEST(test_headless_presentation_swapchain);
    RUN_TEST(test_headless_detection_environment);
    RUN_TEST(test_bug_sync_point_wait_placement);
    RUN_TEST(test_bug_premature_cpu_fence_signal);
    RUN_TEST(test_bug_swapchain_resize_resource_leak);

    std::printf("====================================================================\n");
    std::printf("Stress Test Summary: %d PASSED, %d FAILED\n", g_passed, g_failed);
    std::printf("====================================================================\n");

    return g_failed > 0 ? 1 : 0;
}

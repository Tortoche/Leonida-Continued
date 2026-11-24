#pragma once

// Vulkan presentation swapchain and presenter subsystem (Issue #313, Track 3, Milestone M2):
// Implements Vulkan presentation swapchain initialization, surface creation, image acquisition,
// presentation loop, layout transitions, and headless capture fallback for CI/testing without display server.
//
// Designed for seamless integration with D3D12/Vulkan backends (vulkan_backend.h, vulkan_device.h,
// dxgi_swapchain.h, dxgi_com.h, xbox_present.h).

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
using HWND = void*;
using HINSTANCE = void*;
using HMONITOR = void*;
using BOOL = int;
using DWORD = unsigned long;
using LONG = long;
using UINT = unsigned int;
using ULONG = unsigned long;
using HRESULT = long;
using HANDLE = void*;
#endif

#include "vulkan_device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace leonida {

struct PresenterConfig {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    VkFormat preferred_format = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    std::uint32_t buffer_count = 2;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    bool allow_tearing = false;
    bool force_headless = false;
    VkImageUsageFlags extra_usage = 0;
};

struct PresenterFrame {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore acquired_semaphore = VK_NULL_HANDLE;
    VkSemaphore rendered_semaphore = VK_NULL_HANDLE;
    std::uint32_t index = 0;
    bool initialized = false;
};

struct VulkanPresenterFunctions {
#if defined(_WIN32)
    PFN_vkCreateWin32SurfaceKHR create_win32_surface = nullptr;
#endif
    PFN_vkDestroySurfaceKHR destroy_surface = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR surface_support = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR surface_caps = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR surface_formats = nullptr;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR surface_present_modes = nullptr;

    // Swapchain functions
    PFN_vkCreateSwapchainKHR create_swapchain = nullptr;
    PFN_vkDestroySwapchainKHR destroy_swapchain = nullptr;
    PFN_vkGetSwapchainImagesKHR get_images = nullptr;
    PFN_vkAcquireNextImageKHR acquire_next_image = nullptr;
    PFN_vkQueuePresentKHR queue_present = nullptr;

    // Core device / memory functions
    PFN_vkCreateImage create_image = nullptr;
    PFN_vkDestroyImage destroy_image = nullptr;
    PFN_vkGetImageMemoryRequirements image_mem_req = nullptr;
    PFN_vkAllocateMemory alloc_mem = nullptr;
    PFN_vkFreeMemory free_mem = nullptr;
    PFN_vkBindImageMemory bind_image_mem = nullptr;
    PFN_vkCreateImageView create_image_view = nullptr;
    PFN_vkDestroyImageView destroy_image_view = nullptr;

    // Synchronization functions
    PFN_vkCreateFence create_fence = nullptr;
    PFN_vkDestroyFence destroy_fence = nullptr;
    PFN_vkWaitForFences wait_for_fences = nullptr;
    PFN_vkResetFences reset_fences = nullptr;
    PFN_vkCreateSemaphore create_semaphore = nullptr;
    PFN_vkDestroySemaphore destroy_semaphore = nullptr;
    PFN_vkQueueSubmit queue_submit = nullptr;
    PFN_vkDeviceWaitIdle device_wait_idle = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties mem_props = nullptr;

    // Command pool and command buffer functions
    PFN_vkCreateCommandPool create_cmd_pool = nullptr;
    PFN_vkDestroyCommandPool destroy_cmd_pool = nullptr;
    PFN_vkAllocateCommandBuffers alloc_cmd_bufs = nullptr;
    PFN_vkFreeCommandBuffers free_cmd_bufs = nullptr;
    PFN_vkResetCommandBuffer reset_cmd_buf = nullptr;
    PFN_vkBeginCommandBuffer begin_cmd_buf = nullptr;
    PFN_vkEndCommandBuffer end_cmd_buf = nullptr;
    PFN_vkCmdPipelineBarrier cmd_barrier = nullptr;
    PFN_vkCmdCopyImage cmd_copy_image = nullptr;
    PFN_vkCmdCopyImageToBuffer cmd_copy_image_to_buffer = nullptr;
    PFN_vkCmdBlitImage cmd_blit = nullptr;
    PFN_vkCmdClearColorImage cmd_clear_color = nullptr;

    // Buffer functions for capture / readback
    PFN_vkCreateBuffer create_buffer = nullptr;
    PFN_vkDestroyBuffer destroy_buffer = nullptr;
    PFN_vkGetBufferMemoryRequirements buffer_mem_req = nullptr;
    PFN_vkBindBufferMemory bind_buffer_mem = nullptr;
    PFN_vkMapMemory map_mem = nullptr;
    PFN_vkUnmapMemory unmap_mem = nullptr;

    bool load(PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
              PFN_vkGetDeviceProcAddr gdpa, VkDevice device);
};

class VulkanPresenter {
public:
    VulkanPresenter();
    ~VulkanPresenter();

    VulkanPresenter(const VulkanPresenter&) = delete;
    VulkanPresenter& operator=(const VulkanPresenter&) = delete;

    // Initialization with an optional display surface or automatic headless detection
    bool init(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
              VkQueue queue, std::uint32_t queue_family, VkSurfaceKHR surface,
              const PresenterConfig& config,
              PFN_vkGetInstanceProcAddr gipa = nullptr,
              PFN_vkGetDeviceProcAddr gdpa = nullptr);

    // Explicit headless offscreen initialization (CI / test environment without display server)
    bool init_headless(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
                       VkQueue queue, std::uint32_t queue_family,
                       const PresenterConfig& config,
                       PFN_vkGetInstanceProcAddr gipa = nullptr,
                       PFN_vkGetDeviceProcAddr gdpa = nullptr);

    // Window surface creation helper (Win32)
    static bool create_win32_surface(VkInstance instance, HWND hwnd, VkSurfaceKHR* out_surface,
                                     PFN_vkGetInstanceProcAddr gipa = nullptr);

    // Detects whether an interactive display server is available
    static bool is_display_available();

    // Image acquisition
    bool acquire_next_image(std::uint32_t& out_image_index, std::uint64_t timeout_ns = UINT64_MAX);

    // Presentation loop
    VkResult present(std::uint32_t back_buffer_index, VkSemaphore wait_semaphore = VK_NULL_HANDLE);

    // Clears back buffer to black and presents
    VkResult present_blank(bool vsync = true, bool tear = false);

    // Blits a source texture into the swapchain and presents
    VkResult present_blit(VkImage src_image, std::uint32_t src_width, std::uint32_t src_height,
                          VkFormat src_format, bool vsync = true, bool tear = false);

    // Resizing
    bool resize(std::uint32_t width, std::uint32_t height,
                VkFormat preferred_format = VK_FORMAT_UNDEFINED,
                std::uint32_t buffer_count = 0,
                bool allow_tearing = false);

    // Teardown
    void destroy();

    // Headless capture / testing support
    bool enable_capture(bool enable = true);
    bool read_back(std::vector<std::uint8_t>& out_pixels, VkExtent2D* out_extent = nullptr,
                   VkFormat* out_format = nullptr);

    // Frame presentation callback hook for validation / tests
    void set_frame_callback(std::function<void(std::uint32_t frame_index, VkImage image)> callback) {
        std::lock_guard lock(mutex_);
        frame_callback_ = std::move(callback);
    }

    // Status queries
    bool is_valid() const { return initialized_; }
    bool is_headless() const { return headless_; }
    std::uint32_t image_count() const { return exposed_count_; }
    std::uint32_t current_back_buffer_index() const { return current_index_; }
    std::uint32_t acquired_index() const { return acquired_index_; }
    VkFormat format() const { return format_; }
    VkColorSpaceKHR color_space() const { return color_space_; }
    VkExtent2D extent() const { return extent_; }
    VkPresentModeKHR present_mode() const { return present_mode_; }
    VkSwapchainKHR swapchain() const { return swapchain_; }
    VkSurfaceKHR surface() const { return surface_; }
    std::uint64_t presented_count() const { return presentation_count_; }

    VkImage image(std::uint32_t index) const {
        if (index >= images_.size()) return VK_NULL_HANDLE;
        return images_[index];
    }

    VkImageView image_view(std::uint32_t index) const {
        if (index >= image_views_.size()) return VK_NULL_HANDLE;
        return image_views_[index];
    }

    const std::vector<VkImage>& images() const { return images_; }
    const std::vector<VkImageView>& image_views() const { return image_views_; }

    // Interop ResourceBinding for D3D12 layer
    ResourceBinding resource_binding(std::uint32_t index) const;

    const VulkanPresenterFunctions& functions() const { return fn_; }

private:
    bool create_surface_swapchain(const PresenterConfig& config);
    bool create_headless_images(const PresenterConfig& config);
    bool create_image_views();
    void destroy_image_views();
    void destroy_frames();
    bool setup_capture_buffer();
    void release_capture_buffer();
    bool capture_image_to_buffer(VkImage src_image);
    bool transition_image(VkImage image, VkImageLayout old_layout, VkImageLayout new_layout,
                          VkAccessFlags src_access, VkAccessFlags dst_access,
                          VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage);

    std::uint32_t find_memory_type(std::uint32_t type_filter, VkMemoryPropertyFlags properties);

    VulkanPresenterFunctions fn_{};
    bool loaded_ = false;
    bool initialized_ = false;
    bool headless_ = false;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;

    VkFormat format_ = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR color_space_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D extent_{0, 0};
    VkExtent2D surface_extent_{0, 0};
    VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    bool allow_tearing_ = false;

    std::vector<VkImage> images_;
    std::vector<VkImageView> image_views_;
    std::vector<PresenterFrame> frames_;
    std::vector<bool> image_initialized_;
    std::uint32_t exposed_count_ = 0;
    std::uint32_t current_index_ = 0;
    std::uint32_t acquired_index_ = 0;
    std::uint64_t presentation_count_ = 0;

    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buf_ = VK_NULL_HANDLE;
    VkFence sync_fence_ = VK_NULL_HANDLE;

    // Headless capture buffer
    bool capture_enabled_ = false;
    VkBuffer capture_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory capture_memory_ = VK_NULL_HANDLE;
    std::size_t capture_buffer_size_ = 0;
    VkExtent2D captured_extent_{0, 0};
    VkFormat captured_format_ = VK_FORMAT_UNDEFINED;
    bool frame_captured_ = false;

    std::function<void(std::uint32_t frame_index, VkImage image)> frame_callback_;
    mutable std::mutex mutex_;
};

}  // namespace leonida

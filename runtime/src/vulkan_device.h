#pragma once

// Vulkan presentation swapchain and device presentation management (Issue #313, Track 3):
// Implements Vulkan presentation swapchain initialization, image retrieval, image view creation,
// and retail frame presentation loop with headless/offscreen presenter fallback.
// Compatible with runtime/shims/dxgi_swapchain.h, dxgi_com.cpp, and xbox_present.cpp.

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
#endif

#include <cstdint>
#include <vulkan/vulkan.h>

#ifndef LEONIDA_RESOURCE_BINDING_DECL
#define LEONIDA_RESOURCE_BINDING_DECL
namespace leonida {
struct ResourceBinding {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t mip_levels = 1;
    std::uint32_t array_layers = 1;
    bool presentable = false;
    bool simultaneous_access = false;
    bool not_zeroed = false;
    bool sampled_in_general = false;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize buffer_offset = 0;
    VkDeviceSize size = 0;
    const std::uint8_t* host = nullptr;
    std::uint64_t va = 0;
    std::uint32_t depth = 1;
    bool depth_or_stencil = false;
    bool color_attachment = false;
    bool depth_stencil_attachment = false;
    bool unordered_access = false;
    bool memory_barrier_needed = false;
    std::uint32_t samples = 1;
    bool sampled = false;
};
}
#endif

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace leonida {

struct VulkanSwapchainConfig {
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

struct VulkanSwapchainFrame {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore acquire_semaphore = VK_NULL_HANDLE;
    VkSemaphore render_semaphore = VK_NULL_HANDLE;
    std::uint32_t index = 0;
};

struct VulkanSwapchainFunctions {
    PFN_vkCreateSwapchainKHR create_swapchain = nullptr;
    PFN_vkDestroySwapchainKHR destroy_swapchain = nullptr;
    PFN_vkGetSwapchainImagesKHR get_images = nullptr;
    PFN_vkAcquireNextImageKHR acquire_next_image = nullptr;
    PFN_vkQueuePresentKHR queue_present = nullptr;

    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR surface_caps = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR surface_formats = nullptr;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR surface_present_modes = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR surface_support = nullptr;
    PFN_vkDestroySurfaceKHR destroy_surface = nullptr;

    PFN_vkCreateImageView create_image_view = nullptr;
    PFN_vkDestroyImageView destroy_image_view = nullptr;
    PFN_vkCreateImage create_image = nullptr;
    PFN_vkDestroyImage destroy_image = nullptr;
    PFN_vkGetImageMemoryRequirements image_mem_req = nullptr;
    PFN_vkAllocateMemory alloc_mem = nullptr;
    PFN_vkFreeMemory free_mem = nullptr;
    PFN_vkBindImageMemory bind_image_mem = nullptr;

    PFN_vkCreateFence create_fence = nullptr;
    PFN_vkDestroyFence destroy_fence = nullptr;
    PFN_vkWaitForFences wait_for_fences = nullptr;
    PFN_vkResetFences reset_fences = nullptr;
    PFN_vkCreateSemaphore create_semaphore = nullptr;
    PFN_vkDestroySemaphore destroy_semaphore = nullptr;
    PFN_vkQueueSubmit queue_submit = nullptr;
    PFN_vkDeviceWaitIdle device_wait_idle = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties mem_props = nullptr;

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

    PFN_vkCreateBuffer create_buffer = nullptr;
    PFN_vkDestroyBuffer destroy_buffer = nullptr;
    PFN_vkGetBufferMemoryRequirements buffer_mem_req = nullptr;
    PFN_vkBindBufferMemory bind_buffer_mem = nullptr;
    PFN_vkMapMemory map_mem = nullptr;
    PFN_vkUnmapMemory unmap_mem = nullptr;

    bool load(PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
              PFN_vkGetDeviceProcAddr gdpa, VkDevice device);
};

class VulkanPresentationSwapchain {
public:
    VulkanPresentationSwapchain();
    ~VulkanPresentationSwapchain();

    VulkanPresentationSwapchain(const VulkanPresentationSwapchain&) = delete;
    VulkanPresentationSwapchain& operator=(const VulkanPresentationSwapchain&) = delete;

    bool init(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
              VkQueue queue, std::uint32_t queue_family, VkSurfaceKHR surface,
              const VulkanSwapchainConfig& config,
              PFN_vkGetInstanceProcAddr gipa = nullptr,
              PFN_vkGetDeviceProcAddr gdpa = nullptr);

    bool init_headless(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
                       VkQueue queue, std::uint32_t queue_family,
                       const VulkanSwapchainConfig& config,
                       PFN_vkGetInstanceProcAddr gipa = nullptr,
                       PFN_vkGetDeviceProcAddr gdpa = nullptr);

    bool resize(std::uint32_t width, std::uint32_t height,
                VkFormat preferred_format = VK_FORMAT_UNDEFINED,
                std::uint32_t buffer_count = 0,
                bool allow_tearing = false);

    bool acquire_next_image(std::uint32_t& out_image_index, std::uint64_t timeout_ns = UINT64_MAX);

    VkResult present(std::uint32_t back_buffer_index, VkSemaphore wait_semaphore = VK_NULL_HANDLE);

    void destroy();

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
    std::uint64_t presented_count() const { return presentation_count_; }

    // DXGI compatibility
    ResourceBinding resource_binding(std::uint32_t index) const;

    // Offscreen / capture support
    bool enable_capture(bool enable = true);
    bool read_back(std::vector<std::uint8_t>& out_pixels, VkExtent2D* out_extent = nullptr,
                   VkFormat* out_format = nullptr);
    void set_frame_callback(std::function<void(std::uint32_t frame_index, VkImage image)> callback) {
        frame_callback_ = std::move(callback);
    }

    const VulkanSwapchainFunctions& functions() const { return fn_; }

private:
    bool create_surface_swapchain(const VulkanSwapchainConfig& config);
    bool create_headless_images(const VulkanSwapchainConfig& config);
    bool create_image_views();
    void destroy_image_views();
    void destroy_frames();
    bool setup_capture_buffer();
    void release_capture_buffer();
    bool capture_image_to_buffer(VkImage src_image);

    std::uint32_t find_memory_type(std::uint32_t type_filter, VkMemoryPropertyFlags properties);

    VulkanSwapchainFunctions fn_{};
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
    std::vector<VulkanSwapchainFrame> frames_;
    std::uint32_t exposed_count_ = 0;
    std::uint32_t current_index_ = 0;
    std::uint32_t acquired_index_ = 0;
    std::uint64_t presentation_count_ = 0;

    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buf_ = VK_NULL_HANDLE;
    VkFence sync_fence_ = VK_NULL_HANDLE;

    // Capture / staging
    bool capture_enabled_ = false;
    VkBuffer capture_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory capture_memory_ = VK_NULL_HANDLE;
    std::size_t capture_buffer_size_ = 0;

    std::function<void(std::uint32_t frame_index, VkImage image)> frame_callback_;
    mutable std::mutex mutex_;
};

class VulkanDevice {
public:
    VulkanDevice();
    VulkanDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                 VkQueue queue, std::uint32_t queue_family,
                 PFN_vkGetInstanceProcAddr gipa = nullptr,
                 PFN_vkGetDeviceProcAddr gdpa = nullptr);
    ~VulkanDevice();

    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;

    bool init(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
              VkQueue queue, std::uint32_t queue_family,
              PFN_vkGetInstanceProcAddr gipa = nullptr,
              PFN_vkGetDeviceProcAddr gdpa = nullptr);

    // Presentation swapchain lifecycle
    bool create_swapchain(VkSurfaceKHR surface, const VulkanSwapchainConfig& config);
    bool create_headless_swapchain(const VulkanSwapchainConfig& config);
    bool init_presentation(HWND hwnd, std::uint32_t width, std::uint32_t height,
                           VkFormat preferred_format = VK_FORMAT_B8G8R8A8_UNORM,
                           std::uint32_t buffer_count = 2, bool allow_tearing = false);

    bool acquire_next_back_buffer(std::uint32_t& out_index, std::uint64_t timeout_ns = UINT64_MAX);
    VkResult present_back_buffer(std::uint32_t back_buffer_index, VkSemaphore wait_semaphore = VK_NULL_HANDLE);
    bool resize_back_buffers(std::uint32_t width, std::uint32_t height,
                             VkFormat preferred_format = VK_FORMAT_UNDEFINED,
                             std::uint32_t buffer_count = 0, bool allow_tearing = false);

    std::uint32_t back_buffer_count() const;
    std::uint32_t current_back_buffer_index() const;
    VkImage back_buffer_image(std::uint32_t index) const;
    VkImageView back_buffer_view(std::uint32_t index) const;
    VkExtent2D back_buffer_extent() const;
    VkFormat back_buffer_format() const;
    bool is_headless() const;

    VulkanPresentationSwapchain* swapchain() { return swapchain_.get(); }
    const VulkanPresentationSwapchain* swapchain() const { return swapchain_.get(); }

    // Helpers
    VkImageView create_image_view_2d(VkImage image, VkFormat format,
                                     VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
    void destroy_image_view(VkImageView view);

    static bool is_display_available();
    static bool create_win32_surface(VkInstance instance, HWND hwnd, VkSurfaceKHR* out_surface,
                                     PFN_vkGetInstanceProcAddr gipa = nullptr);

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical_device() const { return physical_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    std::uint32_t queue_family() const { return queue_family_; }
    const VulkanSwapchainFunctions& vk() const { return fn_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    PFN_vkGetInstanceProcAddr gipa_ = nullptr;
    PFN_vkGetDeviceProcAddr gdpa_ = nullptr;
    VulkanSwapchainFunctions fn_{};
    bool loaded_ = false;
    std::unique_ptr<VulkanPresentationSwapchain> swapchain_;
};

}  // namespace leonida

namespace rt::d3d12 {
using VulkanPresentationSwapchain = leonida::VulkanPresentationSwapchain;
using VulkanDevice = leonida::VulkanDevice;
using VulkanSwapchainConfig = leonida::VulkanSwapchainConfig;
}  // namespace rt::d3d12

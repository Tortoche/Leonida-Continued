#include "vulkan_device.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#if !defined(_WIN32)
#include <dlfcn.h>
static PFN_vkGetInstanceProcAddr load_default_gipa() {
    void* lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) return nullptr;
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
}
#else
static PFN_vkGetInstanceProcAddr load_default_gipa() {
    HMODULE mod = GetModuleHandleA("vulkan-1.dll");
    if (!mod) mod = LoadLibraryA("vulkan-1.dll");
    if (!mod) return nullptr;
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(mod, "vkGetInstanceProcAddr"));
}
#endif

namespace leonida {

bool VulkanSwapchainFunctions::load(PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
                                     PFN_vkGetDeviceProcAddr gdpa, VkDevice device) {
    if (!gipa) gipa = load_default_gipa();
    if (!gipa || !instance) return false;

#define LOAD_I(member, name) member = reinterpret_cast<PFN_##name>(gipa(instance, #name))
    LOAD_I(surface_caps, vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    LOAD_I(surface_formats, vkGetPhysicalDeviceSurfaceFormatsKHR);
    LOAD_I(surface_present_modes, vkGetPhysicalDeviceSurfacePresentModesKHR);
    LOAD_I(surface_support, vkGetPhysicalDeviceSurfaceSupportKHR);
    LOAD_I(destroy_surface, vkDestroySurfaceKHR);
    LOAD_I(mem_props, vkGetPhysicalDeviceMemoryProperties);
#undef LOAD_I

    if (!gdpa && gipa) {
        gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
    }
    if (!gdpa || !device) return false;

#define LOAD_D(member, name) member = reinterpret_cast<PFN_##name>(gdpa(device, #name))
    LOAD_D(create_swapchain, vkCreateSwapchainKHR);
    LOAD_D(destroy_swapchain, vkDestroySwapchainKHR);
    LOAD_D(get_images, vkGetSwapchainImagesKHR);
    LOAD_D(acquire_next_image, vkAcquireNextImageKHR);
    LOAD_D(queue_present, vkQueuePresentKHR);

    LOAD_D(create_image_view, vkCreateImageView);
    LOAD_D(destroy_image_view, vkDestroyImageView);
    LOAD_D(create_image, vkCreateImage);
    LOAD_D(destroy_image, vkDestroyImage);
    LOAD_D(image_mem_req, vkGetImageMemoryRequirements);
    LOAD_D(alloc_mem, vkAllocateMemory);
    LOAD_D(free_mem, vkFreeMemory);
    LOAD_D(bind_image_mem, vkBindImageMemory);

    LOAD_D(create_fence, vkCreateFence);
    LOAD_D(destroy_fence, vkDestroyFence);
    LOAD_D(wait_for_fences, vkWaitForFences);
    LOAD_D(reset_fences, vkResetFences);
    LOAD_D(create_semaphore, vkCreateSemaphore);
    LOAD_D(destroy_semaphore, vkDestroySemaphore);
    LOAD_D(queue_submit, vkQueueSubmit);
    LOAD_D(device_wait_idle, vkDeviceWaitIdle);

    LOAD_D(create_cmd_pool, vkCreateCommandPool);
    LOAD_D(destroy_cmd_pool, vkDestroyCommandPool);
    LOAD_D(alloc_cmd_bufs, vkAllocateCommandBuffers);
    LOAD_D(free_cmd_bufs, vkFreeCommandBuffers);
    LOAD_D(reset_cmd_buf, vkResetCommandBuffer);
    LOAD_D(begin_cmd_buf, vkBeginCommandBuffer);
    LOAD_D(end_cmd_buf, vkEndCommandBuffer);
    LOAD_D(cmd_barrier, vkCmdPipelineBarrier);
    LOAD_D(cmd_copy_image, vkCmdCopyImage);
    LOAD_D(cmd_copy_image_to_buffer, vkCmdCopyImageToBuffer);
    LOAD_D(cmd_blit, vkCmdBlitImage);
    LOAD_D(cmd_clear_color, vkCmdClearColorImage);

    LOAD_D(create_buffer, vkCreateBuffer);
    LOAD_D(destroy_buffer, vkDestroyBuffer);
    LOAD_D(buffer_mem_req, vkGetBufferMemoryRequirements);
    LOAD_D(bind_buffer_mem, vkBindBufferMemory);
    LOAD_D(map_mem, vkMapMemory);
    LOAD_D(unmap_mem, vkUnmapMemory);
#undef LOAD_D

    return create_image_view && destroy_image_view && create_image && destroy_image &&
           alloc_mem && free_mem && bind_image_mem && create_fence && destroy_fence &&
           wait_for_fences && reset_fences && create_semaphore && destroy_semaphore &&
           queue_submit && device_wait_idle && mem_props;
}

VulkanPresentationSwapchain::VulkanPresentationSwapchain() = default;

VulkanPresentationSwapchain::~VulkanPresentationSwapchain() {
    destroy();
}

void VulkanPresentationSwapchain::destroy() {
    std::lock_guard lock(mutex_);
    if (!device_) return;

    if (fn_.device_wait_idle) fn_.device_wait_idle(device_);

    release_capture_buffer();
    destroy_image_views();
    destroy_frames();

    if (swapchain_ && fn_.destroy_swapchain) {
        fn_.destroy_swapchain(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }

    if (cmd_pool_ && fn_.destroy_cmd_pool) {
        fn_.destroy_cmd_pool(device_, cmd_pool_, nullptr);
        cmd_pool_ = VK_NULL_HANDLE;
        cmd_buf_ = VK_NULL_HANDLE;
    }

    if (sync_fence_ && fn_.destroy_fence) {
        fn_.destroy_fence(device_, sync_fence_, nullptr);
        sync_fence_ = VK_NULL_HANDLE;
    }

    images_.clear();
    exposed_count_ = 0;
    current_index_ = 0;
    acquired_index_ = 0;
    initialized_ = false;
    headless_ = false;
}

std::uint32_t VulkanPresentationSwapchain::find_memory_type(std::uint32_t type_filter,
                                                            VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props{};
    fn_.mem_props(physical_, &mem_props);
    for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    // Fallback if requested properties (e.g. device local) are not strictly matched
    for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if (type_filter & (1u << i)) return i;
    }
    return 0;
}

bool VulkanPresentationSwapchain::init(VkInstance instance, VkPhysicalDevice physical_device,
                                       VkDevice device, VkQueue queue, std::uint32_t queue_family,
                                       VkSurfaceKHR surface, const VulkanSwapchainConfig& config,
                                       PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa) {
    destroy();
    std::lock_guard lock(mutex_);

    instance_ = instance;
    physical_ = physical_device;
    device_ = device;
    queue_ = queue;
    queue_family_ = queue_family;
    surface_ = surface;

    loaded_ = fn_.load(gipa, instance, gdpa, device);
    if (!loaded_) return false;

    // Command pool and helper synchronization fence for internal blit/copy/transitions
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family_;
    if (fn_.create_cmd_pool(device_, &pci, nullptr, &cmd_pool_) != VK_SUCCESS) return false;

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cmd_pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (fn_.alloc_cmd_bufs(device_, &cai, &cmd_buf_) != VK_SUCCESS) return false;

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (fn_.create_fence(device_, &fci, nullptr, &sync_fence_) != VK_SUCCESS) return false;

    // Check whether to use headless fallback:
    // 1. Explicitly forced headless
    // 2. No surface provided (VK_NULL_HANDLE)
    // 3. Display unavailable in environment
    // 4. Surface functions missing or surface not supported on this physical device
    bool use_headless = config.force_headless || (surface == VK_NULL_HANDLE) ||
                        !VulkanDevice::is_display_available() ||
                        !fn_.surface_support || !fn_.create_swapchain;

    if (!use_headless && surface != VK_NULL_HANDLE) {
        VkBool32 supported = VK_FALSE;
        if (fn_.surface_support(physical_, queue_family_, surface, &supported) != VK_SUCCESS || !supported) {
            use_headless = true;
        }
    }

    if (use_headless) {
        headless_ = true;
        if (!create_headless_images(config)) return false;
    } else {
        headless_ = false;
        if (!create_surface_swapchain(config)) {
            // Surface swapchain creation failed, fallback to headless offscreen presenter
            headless_ = true;
            if (!create_headless_images(config)) return false;
        }
    }

    if (!create_image_views()) {
        destroy();
        return false;
    }

    initialized_ = true;
    return true;
}

bool VulkanPresentationSwapchain::init_headless(VkInstance instance, VkPhysicalDevice physical_device,
                                                VkDevice device, VkQueue queue, std::uint32_t queue_family,
                                                const VulkanSwapchainConfig& config,
                                                PFN_vkGetInstanceProcAddr gipa,
                                                PFN_vkGetDeviceProcAddr gdpa) {
    VulkanSwapchainConfig cfg = config;
    cfg.force_headless = true;
    return init(instance, physical_device, device, queue, queue_family, VK_NULL_HANDLE, cfg, gipa, gdpa);
}

bool VulkanPresentationSwapchain::create_surface_swapchain(const VulkanSwapchainConfig& config) {
    if (!surface_ || !fn_.surface_caps || !fn_.surface_formats || !fn_.create_swapchain) return false;

    VkSurfaceCapabilitiesKHR caps{};
    if (fn_.surface_caps(physical_, surface_, &caps) != VK_SUCCESS) return false;

    std::uint32_t fmt_count = 0;
    fn_.surface_formats(physical_, surface_, &fmt_count, nullptr);
    if (fmt_count == 0) return false;
    std::vector<VkSurfaceFormatKHR> formats(fmt_count);
    fn_.surface_formats(physical_, surface_, &fmt_count, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto& sf : formats) {
        if (sf.format == config.preferred_format && sf.colorSpace == config.color_space) {
            chosen = sf;
            break;
        }
    }

    VkExtent2D ext{};
    if (caps.currentExtent.width != UINT32_MAX && caps.currentExtent.width > 0 && caps.currentExtent.height > 0) {
        ext = caps.currentExtent;
    } else {
        const std::uint32_t w = config.width ? config.width : 1920;
        const std::uint32_t h = config.height ? config.height : 1080;
        ext.width = std::clamp(w, caps.minImageExtent.width, caps.maxImageExtent.width);
        ext.height = std::clamp(h, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    std::uint32_t count = std::max(config.buffer_count ? config.buffer_count : 2u, caps.minImageCount);
    if (caps.maxImageCount > 0) count = std::min(count, caps.maxImageCount);

    VkPresentModeKHR chosen_present_mode = VK_PRESENT_MODE_FIFO_KHR;
    if (config.allow_tearing && fn_.surface_present_modes) {
        std::uint32_t mode_count = 0;
        if (fn_.surface_present_modes(physical_, surface_, &mode_count, nullptr) == VK_SUCCESS && mode_count > 0) {
            std::vector<VkPresentModeKHR> modes(mode_count);
            if (fn_.surface_present_modes(physical_, surface_, &mode_count, modes.data()) == VK_SUCCESS) {
                for (VkPresentModeKHR m : modes) {
                    if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                        chosen_present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                        break;
                    }
                    if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
                        chosen_present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
                    }
                }
            }
        }
    }

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface_;
    sci.minImageCount = count;
    sci.imageFormat = chosen.format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = ext;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | config.extra_usage;
    if (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        sci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) {
        if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
            sci.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        else if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR)
            sci.compositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
        else if (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR)
            sci.compositeAlpha = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
    }
    sci.presentMode = chosen_present_mode;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = swapchain_;

    VkSwapchainKHR next = VK_NULL_HANDLE;
    if (fn_.create_swapchain(device_, &sci, nullptr, &next) != VK_SUCCESS) return false;

    if (swapchain_) fn_.destroy_swapchain(device_, swapchain_, nullptr);
    swapchain_ = next;

    format_ = chosen.format;
    color_space_ = chosen.colorSpace;
    extent_ = (config.width > 0 && config.height > 0) ? VkExtent2D{config.width, config.height} : ext;
    surface_extent_ = ext;
    present_mode_ = chosen_present_mode;
    allow_tearing_ = config.allow_tearing;

    std::uint32_t img_count = 0;
    fn_.get_images(device_, swapchain_, &img_count, nullptr);
    if (img_count == 0) return false;
    images_.resize(img_count);
    fn_.get_images(device_, swapchain_, &img_count, images_.data());
    exposed_count_ = img_count;

    destroy_frames();
    frames_.resize(img_count);
    for (std::uint32_t i = 0; i < img_count; ++i) {
        frames_[i].image = images_[i];
        frames_[i].index = i;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
        fn_.create_fence(device_, &fci, nullptr, &frames_[i].fence);
        VkSemaphoreCreateInfo sci_sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        fn_.create_semaphore(device_, &sci_sem, nullptr, &frames_[i].acquire_semaphore);
        fn_.create_semaphore(device_, &sci_sem, nullptr, &frames_[i].render_semaphore);
    }

    return true;
}

bool VulkanPresentationSwapchain::create_headless_images(const VulkanSwapchainConfig& config) {
    format_ = (config.preferred_format != VK_FORMAT_UNDEFINED) ? config.preferred_format : VK_FORMAT_B8G8R8A8_UNORM;
    color_space_ = config.color_space;
    const std::uint32_t w = config.width ? config.width : 1920;
    const std::uint32_t h = config.height ? config.height : 1080;
    extent_ = {w, h};
    surface_extent_ = extent_;
    exposed_count_ = config.buffer_count ? config.buffer_count : 2;
    present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    allow_tearing_ = config.allow_tearing;

    destroy_frames();
    images_.resize(exposed_count_);
    frames_.resize(exposed_count_);

    for (std::uint32_t i = 0; i < exposed_count_; ++i) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format_;
        ici.extent = {extent_.width, extent_.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | config.extra_usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage img = VK_NULL_HANDLE;
        if (fn_.create_image(device_, &ici, nullptr, &img) != VK_SUCCESS) return false;
        images_[i] = img;
        frames_[i].image = img;
        frames_[i].index = i;

        VkMemoryRequirements req{};
        fn_.image_mem_req(device_, img, &req);
        const std::uint32_t mem_type = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = mem_type;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        if (fn_.alloc_mem(device_, &mai, nullptr, &mem) != VK_SUCCESS) return false;
        if (fn_.bind_image_mem(device_, img, mem, 0) != VK_SUCCESS) {
            fn_.free_mem(device_, mem, nullptr);
            return false;
        }
        frames_[i].memory = mem;

        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
        fn_.create_fence(device_, &fci, nullptr, &frames_[i].fence);
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        fn_.create_semaphore(device_, &sci, nullptr, &frames_[i].acquire_semaphore);
        fn_.create_semaphore(device_, &sci, nullptr, &frames_[i].render_semaphore);
    }

    return true;
}

bool VulkanPresentationSwapchain::create_image_views() {
    destroy_image_views();
    image_views_.resize(images_.size(), VK_NULL_HANDLE);

    for (std::size_t i = 0; i < images_.size(); ++i) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = images_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format_;
        vi.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                         VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.baseMipLevel = 0;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.baseArrayLayer = 0;
        vi.subresourceRange.layerCount = 1;

        if (fn_.create_image_view(device_, &vi, nullptr, &image_views_[i]) != VK_SUCCESS) {
            destroy_image_views();
            return false;
        }
        if (i < frames_.size()) frames_[i].view = image_views_[i];
    }
    return true;
}

void VulkanPresentationSwapchain::destroy_image_views() {
    for (VkImageView view : image_views_) {
        if (view && fn_.destroy_image_view) fn_.destroy_image_view(device_, view, nullptr);
    }
    image_views_.clear();
}

void VulkanPresentationSwapchain::destroy_frames() {
    for (auto& f : frames_) {
        if (f.fence && fn_.destroy_fence) fn_.destroy_fence(device_, f.fence, nullptr);
        if (f.acquire_semaphore && fn_.destroy_semaphore) fn_.destroy_semaphore(device_, f.acquire_semaphore, nullptr);
        if (f.render_semaphore && fn_.destroy_semaphore) fn_.destroy_semaphore(device_, f.render_semaphore, nullptr);
        if (headless_ && f.image && fn_.destroy_image) fn_.destroy_image(device_, f.image, nullptr);
        if (headless_ && f.memory && fn_.free_mem) fn_.free_mem(device_, f.memory, nullptr);
    }
    frames_.clear();
}

bool VulkanPresentationSwapchain::resize(std::uint32_t width, std::uint32_t height,
                                         VkFormat preferred_format, std::uint32_t buffer_count,
                                         bool allow_tearing) {
    std::lock_guard lock(mutex_);
    if (!device_ || !initialized_) return false;

    if (fn_.device_wait_idle) fn_.device_wait_idle(device_);

    VulkanSwapchainConfig cfg{};
    cfg.width = width;
    cfg.height = height;
    cfg.preferred_format = (preferred_format != VK_FORMAT_UNDEFINED) ? preferred_format : format_;
    cfg.color_space = color_space_;
    cfg.buffer_count = buffer_count ? buffer_count : exposed_count_;
    cfg.allow_tearing = allow_tearing;
    cfg.force_headless = headless_;

    destroy_image_views();

    if (headless_) {
        if (!create_headless_images(cfg)) return false;
    } else {
        if (!create_surface_swapchain(cfg)) {
            headless_ = true;
            if (!create_headless_images(cfg)) return false;
        }
    }

    if (!create_image_views()) return false;

    current_index_ = 0;
    acquired_index_ = 0;
    return true;
}

bool VulkanPresentationSwapchain::acquire_next_image(std::uint32_t& out_image_index,
                                                     std::uint64_t timeout_ns) {
    std::lock_guard lock(mutex_);
    if (!initialized_ || images_.empty()) return false;

    if (headless_) {
        out_image_index = current_index_;
        acquired_index_ = current_index_;
        current_index_ = (current_index_ + 1) % exposed_count_;
        return true;
    }

    if (!swapchain_ || !fn_.acquire_next_image) return false;

    const std::uint32_t slot = current_index_ % frames_.size();
    if (frames_[slot].fence && fn_.reset_fences) {
        fn_.reset_fences(device_, 1, &frames_[slot].fence);
    }

    std::uint32_t img_idx = 0;
    const VkResult res = fn_.acquire_next_image(device_, swapchain_, timeout_ns,
                                                frames_[slot].acquire_semaphore,
                                                frames_[slot].fence, &img_idx);
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) return false;

    if (frames_[slot].fence && fn_.wait_for_fences) {
        fn_.wait_for_fences(device_, 1, &frames_[slot].fence, VK_TRUE, timeout_ns);
    }

    out_image_index = img_idx;
    acquired_index_ = img_idx;
    current_index_ = (slot + 1) % frames_.size();
    return true;
}

VkResult VulkanPresentationSwapchain::present(std::uint32_t back_buffer_index,
                                              VkSemaphore wait_semaphore) {
    std::unique_lock lock(mutex_);
    if (!initialized_) return VK_ERROR_INITIALIZATION_FAILED;

    const std::uint32_t idx = (back_buffer_index < images_.size()) ? back_buffer_index : acquired_index_;
    if (idx >= images_.size()) return VK_ERROR_UNKNOWN;

    const VkImage current_img = images_[idx];
    auto callback_snapshot = frame_callback_;

    if (headless_) {
        if (capture_enabled_) capture_image_to_buffer(current_img);
        ++presentation_count_;
        lock.unlock();
        if (callback_snapshot) callback_snapshot(idx, current_img);
        return VK_SUCCESS;
    }

    if (!swapchain_ || !fn_.queue_present) return VK_ERROR_INITIALIZATION_FAILED;

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    if (wait_semaphore != VK_NULL_HANDLE) {
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &wait_semaphore;
    }
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &idx;

    const VkResult r = fn_.queue_present(queue_, &pi);
    if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) {
        if (capture_enabled_) capture_image_to_buffer(current_img);
        ++presentation_count_;
        lock.unlock();
        if (callback_snapshot) callback_snapshot(idx, current_img);
    }
    return r;
}

ResourceBinding VulkanPresentationSwapchain::resource_binding(std::uint32_t index) const {
    ResourceBinding b{};
    if (index >= images_.size()) return b;
    b.image = images_[index];
    b.format = format_;
    b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    b.width = extent_.width;
    b.height = extent_.height;
    b.mip_levels = 1;
    b.array_layers = 1;
    b.depth = 1;
    b.samples = 1;
    b.color_attachment = true;
    b.presentable = !headless_;
    return b;
}

bool VulkanPresentationSwapchain::setup_capture_buffer() {
    release_capture_buffer();
    if (!device_ || !fn_.create_buffer || !fn_.alloc_mem) return false;

    const std::size_t size = std::size_t(extent_.width) * extent_.height * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (fn_.create_buffer(device_, &bci, nullptr, &capture_buffer_) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    fn_.buffer_mem_req(device_, capture_buffer_, &req);
    const std::uint32_t mem_type = find_memory_type(req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mem_type;
    if (fn_.alloc_mem(device_, &mai, nullptr, &capture_memory_) != VK_SUCCESS) {
        fn_.destroy_buffer(device_, capture_buffer_, nullptr);
        capture_buffer_ = VK_NULL_HANDLE;
        return false;
    }
    fn_.bind_buffer_mem(device_, capture_buffer_, capture_memory_, 0);
    capture_buffer_size_ = size;
    return true;
}

void VulkanPresentationSwapchain::release_capture_buffer() {
    if (capture_buffer_ && fn_.destroy_buffer) {
        fn_.destroy_buffer(device_, capture_buffer_, nullptr);
        capture_buffer_ = VK_NULL_HANDLE;
    }
    if (capture_memory_ && fn_.free_mem) {
        fn_.free_mem(device_, capture_memory_, nullptr);
        capture_memory_ = VK_NULL_HANDLE;
    }
    capture_buffer_size_ = 0;
}

bool VulkanPresentationSwapchain::enable_capture(bool enable) {
    std::lock_guard lock(mutex_);
    capture_enabled_ = enable;
    if (enable) return setup_capture_buffer();
    release_capture_buffer();
    return true;
}

bool VulkanPresentationSwapchain::capture_image_to_buffer(VkImage src_image) {
    if (!capture_buffer_ || !cmd_buf_ || !sync_fence_ || !src_image) return false;

    if (fn_.reset_fences(device_, 1, &sync_fence_) != VK_SUCCESS) return false;
    if (fn_.reset_cmd_buf(cmd_buf_, 0) != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (fn_.begin_cmd_buf(cmd_buf_, &bi) != VK_SUCCESS) return false;

    VkImageMemoryBarrier pre{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    pre.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    pre.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    pre.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    pre.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre.image = src_image;
    pre.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    fn_.cmd_barrier(cmd_buf_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &pre);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {extent_.width, extent_.height, 1};

    fn_.cmd_copy_image_to_buffer(cmd_buf_, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 capture_buffer_, 1, &region);

    VkImageMemoryBarrier post{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    post.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    post.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    post.newLayout = headless_ ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    post.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post.image = src_image;
    post.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    fn_.cmd_barrier(cmd_buf_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &post);

    if (fn_.end_cmd_buf(cmd_buf_) != VK_SUCCESS) return false;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buf_;
    if (fn_.queue_submit(queue_, 1, &si, sync_fence_) != VK_SUCCESS) return false;
    return fn_.wait_for_fences(device_, 1, &sync_fence_, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
}

bool VulkanPresentationSwapchain::read_back(std::vector<std::uint8_t>& out_pixels,
                                             VkExtent2D* out_extent, VkFormat* out_format) {
    std::lock_guard lock(mutex_);
    if (!capture_buffer_ || !capture_memory_ || !fn_.map_mem) return false;

    void* mapped = nullptr;
    if (fn_.map_mem(device_, capture_memory_, 0, capture_buffer_size_, 0, &mapped) != VK_SUCCESS || !mapped)
        return false;

    out_pixels.resize(capture_buffer_size_);
    std::memcpy(out_pixels.data(), mapped, capture_buffer_size_);
    fn_.unmap_mem(device_, capture_memory_);

    if (out_extent) *out_extent = extent_;
    if (out_format) *out_format = format_;
    return true;
}

// -------------------------------------------------------------------------------------------------
// VulkanDevice implementation
// -------------------------------------------------------------------------------------------------

VulkanDevice::VulkanDevice() = default;

VulkanDevice::VulkanDevice(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                           VkQueue queue, std::uint32_t queue_family,
                           PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa) {
    init(instance, physical, device, queue, queue_family, gipa, gdpa);
}

VulkanDevice::~VulkanDevice() = default;

bool VulkanDevice::init(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                         VkQueue queue, std::uint32_t queue_family,
                         PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa) {
    instance_ = instance;
    physical_ = physical;
    device_ = device;
    queue_ = queue;
    queue_family_ = queue_family;
    gipa_ = gipa;
    gdpa_ = gdpa;
    loaded_ = fn_.load(gipa, instance, gdpa, device);
    return loaded_;
}

bool VulkanDevice::create_swapchain(VkSurfaceKHR surface, const VulkanSwapchainConfig& config) {
    if (!loaded_) return false;
    swapchain_ = std::make_unique<VulkanPresentationSwapchain>();
    return swapchain_->init(instance_, physical_, device_, queue_, queue_family_, surface, config, gipa_, gdpa_);
}

bool VulkanDevice::create_headless_swapchain(const VulkanSwapchainConfig& config) {
    if (!loaded_) return false;
    swapchain_ = std::make_unique<VulkanPresentationSwapchain>();
    return swapchain_->init_headless(instance_, physical_, device_, queue_, queue_family_, config, gipa_, gdpa_);
}

bool VulkanDevice::init_presentation(HWND hwnd, std::uint32_t width, std::uint32_t height,
                                     VkFormat preferred_format, std::uint32_t buffer_count,
                                     bool allow_tearing) {
    VulkanSwapchainConfig cfg{};
    cfg.width = width;
    cfg.height = height;
    cfg.preferred_format = preferred_format;
    cfg.buffer_count = buffer_count;
    cfg.allow_tearing = allow_tearing;

    if (!is_display_available() || !hwnd) {
        return create_headless_swapchain(cfg);
    }

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!create_win32_surface(instance_, hwnd, &surface)) {
        return create_headless_swapchain(cfg);
    }

    return create_swapchain(surface, cfg);
}

bool VulkanDevice::acquire_next_back_buffer(std::uint32_t& out_index, std::uint64_t timeout_ns) {
    if (!swapchain_) return false;
    return swapchain_->acquire_next_image(out_index, timeout_ns);
}

VkResult VulkanDevice::present_back_buffer(std::uint32_t back_buffer_index, VkSemaphore wait_semaphore) {
    if (!swapchain_) return VK_ERROR_INITIALIZATION_FAILED;
    return swapchain_->present(back_buffer_index, wait_semaphore);
}

bool VulkanDevice::resize_back_buffers(std::uint32_t width, std::uint32_t height,
                                      VkFormat preferred_format, std::uint32_t buffer_count,
                                      bool allow_tearing) {
    if (!swapchain_) return false;
    return swapchain_->resize(width, height, preferred_format, buffer_count, allow_tearing);
}

std::uint32_t VulkanDevice::back_buffer_count() const {
    return swapchain_ ? swapchain_->image_count() : 0;
}

std::uint32_t VulkanDevice::current_back_buffer_index() const {
    return swapchain_ ? swapchain_->current_back_buffer_index() : 0;
}

VkImage VulkanDevice::back_buffer_image(std::uint32_t index) const {
    return swapchain_ ? swapchain_->image(index) : VK_NULL_HANDLE;
}

VkImageView VulkanDevice::back_buffer_view(std::uint32_t index) const {
    return swapchain_ ? swapchain_->image_view(index) : VK_NULL_HANDLE;
}

VkExtent2D VulkanDevice::back_buffer_extent() const {
    return swapchain_ ? swapchain_->extent() : VkExtent2D{0, 0};
}

VkFormat VulkanDevice::back_buffer_format() const {
    return swapchain_ ? swapchain_->format() : VK_FORMAT_UNDEFINED;
}

bool VulkanDevice::is_headless() const {
    return swapchain_ ? swapchain_->is_headless() : true;
}

VkImageView VulkanDevice::create_image_view_2d(VkImage image, VkFormat format, VkImageAspectFlags aspect) {
    if (!fn_.create_image_view || !device_ || !image) return VK_NULL_HANDLE;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                     VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.baseMipLevel = 0;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.baseArrayLayer = 0;
    vi.subresourceRange.layerCount = 1;

    VkImageView view = VK_NULL_HANDLE;
    if (fn_.create_image_view(device_, &vi, nullptr, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
    return view;
}

void VulkanDevice::destroy_image_view(VkImageView view) {
    if (view && fn_.destroy_image_view && device_) {
        fn_.destroy_image_view(device_, view, nullptr);
    }
}

bool VulkanDevice::is_display_available() {
    const char* headless_env = std::getenv("LEONIDA_HEADLESS");
    if (headless_env && (std::strcmp(headless_env, "1") == 0 || std::strcmp(headless_env, "true") == 0)) {
        return false;
    }
#if defined(_WIN32)
    return true;
#else
    const char* disp = std::getenv("DISPLAY");
    const char* wayland = std::getenv("WAYLAND_DISPLAY");
    return (disp && disp[0] != '\0') || (wayland && wayland[0] != '\0');
#endif
}

bool VulkanDevice::create_win32_surface(VkInstance instance, HWND hwnd, VkSurfaceKHR* out_surface,
                                       PFN_vkGetInstanceProcAddr gipa) {
    if (!out_surface) return false;
    *out_surface = VK_NULL_HANDLE;
    if (!instance || !hwnd) return false;

#if defined(_WIN32) && defined(VK_USE_PLATFORM_WIN32_KHR)
    if (!gipa) gipa = load_default_gipa();
    if (!gipa) return false;
    auto create_surface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(gipa(instance, "vkCreateWin32SurfaceKHR"));
    if (!create_surface) return false;

    VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    sci.hinstance = GetModuleHandleW(nullptr);
    sci.hwnd = hwnd;
    return create_surface(instance, &sci, nullptr, out_surface) == VK_SUCCESS;
#else
    (void)instance;
    (void)hwnd;
    (void)gipa;
    return false;
#endif
}

}  // namespace leonida

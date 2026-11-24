#include "vulkan_presenter.h"

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

bool VulkanPresenterFunctions::load(PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
                                    PFN_vkGetDeviceProcAddr gdpa, VkDevice device) {
    if (!gipa) gipa = load_default_gipa();
    if (!gipa || !instance) return false;

#define LOAD_I(member, name) member = reinterpret_cast<PFN_##name>(gipa(instance, #name))
#if defined(_WIN32)
    LOAD_I(create_win32_surface, vkCreateWin32SurfaceKHR);
#endif
    LOAD_I(destroy_surface, vkDestroySurfaceKHR);
    LOAD_I(surface_support, vkGetPhysicalDeviceSurfaceSupportKHR);
    LOAD_I(surface_caps, vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    LOAD_I(surface_formats, vkGetPhysicalDeviceSurfaceFormatsKHR);
    LOAD_I(surface_present_modes, vkGetPhysicalDeviceSurfacePresentModesKHR);
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

    LOAD_D(create_image, vkCreateImage);
    LOAD_D(destroy_image, vkDestroyImage);
    LOAD_D(image_mem_req, vkGetImageMemoryRequirements);
    LOAD_D(alloc_mem, vkAllocateMemory);
    LOAD_D(free_mem, vkFreeMemory);
    LOAD_D(bind_image_mem, vkBindImageMemory);
    LOAD_D(create_image_view, vkCreateImageView);
    LOAD_D(destroy_image_view, vkDestroyImageView);

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

    return create_image && destroy_image && alloc_mem && free_mem && bind_image_mem &&
           create_image_view && destroy_image_view && create_fence && destroy_fence &&
           wait_for_fences && reset_fences && create_semaphore && destroy_semaphore &&
           queue_submit && device_wait_idle && mem_props && create_cmd_pool && destroy_cmd_pool &&
           alloc_cmd_bufs && begin_cmd_buf && end_cmd_buf && cmd_barrier;
}

VulkanPresenter::VulkanPresenter() = default;

VulkanPresenter::~VulkanPresenter() {
    destroy();
}

void VulkanPresenter::destroy() {
    std::lock_guard lock(mutex_);
    if (!device_) return;

    if (fn_.device_wait_idle) {
        fn_.device_wait_idle(device_);
    }

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

bool VulkanPresenter::is_display_available() {
#if defined(_WIN32)
    HWINSTA station = GetProcessWindowStation();
    if (!station) return false;
    USEROBJECTFLAGS flags{};
    if (!GetUserObjectInformationW(station, UOI_FLAGS, &flags, sizeof(flags), nullptr)) return false;
    return (flags.dwFlags & WSF_VISIBLE) != 0;
#else
    const char* dpy = std::getenv("DISPLAY");
    const char* wayland = std::getenv("WAYLAND_DISPLAY");
    return (dpy && dpy[0] != '\0') || (wayland && wayland[0] != '\0');
#endif
}

bool VulkanPresenter::create_win32_surface(VkInstance instance, HWND hwnd, VkSurfaceKHR* out_surface,
                                          PFN_vkGetInstanceProcAddr gipa) {
    if (!out_surface) return false;
    *out_surface = VK_NULL_HANDLE;
#if defined(_WIN32)
    if (!gipa) gipa = load_default_gipa();
    if (!gipa || !instance || !hwnd) return false;

    auto create_surface_fn = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(gipa(instance, "vkCreateWin32SurfaceKHR"));
    if (!create_surface_fn) return false;

    VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    sci.hinstance = GetModuleHandleW(nullptr);
    sci.hwnd = hwnd;
    return create_surface_fn(instance, &sci, nullptr, out_surface) == VK_SUCCESS;
#else
    (void)instance; (void)hwnd; (void)gipa;
    return false;
#endif
}

std::uint32_t VulkanPresenter::find_memory_type(std::uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props{};
    fn_.mem_props(physical_, &mem_props);
    for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if (type_filter & (1u << i)) return i;
    }
    return 0;
}

bool VulkanPresenter::init(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
                           VkQueue queue, std::uint32_t queue_family, VkSurfaceKHR surface,
                           const PresenterConfig& config,
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

    // Helper command pool and synchronization fence
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

    // Determine headless fallback
    bool use_headless = config.force_headless || (surface == VK_NULL_HANDLE) ||
                        !is_display_available() || !fn_.surface_support || !fn_.create_swapchain;

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
            // Graceful fallback to headless offscreen presentation
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

bool VulkanPresenter::init_headless(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device,
                                    VkQueue queue, std::uint32_t queue_family,
                                    const PresenterConfig& config,
                                    PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa) {
    PresenterConfig cfg = config;
    cfg.force_headless = true;
    return init(instance, physical_device, device, queue, queue_family, VK_NULL_HANDLE, cfg, gipa, gdpa);
}

bool VulkanPresenter::create_surface_swapchain(const PresenterConfig& config) {
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
        const std::uint32_t req_w = config.width ? config.width : 1280u;
        const std::uint32_t req_h = config.height ? config.height : 720u;
        ext.width = std::clamp(req_w, caps.minImageExtent.width, caps.maxImageExtent.width);
        ext.height = std::clamp(req_h, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    std::uint32_t desired_count = std::max(config.buffer_count ? config.buffer_count : 2u, caps.minImageCount);
    if (caps.maxImageCount > 0) desired_count = std::min(desired_count, caps.maxImageCount);

    VkPresentModeKHR chosen_mode = VK_PRESENT_MODE_FIFO_KHR;
    if (config.allow_tearing && fn_.surface_present_modes) {
        std::uint32_t mode_count = 0;
        fn_.surface_present_modes(physical_, surface_, &mode_count, nullptr);
        if (mode_count > 0) {
            std::vector<VkPresentModeKHR> modes(mode_count);
            fn_.surface_present_modes(physical_, surface_, &mode_count, modes.data());
            for (VkPresentModeKHR m : modes) {
                if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) {
                    chosen_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                    break;
                }
                if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
                    chosen_mode = VK_PRESENT_MODE_MAILBOX_KHR;
                }
            }
        }
    }

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface_;
    sci.minImageCount = desired_count;
    sci.imageFormat = chosen.format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = ext;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | config.extra_usage;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = chosen_mode;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = swapchain_;

    VkSwapchainKHR next_swapchain = VK_NULL_HANDLE;
    if (fn_.create_swapchain(device_, &sci, nullptr, &next_swapchain) != VK_SUCCESS) return false;

    if (swapchain_) {
        fn_.destroy_swapchain(device_, swapchain_, nullptr);
    }
    swapchain_ = next_swapchain;

    format_ = chosen.format;
    color_space_ = chosen.colorSpace;
    extent_ = ext;
    surface_extent_ = ext;
    present_mode_ = chosen_mode;
    allow_tearing_ = config.allow_tearing;

    std::uint32_t image_count = 0;
    fn_.get_images(device_, swapchain_, &image_count, nullptr);
    if (image_count == 0) return false;
    images_.resize(image_count);
    fn_.get_images(device_, swapchain_, &image_count, images_.data());

    exposed_count_ = image_count;
    image_initialized_.assign(image_count, false);

    destroy_frames();
    frames_.resize(image_count);
    for (std::uint32_t i = 0; i < image_count; ++i) {
        frames_[i].image = images_[i];
        frames_[i].index = i;
        frames_[i].memory = VK_NULL_HANDLE;

        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (fn_.create_fence(device_, &fence_info, nullptr, &frames_[i].fence) != VK_SUCCESS) return false;

        VkSemaphoreCreateInfo sem_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (fn_.create_semaphore(device_, &sem_info, nullptr, &frames_[i].acquired_semaphore) != VK_SUCCESS ||
            fn_.create_semaphore(device_, &sem_info, nullptr, &frames_[i].rendered_semaphore) != VK_SUCCESS) {
            return false;
        }
    }

    return true;
}

bool VulkanPresenter::create_headless_images(const PresenterConfig& config) {
    destroy_frames();

    format_ = config.preferred_format;
    color_space_ = config.color_space;
    extent_ = {config.width ? config.width : 1280u, config.height ? config.height : 720u};
    surface_extent_ = extent_;
    present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    allow_tearing_ = config.allow_tearing;

    const std::uint32_t count = config.buffer_count ? config.buffer_count : 2u;
    images_.resize(count, VK_NULL_HANDLE);
    frames_.resize(count);
    image_initialized_.assign(count, false);

    for (std::uint32_t i = 0; i < count; ++i) {
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

        VkMemoryRequirements req{};
        fn_.image_mem_req(device_, img, &req);

        const std::uint32_t mem_type = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = mem_type;

        VkDeviceMemory mem = VK_NULL_HANDLE;
        if (fn_.alloc_mem(device_, &mai, nullptr, &mem) != VK_SUCCESS ||
            fn_.bind_image_mem(device_, img, mem, 0) != VK_SUCCESS) {
            if (mem) fn_.free_mem(device_, mem, nullptr);
            fn_.destroy_image(device_, img, nullptr);
            return false;
        }

        images_[i] = img;
        frames_[i].image = img;
        frames_[i].memory = mem;
        frames_[i].index = i;

        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (fn_.create_fence(device_, &fence_info, nullptr, &frames_[i].fence) != VK_SUCCESS) return false;

        VkSemaphoreCreateInfo sem_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (fn_.create_semaphore(device_, &sem_info, nullptr, &frames_[i].acquired_semaphore) != VK_SUCCESS ||
            fn_.create_semaphore(device_, &sem_info, nullptr, &frames_[i].rendered_semaphore) != VK_SUCCESS) {
            return false;
        }

        // Transition headless offscreen image from UNDEFINED to GENERAL
        transition_image(img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        image_initialized_[i] = true;
    }

    exposed_count_ = count;
    return true;
}

bool VulkanPresenter::create_image_views() {
    destroy_image_views();
    image_views_.resize(images_.size(), VK_NULL_HANDLE);

    for (std::size_t i = 0; i < images_.size(); ++i) {
        VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivci.image = images_[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = format_;
        ivci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ivci.subresourceRange.baseMipLevel = 0;
        ivci.subresourceRange.levelCount = 1;
        ivci.subresourceRange.baseArrayLayer = 0;
        ivci.subresourceRange.layerCount = 1;

        if (fn_.create_image_view(device_, &ivci, nullptr, &image_views_[i]) != VK_SUCCESS) {
            destroy_image_views();
            return false;
        }
        frames_[i].view = image_views_[i];
    }
    return true;
}

void VulkanPresenter::destroy_image_views() {
    if (!fn_.destroy_image_view || !device_) return;
    for (VkImageView view : image_views_) {
        if (view) fn_.destroy_image_view(device_, view, nullptr);
    }
    image_views_.clear();
}

void VulkanPresenter::destroy_frames() {
    for (PresenterFrame& frame : frames_) {
        if (frame.fence && fn_.destroy_fence) {
            fn_.destroy_fence(device_, frame.fence, nullptr);
            frame.fence = VK_NULL_HANDLE;
        }
        if (frame.acquired_semaphore && fn_.destroy_semaphore) {
            fn_.destroy_semaphore(device_, frame.acquired_semaphore, nullptr);
            frame.acquired_semaphore = VK_NULL_HANDLE;
        }
        if (frame.rendered_semaphore && fn_.destroy_semaphore) {
            fn_.destroy_semaphore(device_, frame.rendered_semaphore, nullptr);
            frame.rendered_semaphore = VK_NULL_HANDLE;
        }
        if (headless_ && frame.memory && fn_.free_mem) {
            fn_.free_mem(device_, frame.memory, nullptr);
            frame.memory = VK_NULL_HANDLE;
        }
        if (headless_ && frame.image && fn_.destroy_image) {
            fn_.destroy_image(device_, frame.image, nullptr);
            frame.image = VK_NULL_HANDLE;
        }
    }
    frames_.clear();
}

bool VulkanPresenter::transition_image(VkImage image, VkImageLayout old_layout, VkImageLayout new_layout,
                                       VkAccessFlags src_access, VkAccessFlags dst_access,
                                       VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    if (!cmd_buf_ || !sync_fence_ || !image) return false;

    if (fn_.reset_fences(device_, 1, &sync_fence_) != VK_SUCCESS ||
        fn_.reset_cmd_buf(cmd_buf_, 0) != VK_SUCCESS) {
        return false;
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (fn_.begin_cmd_buf(cmd_buf_, &bi) != VK_SUCCESS) return false;

    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    fn_.cmd_barrier(cmd_buf_, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    if (fn_.end_cmd_buf(cmd_buf_) != VK_SUCCESS) return false;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buf_;

    if (fn_.queue_submit(queue_, 1, &si, sync_fence_) != VK_SUCCESS ||
        fn_.wait_for_fences(device_, 1, &sync_fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return false;
    }
    return true;
}

bool VulkanPresenter::acquire_next_image(std::uint32_t& out_image_index, std::uint64_t timeout_ns) {
    std::lock_guard lock(mutex_);
    if (!initialized_ || exposed_count_ == 0) return false;

    if (headless_) {
        current_index_ = (current_index_ + 1) % exposed_count_;
        acquired_index_ = current_index_;
        out_image_index = current_index_;
        return true;
    }

    if (!swapchain_) return false;

    std::uint32_t index = 0;
    PresenterFrame& frame = frames_[current_index_];
    fn_.reset_fences(device_, 1, &frame.fence);

    const VkResult res = fn_.acquire_next_image(device_, swapchain_, timeout_ns,
                                               frame.acquired_semaphore, frame.fence, &index);
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
        return false;
    }

    fn_.wait_for_fences(device_, 1, &frame.fence, VK_TRUE, timeout_ns);
    acquired_index_ = index;
    current_index_ = index;
    out_image_index = index;

    if (!image_initialized_[index]) {
        transition_image(images_[index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                         0, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        image_initialized_[index] = true;
    }

    return true;
}

bool VulkanPresenter::setup_capture_buffer() {
    release_capture_buffer();

    const std::size_t size = static_cast<std::size_t>(extent_.width) * extent_.height * 4;
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

    if (fn_.alloc_mem(device_, &mai, nullptr, &capture_memory_) != VK_SUCCESS ||
        fn_.bind_buffer_mem(device_, capture_buffer_, capture_memory_, 0) != VK_SUCCESS) {
        release_capture_buffer();
        return false;
    }

    capture_buffer_size_ = size;
    return true;
}

void VulkanPresenter::release_capture_buffer() {
    if (capture_buffer_ && fn_.destroy_buffer) {
        fn_.destroy_buffer(device_, capture_buffer_, nullptr);
        capture_buffer_ = VK_NULL_HANDLE;
    }
    if (capture_memory_ && fn_.free_mem) {
        fn_.free_mem(device_, capture_memory_, nullptr);
        capture_memory_ = VK_NULL_HANDLE;
    }
    capture_buffer_size_ = 0;
    frame_captured_ = false;
}

bool VulkanPresenter::capture_image_to_buffer(VkImage src_image) {
    if (!capture_buffer_ || !cmd_buf_ || !sync_fence_ || !src_image) return false;

    if (fn_.reset_fences(device_, 1, &sync_fence_) != VK_SUCCESS ||
        fn_.reset_cmd_buf(cmd_buf_, 0) != VK_SUCCESS) {
        return false;
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (fn_.begin_cmd_buf(cmd_buf_, &bi) != VK_SUCCESS) return false;

    VkImageMemoryBarrier pre_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    pre_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    pre_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    pre_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    pre_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_barrier.image = src_image;
    pre_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    fn_.cmd_barrier(cmd_buf_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &pre_barrier);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = extent_.width;
    region.bufferImageHeight = extent_.height;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {extent_.width, extent_.height, 1};

    fn_.cmd_copy_image_to_buffer(cmd_buf_, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 capture_buffer_, 1, &region);

    VkImageMemoryBarrier post_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    post_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    post_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    post_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    post_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    post_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post_barrier.image = src_image;
    post_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    fn_.cmd_barrier(cmd_buf_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &post_barrier);

    if (fn_.end_cmd_buf(cmd_buf_) != VK_SUCCESS) return false;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buf_;

    if (fn_.queue_submit(queue_, 1, &si, sync_fence_) != VK_SUCCESS ||
        fn_.wait_for_fences(device_, 1, &sync_fence_, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return false;
    }

    captured_extent_ = extent_;
    captured_format_ = format_;
    frame_captured_ = true;
    return true;
}

bool VulkanPresenter::enable_capture(bool enable) {
    std::lock_guard lock(mutex_);
    if (!initialized_) return false;
    capture_enabled_ = enable;
    if (enable) {
        return setup_capture_buffer();
    } else {
        release_capture_buffer();
        return true;
    }
}

bool VulkanPresenter::read_back(std::vector<std::uint8_t>& out_pixels, VkExtent2D* out_extent,
                                VkFormat* out_format) {
    std::lock_guard lock(mutex_);
    if (!capture_enabled_ || !capture_memory_ || !frame_captured_) return false;

    void* mapped = nullptr;
    if (fn_.map_mem(device_, capture_memory_, 0, capture_buffer_size_, 0, &mapped) != VK_SUCCESS || !mapped) {
        return false;
    }

    out_pixels.assign(static_cast<const std::uint8_t*>(mapped),
                      static_cast<const std::uint8_t*>(mapped) + capture_buffer_size_);
    fn_.unmap_mem(device_, capture_memory_);

    if (out_extent) *out_extent = captured_extent_;
    if (out_format) *out_format = captured_format_;
    return true;
}

VkResult VulkanPresenter::present(std::uint32_t back_buffer_index, VkSemaphore wait_semaphore) {
    std::lock_guard lock(mutex_);
    if (!initialized_ || back_buffer_index >= exposed_count_) return VK_ERROR_INITIALIZATION_FAILED;

    const VkImage current_img = images_[back_buffer_index];

    if (capture_enabled_) {
        capture_image_to_buffer(current_img);
    }

    if (frame_callback_) {
        frame_callback_(back_buffer_index, current_img);
    }

    if (headless_) {
        ++presentation_count_;
        return VK_SUCCESS;
    }

    if (!swapchain_ || !fn_.queue_present) return VK_ERROR_INITIALIZATION_FAILED;

    // Transition image to PRESENT_SRC_KHR if not already there
    transition_image(current_img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    if (wait_semaphore != VK_NULL_HANDLE) {
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &wait_semaphore;
    }
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &back_buffer_index;

    const VkResult res = fn_.queue_present(queue_, &pi);
    if (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR) {
        ++presentation_count_;
        return VK_SUCCESS;
    }
    return res;
}

VkResult VulkanPresenter::present_blank(bool vsync, bool tear) {
    (void)vsync; (void)tear;
    std::lock_guard lock(mutex_);
    if (!initialized_ || current_index_ >= exposed_count_) return VK_ERROR_INITIALIZATION_FAILED;

    const VkImage img = images_[current_index_];
    if (cmd_buf_ && sync_fence_) {
        fn_.reset_fences(device_, 1, &sync_fence_);
        fn_.reset_cmd_buf(cmd_buf_, 0);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (fn_.begin_cmd_buf(cmd_buf_, &bi) == VK_SUCCESS) {
            VkClearColorValue black{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = img;
            barrier.subresourceRange = range;

            fn_.cmd_barrier(cmd_buf_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 0, nullptr, 1, &barrier);

            fn_.cmd_clear_color(cmd_buf_, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

            fn_.cmd_barrier(cmd_buf_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            0, 0, nullptr, 0, nullptr, 1, &barrier);

            fn_.end_cmd_buf(cmd_buf_);

            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd_buf_;
            fn_.queue_submit(queue_, 1, &si, sync_fence_);
            fn_.wait_for_fences(device_, 1, &sync_fence_, VK_TRUE, UINT64_MAX);
        }
    }

    return present(current_index_);
}

VkResult VulkanPresenter::present_blit(VkImage src_image, std::uint32_t src_width, std::uint32_t src_height,
                                      VkFormat src_format, bool vsync, bool tear) {
    (void)src_format; (void)vsync; (void)tear;
    std::lock_guard lock(mutex_);
    if (!initialized_ || current_index_ >= exposed_count_ || !src_image) return VK_ERROR_INITIALIZATION_FAILED;

    const VkImage dst_image = images_[current_index_];

    if (cmd_buf_ && sync_fence_) {
        fn_.reset_fences(device_, 1, &sync_fence_);
        fn_.reset_cmd_buf(cmd_buf_, 0);

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (fn_.begin_cmd_buf(cmd_buf_, &bi) == VK_SUCCESS) {
            VkImageMemoryBarrier pre[2]{};
            pre[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            pre[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            pre[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            pre[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            pre[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            pre[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre[0].image = src_image;
            pre[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            pre[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            pre[1].srcAccessMask = 0;
            pre[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            pre[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            pre[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            pre[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre[1].image = dst_image;
            pre[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            fn_.cmd_barrier(cmd_buf_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 0, nullptr, 2, pre);

            VkImageBlit blit_region{};
            blit_region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit_region.srcOffsets[1] = {static_cast<std::int32_t>(src_width),
                                         static_cast<std::int32_t>(src_height), 1};
            blit_region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit_region.dstOffsets[1] = {static_cast<std::int32_t>(extent_.width),
                                         static_cast<std::int32_t>(extent_.height), 1};

            fn_.cmd_blit(cmd_buf_, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit_region, VK_FILTER_LINEAR);

            VkImageMemoryBarrier post[2]{};
            post[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            post[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            post[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            post[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            post[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            post[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post[0].image = src_image;
            post[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            post[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            post[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            post[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
            post[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            post[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            post[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post[1].image = dst_image;
            post[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

            fn_.cmd_barrier(cmd_buf_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            0, 0, nullptr, 0, nullptr, 2, post);

            fn_.end_cmd_buf(cmd_buf_);

            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd_buf_;
            fn_.queue_submit(queue_, 1, &si, sync_fence_);
            fn_.wait_for_fences(device_, 1, &sync_fence_, VK_TRUE, UINT64_MAX);
        }
    }

    return present(current_index_);
}

bool VulkanPresenter::resize(std::uint32_t width, std::uint32_t height, VkFormat preferred_format,
                             std::uint32_t buffer_count, bool allow_tearing) {
    std::lock_guard lock(mutex_);
    if (!initialized_) return false;

    if (fn_.device_wait_idle) {
        fn_.device_wait_idle(device_);
    }

    PresenterConfig config{};
    config.width = width;
    config.height = height;
    config.preferred_format = (preferred_format != VK_FORMAT_UNDEFINED) ? preferred_format : format_;
    config.color_space = color_space_;
    config.buffer_count = buffer_count ? buffer_count : exposed_count_;
    config.allow_tearing = allow_tearing;
    config.force_headless = headless_;

    destroy_image_views();

    if (headless_) {
        if (!create_headless_images(config)) return false;
    } else {
        if (!create_surface_swapchain(config)) {
            headless_ = true;
            if (!create_headless_images(config)) return false;
        }
    }

    if (!create_image_views()) return false;

    if (capture_enabled_) {
        setup_capture_buffer();
    }

    current_index_ = 0;
    acquired_index_ = 0;
    return true;
}

ResourceBinding VulkanPresenter::resource_binding(std::uint32_t index) const {
    std::lock_guard lock(mutex_);
    ResourceBinding binding{};
    if (index >= images_.size()) return binding;

    binding.image = images_[index];
    binding.format = format_;
    binding.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    binding.width = extent_.width;
    binding.height = extent_.height;
    binding.mip_levels = 1;
    binding.array_layers = 1;
    binding.samples = 1;
    binding.sampled = true;
    binding.presentable = !headless_;
    return binding;
}

}  // namespace leonida

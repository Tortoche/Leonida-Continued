// Which pipelines LEONIDA_ASYNC_PIPELINES=1 builds asynchronously, by size (LEONIDA_ASYNC_MIN_BYTES). Pure stdlib, so
// async_pipeline_size_test checks it without a device; shims/d3d12_pipeline_compile.h reads the variables.
//
// An asynchronous pipeline that is not built yet has its draws and dispatches skipped until the compile worker has
// it (queue_build in shims/d3d12_vulkan_device.cpp). With every pipeline asynchronous the small lighting and post
// programs are skipped with the large ones, and the screen can stay black for as long as anything is compiling; with
// none, a submission waits minutes for each large ray-tracing program. LEONIDA_ASYNC_MIN_BYTES=<n> splits the two: a
// pipeline with less than n bytes of SPIR-V (compute: its module; graphics: all its stages) is built where it is
// needed, as without LEONIDA_ASYNC_PIPELINES, and only one of n bytes or more is asynchronous.
#pragma once

#include <cstdint>

namespace leonida::async_pipelines {

// LEONIDA_ASYNC_MIN_BYTES as text: decimal bytes. 0 (every pipeline asynchronous, as without the variable) for null,
// empty, "0" or text that is not a whole decimal number; `valid` (when given) says whether the text was usable, so the
// caller can say that it ignored it.
inline std::uint64_t parse_min_bytes(const char* text, bool* valid = nullptr) {
    if (valid) *valid = true;
    if (!text || !*text) return 0;
    std::uint64_t value = 0;
    for (const char* c = text; *c; ++c) {
        const std::uint64_t digit = std::uint64_t(*c - '0');
        if (*c < '0' || *c > '9' || value > (UINT64_MAX - digit) / 10) {  // not a digit, or over 2^64 - 1
            if (valid) *valid = false;
            return 0;
        }
        value = value * 10 + digit;
    }
    return value;
}

// Whether a pipeline of `spirv_bytes` is built asynchronously: only with asynchronous builds on, and then when no
// threshold is set (0) or the pipeline is at least that large.
inline bool asynchronous(bool async_on, std::uint64_t min_bytes, std::uint64_t spirv_bytes) {
    return async_on && (min_bytes == 0 || spirv_bytes >= min_bytes);
}

// Effective LEONIDA_COMPUTE_WARMUP_LIVE_MAX_BYTES threshold:
// - When `warmup_live_max_text` is non-empty and valid (including "0"), that explicit value is used.
// - Otherwise (unset, empty, or invalid), when asynchronous builds are on with a non-zero `async_min_bytes`,
//   defaults to `async_min_bytes` so large compute warm-ups wait for quiet while the title submits; else 0.
inline std::uint64_t effective_warmup_live_max_bytes(const char* warmup_live_max_text,
                                                     bool async_on,
                                                     std::uint64_t async_min_bytes,
                                                     bool* valid = nullptr) {
    if (valid) *valid = true;
    if (warmup_live_max_text && *warmup_live_max_text) {
        bool parsed_ok = true;
        const std::uint64_t explicit_limit = parse_min_bytes(warmup_live_max_text, &parsed_ok);
        if (valid) *valid = parsed_ok;
        if (parsed_ok) return explicit_limit;
    }
    return async_on ? async_min_bytes : 0;
}

}  // namespace leonida::async_pipelines

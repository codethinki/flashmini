/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <ostream>
#include <span>
#include <string>
#include <unordered_set>

#include "flashlight/fl/common/Defines.h"

namespace fl {

/**
 * A runtime type for various device types.
 * NOTE update `fl::getAllDeviceTypes` after changing enum values.
 */
enum class DeviceType {
    DEVICE_TYPES_FIRST,
    x64 = DEVICE_TYPES_FIRST,
    CUDA,
    DEVICE_TYPES_SIZE,
};

namespace detail {
    [[nodiscard]] constexpr auto to_index(DeviceType t) { return static_cast<std::underlying_type_t<DeviceType>>(t); }

    [[nodiscard]] constexpr auto device_types_size() { return to_index(DeviceType::DEVICE_TYPES_SIZE); }

    constexpr std::array DEVICE_TYPES = [] {
        std::array<DeviceType, static_cast<size_t>(DeviceType::DEVICE_TYPES_SIZE)> types{};

        for(auto i = to_index(DeviceType::DEVICE_TYPES_FIRST); i < types.size(); i++)
            types[i] = static_cast<DeviceType>(i);

        return types;
    }();
}


/**
 * Gets string representation of device type
 *
 * @return std::string_view to constexpr string literal
 */
[[nodiscard]] FL_API constexpr std::string_view to_string(DeviceType e) {
    switch(e) {
        case DeviceType::x64: return "x64";
        case DeviceType::CUDA: return "CUDA";
        default: return "unknown";
    }
}

#if FL_BACKEND_CUDA
constexpr DeviceType kDefaultDeviceType = DeviceType::CUDA;
#else
constexpr DeviceType kDefaultDeviceType = DeviceType::x64;
#endif

/**
 * @deprecated use @ref fl::to_string(DeviceType) instead
 */
FL_API inline std::string deviceTypeToString(DeviceType const type) { return std::string{to_string(type)}; }



/**
 * Output a string representation of `type` to `os`.
 */
FL_API inline std::ostream& operator<<(std::ostream& os, DeviceType const& type) { return (os << to_string(type)); }

/**
 * Returns all device types.
 *
 * @return span of immutable device types.
 */
[[nodiscard]] FL_API constexpr std::span<DeviceType const> device_types() { return detail::DEVICE_TYPES; }

/**
 * @deprecated use @ref device_types() instead
 */
FL_API std::unordered_set<DeviceType> const& getDeviceTypes();


} // namespace fl

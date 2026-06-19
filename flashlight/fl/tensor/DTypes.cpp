/*
 * SPDX-License-Identifier: MIT
 *
 * Original code: Copyright (c) Meta Platforms, Inc. (see FLASHLIGHT_LICENSE)
 * Modifications: Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */

#include "flashlight/fl/tensor/Types.h"

#include <stdexcept>
#include <unordered_map>

namespace fl {

auto const STRING_DTYPE_MAP = [] {
    std::unordered_map<std::string_view, dtype> map{};
    map.reserve(detail::DTYPES_SIZE);
    for(size_t i = 0; i < detail::DTYPES_SIZE; i++) {
        auto type = static_cast<dtype>(i);
        map.emplace(to_string(type), type);
    }
    return map;
}();


std::optional<fl::dtype> dtype_from_string(std::string_view str) {
    auto const it = STRING_DTYPE_MAP.find(str);

    if(it == STRING_DTYPE_MAP.end())
        return {};

    return {it->second};
}



} // namespace fl

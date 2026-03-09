/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/runtime/DeviceType.h"

namespace fl {

std::unordered_set<DeviceType> const& getDeviceTypes() {
    static std::unordered_set<DeviceType> types = {
        DeviceType::x64,
        DeviceType::CUDA
    };
    return types;
}

} // namespace fl

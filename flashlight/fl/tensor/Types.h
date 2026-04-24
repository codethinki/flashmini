/*
 * SPDX-License-Identifier: MIT
 *
 * Original code: Copyright (c) Meta Platforms, Inc. (see FLASHLIGHT_LICENSE)
 * Modifications: Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */
#pragma once

#include "DTypes.h"
#include "Traits.h"

namespace fl {
template<class T>
concept not_void = !std::is_void_v<std::decay_t<T>>;
}

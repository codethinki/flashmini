/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <af/array.h>

#include <stdexcept>
#include <vector>

namespace fl {
namespace detail {

    void advancedIndex(
        af::array const& inp,
        af::dim4 const& idxStart,
        af::dim4 const& idxEnd,
        af::dim4 const& outDims,
        std::vector<af::array> const& idxArr,
        af::array& out
    ) { throw std::runtime_error("gradAdvancedIndex not implemented for cpu"); }

} // namespace detail
} // namespace fl

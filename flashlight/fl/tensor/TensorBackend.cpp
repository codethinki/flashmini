/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {
namespace detail {

    bool areBackendsEqual(Tensor const& a, Tensor const& b) { return a.backendType() == b.backendType(); }

} // namespace detail

bool TensorBackend::isDataTypeSupported(fl::dtype const& dtype) const {
    bool supported = this->supportsDataType(dtype);
    for(auto& p : extensions_)
        supported &= p.second->isDataTypeSupported(dtype);
    return supported;
}

Tensor TensorBackend::clip(
    Tensor const& tensor,
    Tensor const& low,
    double const& high
) {
    return clip(
        tensor,
        low,
        full(tensor.shape(), high, tensor.type())
    );
}

Tensor TensorBackend::clip(
    Tensor const& tensor,
    double const& low,
    Tensor const& high
) {
    return clip(
        tensor,
        // TODO review, truncated to float in original impl
        full(tensor.shape(), low, tensor.type()),
        high
    );
}

Tensor TensorBackend::clip(
    Tensor const& tensor,
    double const& low,
    double const& high
) {
    return clip(
        tensor,
        // TODO review, truncated to float in original impl
        full(tensor.shape(), low, tensor.type()),
        full(tensor.shape(), high, tensor.type())
    );
}

Tensor TensorBackend::where(
    Tensor const& condition,
    Tensor const& x,
    double const& y
) { return where(condition, x, full(condition.shape(), y, x.type())); }

Tensor TensorBackend::where(
    Tensor const& condition,
    double const& x,
    Tensor const& y
) { return where(condition, full(condition.shape(), x, y.type()), y); }

Tensor TensorBackend::minimum(Tensor const& lhs, double const& rhs) {
    return minimum(lhs, full(lhs.shape(), rhs, lhs.type()));
}

Tensor TensorBackend::minimum(double const& lhs, Tensor const& rhs) {
    return minimum(full(rhs.shape(), lhs, rhs.type()), rhs);
}

Tensor TensorBackend::maximum(Tensor const& lhs, double const& rhs) {
    return maximum(lhs, full(lhs.shape(), rhs, lhs.type()));
}

Tensor TensorBackend::maximum(double const& lhs, Tensor const& rhs) {
    return maximum(full(rhs.shape(), lhs, rhs.type()), rhs);
}

Tensor TensorBackend::power(Tensor const& lhs, double const& rhs) {
    return power(lhs, full(lhs.shape(), rhs, lhs.type()));
}

Tensor TensorBackend::power(double const& lhs, Tensor const& rhs) {
    return power(full(rhs.shape(), lhs, rhs.type()), rhs);
}

} // namespace fl

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/arith.h>
#include <af/data.h>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

namespace fl {

Tensor ArrayFireBackend::exp(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::exp(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::log(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::log(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::negative(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(-toArray(tensor), tensor.ndim());
}

Tensor ArrayFireBackend::logicalNot(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(!toArray(tensor), tensor.ndim());
}

Tensor ArrayFireBackend::log1p(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::log1p(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sin(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::sin(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::cos(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::cos(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sqrt(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::sqrt(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::tanh(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::tanh(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::floor(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::floor(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::ceil(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::ceil(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::rint(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::round(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::absolute(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::abs(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sigmoid(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::sigmoid(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::erf(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::erf(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::flip(Tensor const& tensor, unsigned const dim) {
    return toTensor<ArrayFireTensor>(
        af::flip(toArray(tensor), dim),
        tensor.ndim()
    );
}

Tensor ArrayFireBackend::clip(
    Tensor const& tensor,
    Tensor const& low,
    Tensor const& high
) {
    return toTensor<ArrayFireTensor>(
        af::clamp(toArray(tensor), toArray(low), toArray(high)),
        tensor.ndim()
    );
}

Tensor ArrayFireBackend::roll(
    Tensor const& tensor,
    int const shift,
    unsigned const axis
) {
    if(axis > AF_MAX_DIMS)
        throw std::invalid_argument(
            "ArrayFireBackend::roll - given axis > 3 - unsupported"
        );
    std::vector<Dim> shifts(AF_MAX_DIMS, 0);
    shifts[axis] = shift;
    return toTensor<ArrayFireTensor>(
        af::shift(toArray(tensor), shifts[0], shifts[1], shifts[2], shifts[3]),
        tensor.ndim()
    );
}

Tensor ArrayFireBackend::isnan(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::isNaN(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::isinf(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(af::isInf(toArray(tensor)), tensor.ndim());
}

Tensor ArrayFireBackend::sign(Tensor const& tensor) {
    auto wSigned = 1 - 2 * af::sign(toArray(tensor));
    wSigned(toArray(tensor) == 0) = 0;
    return toTensor<ArrayFireTensor>(std::move(wSigned), tensor.ndim());
}

Tensor ArrayFireBackend::tril(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(
        af::lower(toArray(tensor), /* is_unit_diag = */ false),
        tensor.ndim()
    );
}

Tensor ArrayFireBackend::triu(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(
        af::upper(toArray(tensor), /* is_unit_diag = */ false),
        tensor.ndim()
    );
}
} // namespace fl

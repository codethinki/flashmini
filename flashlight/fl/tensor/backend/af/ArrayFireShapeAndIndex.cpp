/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/blas.h>

#include <numeric>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"
#include <span>

namespace fl {

Tensor ArrayFireBackend::reshape(Tensor const& tensor, Shape const& shape) {
    return toTensor<ArrayFireTensor>(
        af::moddims(toArray(tensor), detail::flToAfDims(shape)),
        shape.ndim()
    );
}

Tensor ArrayFireBackend::transpose(
    Tensor const& tensor,
    Shape const& axes /* = {} */
) {
    if(tensor.ndim() == 1)
        return tensor;
    else if(
        tensor.ndim() == 2 && (axes.ndim() == 0 || axes == Shape({1, 0})))
        // fastpath for matrices
        return toTensor<ArrayFireTensor>(
            af::transpose(toArray(tensor)),
            tensor.ndim()
        );
    else if(axes.ndim() == 0) {
        std::vector<Dim> dims(AF_MAX_DIMS);
        std::iota(std::begin(dims), std::end(dims), 0);
        // Compute the reversed dimensions for as many ndims as are in the input
        for(unsigned i = 0; i < tensor.ndim(); ++i)
            dims[i] = tensor.ndim() - 1 - i;

        // flip all dimensions
        return toTensor<ArrayFireTensor>(
            af::reorder(toArray(tensor), dims[0], dims[1], dims[2], dims[3]),
            tensor.ndim()
        );
    }
    else {
        if(axes.ndim() > AF_MAX_DIMS)
            throw std::invalid_argument(
                "ArrayFire tensor transpose was given "
                "permutation dims with > 4 axes"
            );
        if(axes.ndim() != tensor.ndim())
            throw std::invalid_argument(
                "ArrayFire tensor transpose axes don't match tensor's for "
                "permutation - axes must have the same number of "
                "dimensions as the tensor"
            );
        // reorder based on specified dimensions
        std::vector<dim_t> d(AF_MAX_DIMS);
        std::iota(std::begin(d), std::end(d), 0);
        for(size_t i = 0; i < axes.ndim(); ++i) {
            if(axes[i] > tensor.ndim() - 1)
                throw std::invalid_argument(
                    "ArrayFireBackend::transpose - given dimension is larger "
                    "than the number of dimensions in the tensor"
                );

            d[i] = axes[i];
        }
        return toTensor<ArrayFireTensor>(
            af::reorder(toArray(tensor), d[0], d[1], d[2], d[3]),
            tensor.ndim()
        );
    }
}

Tensor ArrayFireBackend::tile(Tensor const& tensor, Shape const& shape) {
    return toTensor<ArrayFireTensor>(
        af::tile(toArray(tensor), detail::flToAfDims(shape)),
        // TODO: check
        std::max(tensor.ndim(), shape.ndim())
    );
}


namespace {

    af::array join_chunk(std::span<af::array const> chunk, unsigned const axis) {
        switch(chunk.size()) {
            case 0: throw std::invalid_argument{"Cannot concatenate empty chunk"};
            case 1: return chunk[0];
            case 2: return af::join(axis, chunk[0], chunk[1]);
            case 3: return af::join(axis, chunk[0], chunk[1], chunk[2]);
            case 4: return af::join(axis, chunk[0], chunk[1], chunk[2], chunk[3]);
            default: {
                std::vector<af_array> handles{};
                handles.reserve(chunk.size());
                for(auto const& arr : chunk)
                    handles.push_back(arr.get());

                af_array outHandle = nullptr;
                AF_CHECK(af_join_many(&outHandle, axis, chunk.size(), handles.data()));
                return af::array{outHandle};
            }
        }
    }

} // namespace

Tensor ArrayFireBackend::concatenate(
    std::vector<Tensor> const& tensors,
    unsigned axis
) {
    if(tensors.empty())
        return toTensor<ArrayFireTensor>(); // empty tensor

    //TODO use std::from_range and views::transform once c++23
    std::vector<af::array> arrays{};
    arrays.reserve(tensors.size());


    for(auto const& t : tensors){
        arrays.push_back(toArray(t));
    }
    constexpr size_t maxChunkSize = 10; //https://arrayfire.org/docs/group__manip__func__join.htm

    //greedy chunk and join
    while(arrays.size() > 1) {
        size_t const chunks = (arrays.size() + maxChunkSize - 1) / maxChunkSize;

        for(size_t i = 0; i < chunks; i++) {
            auto const begin = i * maxChunkSize;
            auto const size = std::min<size_t>(maxChunkSize, arrays.size() - begin);

            arrays[i] = join_chunk({&arrays[begin], size}, axis);
        }
        arrays.resize(chunks);
    }

    unsigned numDims = tensors[0].ndim();
    if(axis >= numDims)
        numDims = axis + 1;

    // All tensors have the same numdims else AF would throw
    return toTensor<ArrayFireTensor>(std::move(arrays[0]), numDims);
}

Tensor ArrayFireBackend::nonzero(Tensor const& tensor) {
    return toTensor<ArrayFireTensor>(
        af::where(toArray(tensor)),
        /* numDims = */
        1
    );
}

Tensor ArrayFireBackend::pad(
    Tensor const& input,
    std::vector<std::pair<Dim, Dim>> const& padWidths,
    PadType const type
) {
    if(padWidths.size() > AF_MAX_DIMS)
        throw std::invalid_argument(
            "ArrayFireBackend::pad - given padWidths for more than 4 dimensions"
        );

    // convert ((begin_1, end_1), ..., (begin_k, end_k)) to ((begin_1, ...,
    // begin_k), (end_1, ..., end_k)) for ArrayFire
    af::dim4 beginPadding, endPadding;
    for(size_t i = 0; i < padWidths.size(); ++i) {
        auto& [first, second] = padWidths[i];
        beginPadding[i] = first;
        endPadding[i] = second;
    }

    return toTensor<ArrayFireTensor>(
        af::pad(
            toArray(input),
            beginPadding,
            endPadding,
            detail::flToAfPadType(type)
        ),
        /* numDims = */
        // TODO: check
        std::max(input.ndim(), padWidths.size())
    );
}
} // namespace fl

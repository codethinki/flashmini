/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/Utils.h"

#include <stdexcept>
#include <unordered_map>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

namespace fl::detail {

af::dtype flToAfType(fl::dtype type) {
    static std::unordered_map<fl::dtype, af::dtype> const
        kFlashlightTypeToArrayFire = {
            {fl::dtype::f16, af::dtype::f16},
            {fl::dtype::f32, af::dtype::f32},
            {fl::dtype::f64, af::dtype::f64},
            {fl::dtype::b8, af::dtype::b8},
            {fl::dtype::s16, af::dtype::s16},
            {fl::dtype::s32, af::dtype::s32},
            {fl::dtype::s64, af::dtype::s64},
            {fl::dtype::u8, af::dtype::u8},
            {fl::dtype::u16, af::dtype::u16},
            {fl::dtype::u32, af::dtype::u32},
            {
                fl::dtype::u64,
                af::dtype::u64
            }
        };
    return kFlashlightTypeToArrayFire.at(type);
}

fl::dtype afToFlType(af::dtype type) {
    static std::unordered_map<af::dtype, fl::dtype> const
        kArrayFireTypeToFlashlight = {
            {af::dtype::f16, fl::dtype::f16},
            {af::dtype::f32, fl::dtype::f32},
            {af::dtype::f64, fl::dtype::f64},
            {af::dtype::b8, fl::dtype::b8},
            {af::dtype::s16, fl::dtype::s16},
            {af::dtype::s32, fl::dtype::s32},
            {af::dtype::s64, fl::dtype::s64},
            {af::dtype::u8, fl::dtype::u8},
            {af::dtype::u16, fl::dtype::u16},
            {af::dtype::u32, fl::dtype::u32},
            {
                af::dtype::u64,
                fl::dtype::u64
            }
        };
    return kArrayFireTypeToFlashlight.at(type);
}

af_mat_prop flToAfMatrixProperty(MatrixProperty property) {
    switch(property) {
        case MatrixProperty::None: return AF_MAT_NONE;
        case MatrixProperty::Transpose: return AF_MAT_TRANS;
        default: throw std::invalid_argument(
                "flToAfMatrixProperty: invalid property specified"
            );
    }
}

af_storage flToAfStorageType(StorageType storageType) {
    switch(storageType) {
        case StorageType::Dense: return AF_STORAGE_DENSE;
        case StorageType::CSR: return AF_STORAGE_CSR;
        case StorageType::CSC: return AF_STORAGE_CSC;
        case StorageType::COO: return AF_STORAGE_COO;
        default: throw std::invalid_argument(
                "flToAfStorageType: Flashlight storage type "
                "doesn't have an ArrayFire analog"
            );
    }
}

af_topk_function flToAfTopKSortMode(SortMode sortMode) {
    switch(sortMode) {
        case SortMode::Descending: return AF_TOPK_MAX;
        case SortMode::Ascending: return AF_TOPK_MIN;
        default: throw std::invalid_argument(
                "flToAfTopKSortMode: sort mode with no ArrayFire analog specified"
            );
    }
}

af::dim4 flToAfDims(Shape const& shape) {
    if(shape.ndim() > 4)
        throw std::invalid_argument(
            "flToAfDims: ArrayFire shapes can't be more than 4 dimensions"
        );

    af::dim4 out(1, 1, 1, 1);
    for(size_t i = 0; i < shape.ndim(); ++i)
        out.dims[i] = shape.dim(i);
    return out;
}

void afToFlDims(af::dim4 const& d, unsigned const numDims, Shape& s) {
    if(numDims > AF_MAX_DIMS)
        throw std::invalid_argument("afToFlDims - numDims > AF_MAX_DIMS");

    auto& storage = s.get();

    // numdims constraint is enforced by the internal API per condenseDims
    if(numDims == 1 && d.elements() == 0) {
        // Empty tensor
        storage.resize(1);
        s[0] = 0;
        return;
    }

    // numDims == 0 --> scalar tensor
    if(numDims == 0) {
        storage.resize(0);
        return;
    }

    storage.resize(numDims);
    for(unsigned i = 0; i < numDims; ++i)
        s[i] = d[i];
}

Shape afToFlDims(af::dim4 const& d, unsigned const numDims) {
    Shape s;
    afToFlDims(d, numDims, s);
    return s;
}

af::seq flRangeToAfSeq(fl::range const& range) {
    int const start = range.start();
    auto const& optEnd = range.end();
    int const end = optEnd.has_value() ? optEnd.value() - 1 : af::end;
    // There could be have other empty sequence representations, e.g., (0, -1)
    // for axis with 1 element. In those cases, AF will throw internally --
    // we can't throw here because these cases  axis-size dependent.
    if(optEnd.has_value() && optEnd.value() == start)
        throw std::runtime_error(
            "flRangeToAfSeq: AF seq can't represent empty sequence"
        );
    return af::seq(start, end, range.stride());
}

af::index flToAfIndex(fl::Index const& idx) {
    switch(idx.type()) {
        case IndexType::Tensor: return af::index(toArray(idx.get<Tensor>()));
        case IndexType::Span: return af::index(af::span);
        case IndexType::Range: return af::index(flRangeToAfSeq(idx.get<range>()));
        case IndexType::Literal: return af::index(idx.get<Dim>());
        default: throw std::invalid_argument(
                "flToAfIndex: fl::Index has unknown or invalid type."
            );
    }
}

af::dim4 condenseDims(af::dim4 const& dims) {
    if(dims.elements() == 0)
        return af::dim4(0);

    // Find the condensed shape
    af::dim4 newDims(1, 1, 1, 1);
    unsigned newDimIdx = 0;
    for(unsigned i = 0; i < AF_MAX_DIMS; ++i)
        if(dims[i] != 1) {
            // found a non-1 dim size - populate newDims
            newDims[newDimIdx] = dims[i];
            newDimIdx++;
        }
    return newDims;
}

af::array condenseIndices(
    af::array const& arr,
    bool const keepDims /* = false */,
    std::optional<std::vector<detail::IndexType>> const& indexTypes /* = {} */,
    bool const isFlat /* = false */
) {
    // Fast path - return the Array as is if keepDims - don't consolidate
    if(keepDims)
        return arr;
    // Fast path - Array has zero elements or a dim of size zero
    if(arr.elements() == 0)
        return arr;

    af::dim4 const& dims = arr.dims();
    af::dim4 newDims{1, 1, 1, 1};
    unsigned newDimIdx = 0;
    for(unsigned i = 0; i < AF_MAX_DIMS; ++i) {
        // If we're doing an index op (indexTypes is non-empty), then only collapse
        // the dimension if it contains an index literal and we aren't doing flat
        // indexing (which collapses all dims)
        if(
            dims[i] == 1 && indexTypes && indexTypes.value().size() > i
            && indexTypes.value()[i] != detail::IndexType::Literal && !isFlat
        ) {
            newDims[newDimIdx] = 1;
            newDimIdx++;
        }
        else if(dims[i] != 1) {
            // found a non-1 dim size - populate newDims.
            newDims[newDimIdx] = dims[i];
            newDimIdx++;
        }
    }

    // Only change dims if condensing is possible
    if(newDims != arr.dims())
        return af::moddims(arr, newDims);


    return arr;
}

af_source flToAfLocation(Location location) {
    switch(location) {
        case Location::Host: return afHost;
        case Location::Device: return afDevice;
        default: throw std::invalid_argument{
                "flToAfLocation: no valid ArrayFire location exists "
                " for given Flashlight location."
            };
    }
}

af::array fromFlData(
    Shape const& shape,
    void const* ptr,
    fl::dtype type,
    fl::Location memoryLocation
) {
    auto dims = detail::flToAfDims(shape);
    auto const afType = detail::flToAfType(type);
    af_source loc = detail::flToAfLocation(memoryLocation);

    // No or null buffer
    if(!ptr)
        return af::array{dims, afType};


    using af_fundamental_types = std::tuple<
        float,
        double,
        char,
        signed short,
        int,
        long long,
        unsigned char,
        unsigned short,
        unsigned int,
        unsigned long long>;

    return dispatch_dtype<af::array, af_fundamental_types>(
        type,
        [&]<class T>() { return af::array{dims, static_cast<T const*>(ptr), loc}; }
    );
}

af_border_type flToAfPadType(PadType type) {
    switch(type) {
        case PadType::Constant: return AF_PAD_ZERO; // constant padding --> zero padding in AF
        case PadType::Edge: return AF_PAD_CLAMP_TO_EDGE;
        case PadType::Symmetric: return AF_PAD_SYM;
        default: throw std::invalid_argument(
                "flToAfPadType: Flashlight padding "
                "type not supported by ArrayFire"
            );
    }
}

} // namespace fl

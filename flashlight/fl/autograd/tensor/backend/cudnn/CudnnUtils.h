/*
 * SPDX-License-Identifier: MIT
 *
 * Original code: Copyright (c) Meta Platforms, Inc. (see FLASHLIGHT_LICENSE)
 * Modifications: Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */
#pragma once

#include <cudnn.h>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/tensor/TensorBase.h"

#include <span>

namespace fl {

class TensorDescriptor {
public:
    explicit TensorDescriptor(Tensor const& a);

    TensorDescriptor(fl::dtype const type, Shape const& afDims);
    ~TensorDescriptor();

private:
    cudnnTensorDescriptor_t _handle;

public:
    [[nodiscard]] constexpr auto get() const { return _handle; }
};

class TensorDescriptorArray {
public:
    TensorDescriptorArray(int size, fl::dtype const type, Shape const& dims);

    cudnnTensorDescriptor_t* descriptors;
    ~TensorDescriptorArray();

private:
    std::vector<TensorDescriptor> _descVec;
    std::vector<cudnnTensorDescriptor_t> _descRawVec;
};

class FilterDescriptor {
public:
    explicit FilterDescriptor(Tensor const& input);
    ~FilterDescriptor();

private:
    cudnnFilterDescriptor_t _handle;

public:
    [[nodiscard]] constexpr auto get() const { return _handle; }
};

class ConvDescriptor {
public:
    ConvDescriptor(
        fl::dtype type,
        int px,
        int py,
        int sx,
        int sy,
        int dx,
        int dy,
        int groups = 1
    );
    ~ConvDescriptor();

private:
    cudnnConvolutionDescriptor_t _handle;

public:
    [[nodiscard]] constexpr auto get() const { return _handle; }
};

class PoolingDescriptor {
public:
    PoolingDescriptor(
        int wx,
        int wy,
        int sx,
        int sy,
        int px,
        int py,
        PoolingMode mode
    );
    ~PoolingDescriptor();

private:
    cudnnPoolingDescriptor_t _handle;

public:
    [[nodiscard]] constexpr auto get() const { return _handle; }
};

class DropoutDescriptor {
public:
    explicit DropoutDescriptor(float dropProb);
    ~DropoutDescriptor();

    Tensor& getDropoutStates();

private:
    cudnnDropoutDescriptor_t _handle;

public:
    [[nodiscard]] constexpr auto get() const { return _handle; }

};

class RNNDescriptor {
public:
    RNNDescriptor(
        fl::dtype type,
        int inputSize,
        int hiddenSize,
        int numLayers,
        RnnMode mode,
        bool bidirectional,
        DropoutDescriptor& dropout
    );
    ~RNNDescriptor();

private:
    cudnnRNNDescriptor_t _handle = nullptr;

    static constexpr auto mathType(fl::dtype type) {
        return type == fl::dtype::f16 ? CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION : CUDNN_DEFAULT_MATH;
    }

public:
    /**
     * @return descriptor handle
     */
    constexpr auto get() const { return _handle; }
};


class RNNDataDescriptor {
public:
    RNNDataDescriptor(
        fl::dtype type,
        Shape const& dims
    );

    ~RNNDataDescriptor();

private:
    void create();
    void set(dtype type, int inputSize, int maxSeqSize, std::span<int const> sequenceSizes) const;

    cudnnRNNDataDescriptor_t _handle = nullptr;

public:
    /**
     * @return descriptor handle
     */
    constexpr auto get() const { return _handle; }
};

}

namespace fl {

#define CUDNN_CHECK_ERR(expr) ::fl::cudnnCheckErr((expr))

void cudnnCheckErr(cudnnStatus_t status);

cudnnDataType_t cudnnMapToType(fl::dtype const& t);

void const* kOne(fl::dtype const t);

void const* kZero(fl::dtype const t);

// TODO: move this to CudnnAutogradExtension if we make it a singleton
cudnnHandle_t getCudnnHandle();
CUDAStream const& getCudnnStream();


} // namespace fl



/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <mutex>
#include <unordered_map>

#include "flashlight/fl/tensor/TensorBackend.h"

#include <af/array.h>

namespace fl {

/**
 * A tensor backend implementation of the ArrayFire tensor library.
 *
 * Given that ArrayFire has an internal DeviceManager singleton to manage its
 * global state, nothing is stored here as those internals are opaquely handled.
 * This class simply dispatches operations on global tensor functions to their
 * ArrayFire counterparts.
 */
class ArrayFireBackend : public TensorBackend {
    // TODO: consolidate the ArrayFire memory manager here so its global state can
    // be stored/we can reduce the number of singletons.
    std::once_flag memoryInitFlag;

    // These help ensure we are using native device id in public methods.
    std::unordered_map<int, int> nativeIdToId_;
    std::unordered_map<int, int> idToNativeId_;

    // keep track of the individual active stream on each ArrayFire device
    // NOTE using a `shared_ptr` to allow its capture in setActive callback;
    // see constructor for details.
    std::shared_ptr<std::unordered_map<int, std::shared_ptr<Stream const>>>
    afIdToStream_{
        std::make_shared<
            std::unordered_map<int, std::shared_ptr<Stream const>>>()
    };

    // Intentionally private. Only one instance should exist/it should be accessed
    // via getInstance().
    ArrayFireBackend();

public:
    static ArrayFireBackend& getInstance();
    ~ArrayFireBackend() override = default;
    TensorBackendType backendType() const override;

    // No copy or move construction or assignment
    ArrayFireBackend(ArrayFireBackend&&) = delete;
    ArrayFireBackend(ArrayFireBackend const&) = delete;
    ArrayFireBackend& operator=(ArrayFireBackend&&) = delete;
    ArrayFireBackend& operator=(ArrayFireBackend const&) = delete;

    /* -------------------------- Compute Functions -------------------------- */
    void eval(Tensor const& tensor) override;

    /**
     * Return the stream from which the given array was created.
     *
     * @return an immutable reference to the stream from which `arr` was created.
     */
    Stream const& getStreamOfArray(af::array const& arr);
    bool supportsDataType(fl::dtype const& dtype) const override;
    // Memory management
    void getMemMgrInfo(char const* msg, int nativeDeviceId, std::ostream* ostream)
    override;
    void setMemMgrLogStream(std::ostream* stream) override;
    void setMemMgrLoggingEnabled(bool enabled) override;
    void setMemMgrFlushInterval(size_t interval) override;

    /* -------------------------- Rand Functions -------------------------- */
    void setSeed(int seed) override;
    Tensor randn(Shape const& shape, dtype type) override;
    Tensor rand(Shape const& shape, dtype type) override;

    /* --------------------------- Tensor Operators --------------------------- */
    /******************** Tensor Creation Functions ********************/
#define AF_BACKEND_CREATE_FUN_LITERAL_DECL(TYPE)                  \
        Tensor fromScalar(TYPE value, const dtype type) override; \
        Tensor full(const Shape& dims, TYPE value, const dtype type) override;
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const double&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const float&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const int&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const char&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned char&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const long&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned long&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const long long&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned long long&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const bool&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const short&);
    AF_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned short&);
#undef AF_BACKEND_CREATE_FUN_LITERAL_DECL

    Tensor identity(Dim dim, dtype type) override;
    Tensor arange(Shape const& shape, Dim seqDim, dtype type)
    override;
    Tensor iota(Shape const& dims, Shape const& tileDims, dtype type)
    override;

    /************************ Shaping and Indexing *************************/
    Tensor reshape(Tensor const& tensor, Shape const& shape) override;
    Tensor transpose(Tensor const& tensor, Shape const& axes /* = {} */) override;
    Tensor tile(Tensor const& tensor, Shape const& shape) override;
    Tensor concatenate(std::vector<Tensor> const& tensors, unsigned axis)
    override;
    Tensor nonzero(Tensor const& tensor) override;
    Tensor pad(
        Tensor const& input,
        std::vector<std::pair<int, int>> const& padWidths,
        PadType type
    ) override;

    /************************** Unary Operators ***************************/
    Tensor exp(Tensor const& tensor) override;
    Tensor log(Tensor const& tensor) override;
    Tensor negative(Tensor const& tensor) override;
    Tensor logicalNot(Tensor const& tensor) override;
    Tensor log1p(Tensor const& tensor) override;
    Tensor sin(Tensor const& tensor) override;
    Tensor cos(Tensor const& tensor) override;
    Tensor sqrt(Tensor const& tensor) override;
    Tensor tanh(Tensor const& tensor) override;
    Tensor floor(Tensor const& tensor) override;
    Tensor ceil(Tensor const& tensor) override;
    Tensor rint(Tensor const& tensor) override;
    Tensor absolute(Tensor const& tensor) override;
    Tensor sigmoid(Tensor const& tensor) override;
    Tensor erf(Tensor const& tensor) override;
    Tensor flip(Tensor const& tensor, unsigned dim) override;
    Tensor clip(Tensor const& tensor, Tensor const& low, Tensor const& high)
    override;
    Tensor roll(Tensor const& tensor, int shift, unsigned axis)
    override;
    Tensor isnan(Tensor const& tensor) override;
    Tensor isinf(Tensor const& tensor) override;
    Tensor sign(Tensor const& tensor) override;
    Tensor tril(Tensor const& tensor) override;
    Tensor triu(Tensor const& tensor) override;
    Tensor where(Tensor const& condition, Tensor const& x, Tensor const& y)
    override;
    void topk(
        Tensor& values,
        Tensor& indices,
        Tensor const& input,
        unsigned k,
        Dim axis,
        SortMode sortMode
    ) override;
    Tensor sort(Tensor const& input, Dim axis, SortMode sortMode)
    override;
    void sort(
        Tensor& values,
        Tensor& indices,
        Tensor const& input,
        Dim axis,
        SortMode sortMode
    ) override;
    Tensor argsort(Tensor const& input, Dim axis, SortMode sortMode)
    override;

    /************************** Binary Operators ***************************/
#define FL_AF_BINARY_OP_TYPE_DECL(FUNC, TYPE)            \
        Tensor FUNC(const Tensor& a, TYPE rhs) override; \
        Tensor FUNC(TYPE lhs, const Tensor& a) override;

#define FL_AF_BINARY_OP_LITERALS_DECL(FUNC)                         \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const bool&);               \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const int&);                \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const unsigned&);           \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const char&);               \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const unsigned char&);      \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const long&);               \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const unsigned long&);      \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const long long&);          \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const unsigned long long&); \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const double&);             \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const float&);              \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const short&);              \
        FL_AF_BINARY_OP_TYPE_DECL(FUNC, const unsigned short&);

#define FL_AF_BINARY_OP_DECL(FUNC)                                  \
        Tensor FUNC(const Tensor& lhs, const Tensor& rhs) override; \
        FL_AF_BINARY_OP_LITERALS_DECL(FUNC);

    FL_AF_BINARY_OP_DECL(add);
    FL_AF_BINARY_OP_DECL(sub);
    FL_AF_BINARY_OP_DECL(mul);
    FL_AF_BINARY_OP_DECL(div);
    FL_AF_BINARY_OP_DECL(eq);
    FL_AF_BINARY_OP_DECL(neq);
    FL_AF_BINARY_OP_DECL(lessThan);
    FL_AF_BINARY_OP_DECL(lessThanEqual);
    FL_AF_BINARY_OP_DECL(greaterThan);
    FL_AF_BINARY_OP_DECL(greaterThanEqual);
    FL_AF_BINARY_OP_DECL(logicalOr);
    FL_AF_BINARY_OP_DECL(logicalAnd);
    FL_AF_BINARY_OP_DECL(mod);
    FL_AF_BINARY_OP_DECL(bitwiseAnd);
    FL_AF_BINARY_OP_DECL(bitwiseOr);
    FL_AF_BINARY_OP_DECL(bitwiseXor);
    FL_AF_BINARY_OP_DECL(lShift);
    FL_AF_BINARY_OP_DECL(rShift);
#undef FL_AF_BINARY_OP_DECL
#undef FL_AF_BINARY_OP_TYPE_DECL
#undef FL_AF_BINARY_OP_LITERALS_DECL

    Tensor minimum(Tensor const& lhs, Tensor const& rhs) override;
    Tensor maximum(Tensor const& lhs, Tensor const& rhs) override;
    Tensor power(Tensor const& lhs, Tensor const& rhs) override;

    /******************************* BLAS ********************************/
    Tensor matmul(
        Tensor const& lhs,
        Tensor const& rhs,
        MatrixProperty lhsProp,
        MatrixProperty rhsProp
    ) override;

    /************************** Reductions ***************************/
    Tensor amin(Tensor const& input, std::vector<int> const& axes, bool keepDims)
    override;
    Tensor amax(Tensor const& input, std::vector<int> const& axes, bool keepDims)
    override;
    void min(
        Tensor& values,
        Tensor& indices,
        Tensor const& input,
        unsigned axis,
        bool keepDims
    ) override;
    void max(
        Tensor& values,
        Tensor& indices,
        Tensor const& input,
        unsigned axis,
        bool keepDims
    ) override;
    Tensor sum(Tensor const& input, std::vector<int> const& axes, bool keepDims)
    override;
    Tensor cumsum(Tensor const& input, unsigned axis) override;
    Tensor argmax(Tensor const& input, unsigned axis, bool keepDims)
    override;
    Tensor argmin(Tensor const& input, unsigned axis, bool keepDims)
    override;
    Tensor mean(Tensor const& input, std::vector<int> const& axes, bool keepDims)
    override;
    Tensor median(
        Tensor const& input,
        std::vector<int> const& axes,
        bool keepDims
    ) override;
    Tensor var(
        Tensor const& input,
        std::vector<int> const& axes,
        bool bias,
        bool keepDims
    ) override;
    Tensor std(Tensor const& input, std::vector<int> const& axes, bool keepDims)
    override;
    Tensor norm(
        Tensor const& input,
        std::vector<int> const& axes,
        double p,
        bool keepDims
    ) override;
    Tensor countNonzero(
        Tensor const& input,
        std::vector<int> const& axes,
        bool keepDims
    ) override;
    Tensor any(Tensor const& input, std::vector<int> const& axes, bool keepDims)
    override;
    Tensor all(Tensor const& input, std::vector<int> const& axes, bool keepDims)
    override;

    /************************** Utils ***************************/
    void print(Tensor const& tensor) override;
};

} // namespace fl

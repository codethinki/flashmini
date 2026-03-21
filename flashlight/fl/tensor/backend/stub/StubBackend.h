/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {

/**
 * A stub Tensor backend implementation to make it easy to get started with the
 * Flashlight Tensor API.
 *
 * This stub can be copied, renamed, and implemented as needed.
 */
class StubBackend : public TensorBackend {
public:
    StubBackend();

    static StubBackend& getInstance();
    ~StubBackend() override = default;
    TensorBackendType backendType() const override;

    // No copy or move construction or assignment
    StubBackend(StubBackend&&) = delete;
    StubBackend(StubBackend const&) = delete;
    StubBackend& operator=(StubBackend&&) = delete;
    StubBackend& operator=(StubBackend const&) = delete;

    /* -------------------------- Compute Functions -------------------------- */
    void eval(Tensor const& tensor) override;
    bool supportsDataType(fl::dtype const& dtype) const override;
    // Memory management
    void getMemMgrInfo(char const* msg, int const deviceId, std::ostream* ostream)
    override;
    void setMemMgrLogStream(std::ostream* stream) override;
    void setMemMgrLoggingEnabled(bool const enabled) override;
    void setMemMgrFlushInterval(size_t const interval) override;

    /* -------------------------- Rand Functions -------------------------- */
    void setSeed(int const seed) override;
    Tensor randn(Shape const& shape, dtype type) override;
    Tensor rand(Shape const& shape, dtype type) override;

    /* --------------------------- Tensor Operators --------------------------- */
    /******************** Tensor Creation Functions ********************/
#define FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(TYPE)             \
        Tensor fromScalar(TYPE value, const dtype type) override; \
        Tensor full(const Shape& dims, TYPE value, const dtype type) override;
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const double&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const float&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const int&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const char&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned char&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const long&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned long&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const long long&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned long long&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const bool&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const short&);
    FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL(const unsigned short&);
#undef FL_STUB_BACKEND_CREATE_FUN_LITERAL_DECL

    Tensor identity(Dim const dim, dtype const type) override;
    Tensor arange(Shape const& shape, Dim const seqDim, dtype const type)
    override;
    Tensor iota(Shape const& dims, Shape const& tileDims, dtype const type)
    override;

    /************************ Shaping and Indexing *************************/
    Tensor reshape(Tensor const& tensor, Shape const& shape) override;
    Tensor transpose(Tensor const& tensor, Shape const& axes /* = {} */) override;
    Tensor tile(Tensor const& tensor, Shape const& shape) override;
    Tensor concatenate(std::vector<Tensor> const& tensors, unsigned const axis)
    override;
    Tensor nonzero(Tensor const& tensor) override;
    Tensor pad(
        Tensor const& input,
        std::vector<std::pair<Dim, Dim>> const& padWidths,
        PadType const type
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
    Tensor flip(Tensor const& tensor, unsigned const dim) override;
    Tensor clip(Tensor const& tensor, Tensor const& low, Tensor const& high)
    override;
    Tensor roll(Tensor const& tensor, int const shift, unsigned const axis)
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
        unsigned const k,
        Dim const axis,
        SortMode const sortMode
    ) override;
    Tensor sort(Tensor const& input, Dim const axis, SortMode const sortMode)
    override;
    void sort(
        Tensor& values,
        Tensor& indices,
        Tensor const& input,
        Dim const axis,
        SortMode const sortMode
    ) override;
    Tensor argsort(Tensor const& input, Dim const axis, SortMode const sortMode)
    override;

    /************************** Binary Operators ***************************/
#define FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, TYPE)  \
        Tensor FUNC(const Tensor& a, TYPE rhs) override; \
        Tensor FUNC(TYPE lhs, const Tensor& a) override;

#define FL_STUB_BACKEND_BINARY_OP_LITERALS_DECL(FUNC)                         \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const bool&);               \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const int&);                \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned&);           \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const char&);               \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned char&);      \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const long&);               \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned long&);      \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const long long&);          \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned long long&); \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const double&);             \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const float&);              \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const short&);              \
        FL_STUB_BACKEND_BINARY_OP_TYPE_DECL(FUNC, const unsigned short&);

#define FL_STUB_BACKEND_BINARY_OP_DECL(FUNC)                        \
        Tensor FUNC(const Tensor& lhs, const Tensor& rhs) override; \
        FL_STUB_BACKEND_BINARY_OP_LITERALS_DECL(FUNC);

    FL_STUB_BACKEND_BINARY_OP_DECL(add);
    FL_STUB_BACKEND_BINARY_OP_DECL(sub);
    FL_STUB_BACKEND_BINARY_OP_DECL(mul);
    FL_STUB_BACKEND_BINARY_OP_DECL(div);
    FL_STUB_BACKEND_BINARY_OP_DECL(eq);
    FL_STUB_BACKEND_BINARY_OP_DECL(neq);
    FL_STUB_BACKEND_BINARY_OP_DECL(lessThan);
    FL_STUB_BACKEND_BINARY_OP_DECL(lessThanEqual);
    FL_STUB_BACKEND_BINARY_OP_DECL(greaterThan);
    FL_STUB_BACKEND_BINARY_OP_DECL(greaterThanEqual);
    FL_STUB_BACKEND_BINARY_OP_DECL(logicalOr);
    FL_STUB_BACKEND_BINARY_OP_DECL(logicalAnd);
    FL_STUB_BACKEND_BINARY_OP_DECL(mod);
    FL_STUB_BACKEND_BINARY_OP_DECL(bitwiseAnd);
    FL_STUB_BACKEND_BINARY_OP_DECL(bitwiseOr);
    FL_STUB_BACKEND_BINARY_OP_DECL(bitwiseXor);
    FL_STUB_BACKEND_BINARY_OP_DECL(lShift);
    FL_STUB_BACKEND_BINARY_OP_DECL(rShift);
#undef FL_STUB_BACKEND_BINARY_OP_DECL
#undef FL_STUB_BACKEND_BINARY_OP_TYPE_DECL
#undef FL_STUB_BACKEND_BINARY_OP_LITERALS_DECL

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
    Tensor amin(
        Tensor const& input,
        std::vector<int> const& axes,
        bool const keepDims
    ) override;
    Tensor amax(
        Tensor const& input,
        std::vector<int> const& axes,
        bool const keepDims
    ) override;
    void min(
        Tensor& values,
        Tensor& indices,
        Tensor const& input,
        unsigned const axis,
        bool const keepDims
    ) override;
    void max(
        Tensor& values,
        Tensor& indices,
        Tensor const& input,
        unsigned const axis,
        bool const keepDims
    ) override;
    Tensor sum(
        Tensor const& input,
        std::vector<int> const& axes,
        bool const keepDims
    ) override;
    Tensor cumsum(Tensor const& input, unsigned const axis) override;
    Tensor argmax(Tensor const& input, unsigned const axis, bool const keepDims)
    override;
    Tensor argmin(Tensor const& input, unsigned const axis, bool const keepDims)
    override;
    Tensor mean(
        Tensor const& input,
        std::vector<int> const& axes,
        bool const keepDims
    ) override;
    Tensor median(
        Tensor const& input,
        std::vector<int> const& axes,
        bool const keepDims
    ) override;
    Tensor var(
        Tensor const& input,
        std::vector<int> const& axes,
        bool const bias,
        bool const keepDims
    ) override;
    Tensor std(
        Tensor const& input,
        std::vector<int> const& axes,
        bool const keepDims
    ) override;
    Tensor norm(
        Tensor const& input,
        std::vector<int> const& axes,
        double p,
        bool const keepDims
    ) override;
    Tensor countNonzero(
        Tensor const& input,
        std::vector<int> const& axes,
        bool const keepDims
    ) override;
    Tensor any(
        Tensor const& input,
        std::vector<int> const& axes,
        bool const keepDims
    ) override;
    Tensor all(
        Tensor const& input,
        std::vector<int> const& axes,
        bool const keepDims
    ) override;

    /************************** Utils ***************************/
    void print(Tensor const& tensor) override;
};

} // namespace fl

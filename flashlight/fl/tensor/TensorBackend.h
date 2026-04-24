/*
 * SPDX-License-Identifier: MIT
 *
 * Original code: Copyright (c) Meta Platforms, Inc. (see FLASHLIGHT_LICENSE)
 * Modifications: Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */
#pragma once

#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/TensorExtension.h"

namespace fl {

class Stream;

/**
 * A Tensor backend that can be used to store global state associated with a
 * particular tensor implementation.
 *
 * This abstraction facilitates adherence to the implementation requirements for
 * global operators that operate on tensors (e.g. those functions that are not
 * members of `fl::Tensor`).
 *
 * Flashlight Tensors dispatch to their corresponding backends using
 * fl::Tensor::backend() --> typeToBackend (see below) to grab the correct
 * instance.
 */
class TensorBackend {
public:
    TensorBackend() = default;
    virtual ~TensorBackend() = default;
    virtual TensorBackendType backendType() const = 0;

    /* -------------------------- Compute Functions -------------------------- */
    virtual void eval(Tensor const& tensor) = 0;
    virtual bool supportsDataType(fl::dtype const& dtype) const = 0;
    // Memory Management
    virtual void getMemMgrInfo(char const* msg, int deviceId, std::ostream* ostream) = 0;
    virtual void setMemMgrLogStream(std::ostream* stream) = 0;
    virtual void setMemMgrLoggingEnabled(bool enabled) = 0;
    virtual void setMemMgrFlushInterval(size_t interval) = 0;

    /* -------------------------- Rand Functions -------------------------- */
    virtual void setSeed(int seed) = 0;
    virtual Tensor randn(Shape const& shape, dtype type) = 0;
    virtual Tensor rand(Shape const& shape, dtype type) = 0;

    /* --------------------------- Tensor Operators ---------------------------
     * For operator documentation and expected behavior, see TensorBase.h.
     */
    /******************** Tensor Creation Functions ********************/
#define FL_CREATE_FUN_LITERAL_BACKEND_DECL(TYPE)                     \
        virtual Tensor fromScalar(TYPE value, const dtype type) = 0; \
        virtual Tensor full(const Shape& dims, TYPE value, const dtype type) = 0;
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const double&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const float&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const int&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const unsigned&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const char&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const unsigned char&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const long&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const unsigned long&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const long long&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const unsigned long long&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const bool&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const short&);
    FL_CREATE_FUN_LITERAL_BACKEND_DECL(const unsigned short&);
#undef FL_CREATE_FUN_LITERAL_BACKEND_DECL

    virtual Tensor identity(Dim dim, dtype type) = 0;
    virtual Tensor arange(Shape const& shape, Dim seqDim, dtype type) = 0;

    virtual Tensor iota(Shape const& dims, Shape const& tileDims, dtype type) = 0;

    /************************ Shaping and Indexing *************************/
    virtual Tensor reshape(Tensor const& tensor, Shape const& shape) = 0;
    virtual Tensor transpose(
        Tensor const& tensor,
        Shape const& axes /* = {} */
    ) = 0;
    virtual Tensor tile(Tensor const& tensor, Shape const& shape) = 0;
    virtual Tensor concatenate(
        std::vector<Tensor> const& tensors,
        unsigned axis
    ) = 0;
    virtual Tensor nonzero(Tensor const& tensor) = 0;
    virtual Tensor pad(
        Tensor const& input,
        std::vector<std::pair<Dim, Dim>> const& padWidths,
        PadType type
    ) = 0;

    /************************** Unary Operators ***************************/
    virtual Tensor exp(Tensor const& tensor) = 0;
    virtual Tensor log(Tensor const& tensor) = 0;
    virtual Tensor negative(Tensor const& tensor) = 0;
    virtual Tensor logicalNot(Tensor const& tensor) = 0;
    virtual Tensor log1p(Tensor const& tensor) = 0;
    virtual Tensor sin(Tensor const& tensor) = 0;
    virtual Tensor cos(Tensor const& tensor) = 0;
    virtual Tensor sqrt(Tensor const& tensor) = 0;
    virtual Tensor tanh(Tensor const& tensor) = 0;
    virtual Tensor floor(Tensor const& tensor) = 0;
    virtual Tensor ceil(Tensor const& tensor) = 0;
    virtual Tensor rint(Tensor const& tensor) = 0;
    virtual Tensor absolute(Tensor const& tensor) = 0;
    virtual Tensor sigmoid(Tensor const& tensor) = 0;
    virtual Tensor erf(Tensor const& tensor) = 0;
    virtual Tensor flip(Tensor const& tensor, unsigned dim) = 0;
    virtual Tensor clip(Tensor const& tensor, Tensor const& low, Tensor const& high) = 0;
    virtual Tensor clip(Tensor const& tensor, Tensor const& low, double const& high);
    virtual Tensor clip(Tensor const& tensor, double const& low, Tensor const& high);
    virtual Tensor clip(Tensor const& tensor, double const& low, double const& high);
    virtual Tensor roll(Tensor const& tensor, int shift, unsigned axis) = 0;
    virtual Tensor isnan(Tensor const& tensor) = 0;
    virtual Tensor isinf(Tensor const& tensor) = 0;
    virtual Tensor sign(Tensor const& tensor) = 0;
    virtual Tensor tril(Tensor const& tensor) = 0;
    virtual Tensor triu(Tensor const& tensor) = 0;
    virtual Tensor where(Tensor const& condition, Tensor const& x, Tensor const& y) = 0;
    virtual Tensor where(Tensor const& condition, Tensor const& x, double const& y);
    virtual Tensor where(Tensor const& condition, double const& x, Tensor const& y);
    virtual void topk(
        Tensor& values,
        Tensor& indices,
        Tensor const& input,
        unsigned k,
        Dim axis,
        SortMode sortMode
    ) = 0;
    virtual Tensor sort(Tensor const& input, Dim axis, SortMode sortMode) = 0;
    virtual void sort(
        Tensor& values,
        Tensor& indices,
        Tensor const& input,
        Dim axis,
        SortMode sortMode
    ) = 0;
    virtual Tensor argsort(Tensor const& input, Dim axis, SortMode sortMode) = 0;

    /************************** Binary Operators ***************************/
#define FL_BINARY_OP_TYPE_DECL(FUNC, TYPE)                  \
        virtual Tensor FUNC(const Tensor& a, TYPE rhs) = 0; \
        virtual Tensor FUNC(TYPE lhs, const Tensor& a) = 0;

#define FL_BINARY_OP_LITERALS_DECL(FUNC)                         \
        FL_BINARY_OP_TYPE_DECL(FUNC, const bool&);               \
        FL_BINARY_OP_TYPE_DECL(FUNC, const int&);                \
        FL_BINARY_OP_TYPE_DECL(FUNC, const unsigned&);           \
        FL_BINARY_OP_TYPE_DECL(FUNC, const char&);               \
        FL_BINARY_OP_TYPE_DECL(FUNC, const unsigned char&);      \
        FL_BINARY_OP_TYPE_DECL(FUNC, const long&);               \
        FL_BINARY_OP_TYPE_DECL(FUNC, const unsigned long&);      \
        FL_BINARY_OP_TYPE_DECL(FUNC, const long long&);          \
        FL_BINARY_OP_TYPE_DECL(FUNC, const unsigned long long&); \
        FL_BINARY_OP_TYPE_DECL(FUNC, const double&);             \
        FL_BINARY_OP_TYPE_DECL(FUNC, const float&);              \
        FL_BINARY_OP_TYPE_DECL(FUNC, const short&);              \
        FL_BINARY_OP_TYPE_DECL(FUNC, const unsigned short&);

#define FL_BINARY_OP_DECL(FUNC)                                        \
        virtual Tensor FUNC(const Tensor& lhs, const Tensor& rhs) = 0; \
        FL_BINARY_OP_LITERALS_DECL(FUNC);

    FL_BINARY_OP_DECL(add);
    FL_BINARY_OP_DECL(sub);
    FL_BINARY_OP_DECL(mul);
    FL_BINARY_OP_DECL(div);
    FL_BINARY_OP_DECL(eq);
    FL_BINARY_OP_DECL(neq);
    FL_BINARY_OP_DECL(lessThan);
    FL_BINARY_OP_DECL(lessThanEqual);
    FL_BINARY_OP_DECL(greaterThan);
    FL_BINARY_OP_DECL(greaterThanEqual);
    FL_BINARY_OP_DECL(logicalOr);
    FL_BINARY_OP_DECL(logicalAnd);
    FL_BINARY_OP_DECL(mod);
    FL_BINARY_OP_DECL(bitwiseAnd);
    FL_BINARY_OP_DECL(bitwiseOr);
    FL_BINARY_OP_DECL(bitwiseXor);
    FL_BINARY_OP_DECL(lShift);
    FL_BINARY_OP_DECL(rShift);
#undef FL_BINARY_OP_DECL
#undef FL_BINARY_OP_TYPE_DECL
#undef FL_BINARY_OP_LITERALS_DECL

    virtual Tensor minimum(Tensor const& lhs, Tensor const& rhs) = 0;
    virtual Tensor minimum(Tensor const& lhs, double const& rhs);
    virtual Tensor minimum(double const& lhs, Tensor const& rhs);
    virtual Tensor maximum(Tensor const& lhs, Tensor const& rhs) = 0;
    virtual Tensor maximum(Tensor const& lhs, double const& rhs);
    virtual Tensor maximum(double const& lhs, Tensor const& rhs);
    virtual Tensor power(Tensor const& lhs, Tensor const& rhs) = 0;
    virtual Tensor power(Tensor const& lhs, double const& rhs);
    virtual Tensor power(double const& lhs, Tensor const& rhs);

    /******************************* BLAS ********************************/
    virtual Tensor matmul(
        Tensor const& lhs,
        Tensor const& rhs,
        MatrixProperty lhsProp,
        MatrixProperty rhsProp
    ) = 0;

    /************************** Reductions ***************************/
    virtual Tensor amin(Tensor const& input, std::vector<int> const& axes, bool keepDims) = 0;
    virtual Tensor amax(Tensor const& input, std::vector<int> const& axes, bool keepDims) = 0;
    virtual void min(
        Tensor& values,
        Tensor& indices,
        Tensor const& input,
        unsigned axis,
        bool keepDims
    ) = 0;
    virtual void max(
        Tensor& values,
        Tensor& indices,
        Tensor const& input,
        unsigned axis,
        bool keepDims
    ) = 0;
    virtual Tensor sum(Tensor const& input, std::vector<int> const& axes, bool keepDims) = 0;
    virtual Tensor cumsum(Tensor const& input, unsigned axis) = 0;
    virtual Tensor argmax(Tensor const& input, unsigned axis, bool keepDims) = 0;
    virtual Tensor argmin(Tensor const& input, unsigned axis, bool keepDims) = 0;
    virtual Tensor mean(Tensor const& input, std::vector<int> const& axes, bool keepDims) = 0;
    virtual Tensor median(Tensor const& input, std::vector<int> const& axes, bool keepDims) = 0;
    virtual Tensor var(
        Tensor const& input,
        std::vector<int> const& axes,
        bool bias,
        bool keepDims
    ) = 0;
    virtual Tensor std(Tensor const& input, std::vector<int> const& axes, bool keepDims) = 0;
    virtual Tensor norm(
        Tensor const& input,
        std::vector<int> const& axes,
        double p,
        bool keepDims
    ) = 0;
    virtual Tensor countNonzero(
        Tensor const& input,
        std::vector<int> const& axes,
        bool keepDims
    ) = 0;
    virtual Tensor any(Tensor const& input, std::vector<int> const& axes, bool keepDims) = 0;
    virtual Tensor all(Tensor const& input, std::vector<int> const& axes, bool keepDims) = 0;

    /************************** Utils ***************************/
    virtual void print(Tensor const& tensor) = 0;

    /**
     * Checks if a datatype is supported by a TensorBackend and its registered
     * extensions.
     *
     * @param[in] dtype the datatype to check
     *
     * @return true if the data type is supported, false otherwise
     */
    virtual bool isDataTypeSupported(fl::dtype const& dtype) const final;

    /********************* Tensor Extensions **********************/
    template<typename T>
    T& getExtension() {
        static_assert(
            std::is_base_of<TensorExtensionBase, T>::value,
            "TensorBackend::getExtension<T>() called with type T "
            "that is not derived from TensorExtensionBase."
        );

        TensorExtensionType e = T::getExtensionType();

        // If an extension isn't present, instantiate it via its registered
        // creation function - only do this once per extension.
        if(extensions_.find(e) == extensions_.end()) {
            auto& creationFunc =
                detail::TensorExtensionRegistrar::getInstance()
                .getTensorExtensionCreationFunc(this->backendType(), e);
            extensions_.emplace(e, creationFunc());
        }
        return *(static_cast<T*>(extensions_.at(e).get()));
    }

protected:
    std::unordered_map<TensorExtensionType, std::unique_ptr<TensorExtensionBase>>
    extensions_;
};

/**
 * Convert a Tensor from one backend to another.
 *
 * The resulting tensor will have the same shape, type, and contents.
 *
 * @param[in] in a tensor rvalue reference
 * @return a tensor with backend type specified by the template
 */
template<typename T>
Tensor toTensorType(Tensor&& in) {
    static_assert(
        std::is_base_of<TensorAdapterBase, T>::value,
        "toTensorType: T must be a derived type of TensorAdapterBase"
    );
    // Fast path - backend is the same
    // TODO: make fl::TensorBackendType a static constexpr on the class as well so
    // as to not need to instantiate a backend to check the type
    if(in.backendType() == T().backendType())
        return std::move(in);

    // As per impl requirements, Tensor::device() should return a pointer to host
    // memory if the tensor resides on the host.
    return Tensor(
        std::make_unique<T>(
            in.shape(),
            in.type(),
            // TODO: use the void specialization instead of a reinterpret cast
            reinterpret_cast<void*>(in.device<char>()),
            // expects contiguous memory
            in.location()
        )
    );
}

namespace detail {

    /**
     * Compare the backends of two tensors.
     *
     * @return true if the backends of both tensors are the same, else false.
     */
    bool areBackendsEqual(const Tensor& a, const Tensor& b);

    /**
     * Compare the backends of multiple tensors.
     *
     * @return true if all tensors' backends are the same, false otherwise.
     */
    template<typename... Args>
    bool areBackendsEqual(const Tensor& a, const Tensor& b, const Args&... args) {
        return areBackendsEqual(a, b) && areBackendsEqual(a, args...)
            && areBackendsEqual(b, args...);
    }

    /**
     *
     * @return a reference to a tensor backend instance descripting the default
       backend.
     */
    TensorBackend& getDefaultBackend();

} // namespace detail
} // namespace fl

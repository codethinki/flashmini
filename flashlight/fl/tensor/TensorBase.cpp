/*
 * SPDX-License-Identifier: MIT
 *
 * Original code: Copyright (c) Meta Platforms, Inc. (see FLASHLIGHT_LICENSE)
 * Modifications: Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */
#include "flashlight/fl/tensor/TensorBase.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBackend.h"

#define FL_TENSOR_BACKENDS_MATCH_CHECK(...)          \
        if(!detail::areBackendsEqual(__VA_ARGS__)) { \
            throw std::invalid_argument(             \
    std::string(__func__) +                          \
    " called with tensors of different backends."    \
            );                                       \
        }

namespace fl {

Tensor::Tensor(std::unique_ptr<TensorAdapterBase> adapter) : impl_(std::move(adapter)) {}

std::unique_ptr<TensorAdapterBase> Tensor::releaseAdapter() { return std::move(impl_); }

Tensor::~Tensor() = default;

Tensor::Tensor(Tensor const& tensor) : impl_(tensor.impl_->clone()) {}

Tensor::Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) {}

Tensor::Tensor() : impl_(detail::getDefaultAdapter()) {}

Tensor::Tensor(
    Shape const& shape,
    fl::dtype type,
    void const* ptr,
    MemoryLocation memoryLocation
) : impl_(detail::getDefaultAdapter(shape, type, ptr, memoryLocation)) {}

Tensor::Tensor(
    Dim const nRows,
    Dim const nCols,
    Tensor const& values,
    Tensor const& rowIdx,
    Tensor const& colIdx,
    StorageType storageType
) : impl_(
    detail::getDefaultAdapter(
        nRows,
        nCols,
        values,
        rowIdx,
        colIdx,
        storageType
    )
) {}

Tensor::Tensor(
    Shape const& shape,
    fl::dtype type /* = fl::dtype::f32 */
) : impl_(detail::getDefaultAdapter(shape, type)) {}

Tensor::Tensor(fl::dtype type) : impl_(detail::getDefaultAdapter(Shape({0}), type)) {}

Tensor Tensor::copy() const { return impl_->copy(); }

Tensor Tensor::shallowCopy() const { return impl_->shallowCopy(); }

Shape const& Tensor::shape() const { return impl_->shape(); }

Location Tensor::location() const { return impl_->location(); }

size_t Tensor::elements() const { return impl_->shape().elements(); }

Dim Tensor::dim(size_t const dim) const { return shape().dim(dim); }

int Tensor::ndim() const { return shape().ndim(); }

bool Tensor::isEmpty() const { return elements() == 0; }

bool Tensor::hasAdapter() const { return impl_.get() != nullptr; }

size_t Tensor::bytes() const { return elements() * getTypeSize(type()); }

dtype Tensor::type() const { return impl_->type(); }

bool Tensor::isSparse() const { return impl_->isSparse(); }

Tensor Tensor::asType(dtype const type) const { return impl_->astype(type); }

Tensor Tensor::operator()(std::vector<Index> const& indices) const { return impl_->index(indices); }

Tensor Tensor::flatten() const { return impl_->flatten(); }

Tensor Tensor::flat(Index const& idx) const { return impl_->flat(idx); }

Tensor Tensor::asContiguousTensor() const { return impl_->asContiguousTensor(); }

TensorBackendType Tensor::backendType() const { return impl_->backendType(); }

TensorBackend& Tensor::backend() const { return impl_->backend(); }

#define FL_CREATE_MEMORY_OPS(TYPE)                                                    \
        template<> FL_API TYPE Tensor::scalar() const {                               \
            if(isEmpty()) {                                                           \
                throw std::invalid_argument("Tensor::scalar called on empty tensor"); \
            }                                                                         \
            if(type() != dtype_traits<TYPE>::fl_type) {                               \
                throw std::invalid_argument(                                          \
    "Tensor::scalar: requested type of " +                                            \
    std::string(dtype_traits<TYPE>::name()) +                                         \
    " doesn't match tensor type, which is " + std::string{to_string(type())}          \
                );                                                                    \
            }                                                                         \
            TYPE out;                                                                 \
            impl_->scalar(&out);                                                      \
            return out;                                                               \
        }                                                                             \
                                                                                      \
        template<> FL_API TYPE * Tensor::device() const {                             \
            if(isEmpty()) {                                                           \
                return nullptr;                                                       \
            }                                                                         \
            TYPE* out;                                                                \
            void** addr = reinterpret_cast<void**>(&out);                             \
            impl_->device(addr);                                                      \
            return out;                                                               \
        }                                                                             \
                                                                                      \
        template<>                                                                    \
        FL_API void Tensor::device(TYPE * *ptr) const {                               \
            if(isEmpty()) {                                                           \
                return;                                                               \
            }                                                                         \
            impl_->device(reinterpret_cast<void**>(ptr));                             \
        }                                                                             \
                                                                                      \
        template<> FL_API TYPE * Tensor::host() const {                               \
            if(isEmpty()) {                                                           \
                return nullptr;                                                       \
            }                                                                         \
            auto* out = reinterpret_cast<TYPE*>(new char[bytes()]);                   \
            impl_->host(out);                                                         \
            return out;                                                               \
        }                                                                             \
                                                                                      \
        template<>                                                                    \
        FL_API void Tensor::host(TYPE * ptr) const {                                  \
            if(!isEmpty()) {                                                          \
                impl_->host(ptr);                                                     \
            }                                                                         \
        }
// TODO this should be normal templates
FL_CREATE_MEMORY_OPS(int);
FL_CREATE_MEMORY_OPS(unsigned);
FL_CREATE_MEMORY_OPS(char);
FL_CREATE_MEMORY_OPS(unsigned char);
FL_CREATE_MEMORY_OPS(long);
FL_CREATE_MEMORY_OPS(unsigned long);
FL_CREATE_MEMORY_OPS(long long);
FL_CREATE_MEMORY_OPS(unsigned long long);
FL_CREATE_MEMORY_OPS(double);
FL_CREATE_MEMORY_OPS(float);
FL_CREATE_MEMORY_OPS(short);
FL_CREATE_MEMORY_OPS(unsigned short);
// void specializations
template<>
FL_API void* Tensor::device() const {
    if(isEmpty())
        return nullptr;
    void* out;
    impl_->device(&out);
    return out;
}

template<>
FL_API void Tensor::device(void** ptr) const {
    if(isEmpty())
        return;
    impl_->device(ptr);
}

template<>
FL_API void* Tensor::host() const {
    if(isEmpty())
        return nullptr;
    auto* out = reinterpret_cast<void*>(new char[bytes()]);
    impl_->host(out);
    return out;
}

template<>
FL_API void Tensor::host(void* ptr) const { impl_->host(ptr); }
#undef FL_CREATE_MEMORY_OPS

void Tensor::unlock() const { impl_->unlock(); }

bool Tensor::isLocked() const { return impl_->isLocked(); }

bool Tensor::isContiguous() const { return impl_->isContiguous(); }

Shape Tensor::strides() const { return impl_->strides(); }

Stream const& Tensor::stream() const { return impl_->stream(); }

void Tensor::setContext(void* context) { impl_->setContext(context); }

void* Tensor::getContext() const { return impl_->getContext(); }

std::string Tensor::toString() const { return impl_->toString(); }

std::ostream& Tensor::operator<<(std::ostream& ostr) const { return impl_->operator<<(ostr); }

/******************** Assignment Operators ********************/
#define FL_ASSIGN_OP_TYPE(OP, FUN, TYPE) \
        Tensor & Tensor::OP(TYPE val) {  \
            impl_->FUN(val);             \
            return *this;                \
        }
#define FL_ASSIGN_TENSOR_OP(OP, FUN) FL_ASSIGN_OP_TYPE(OP, FUN, const Tensor&);
#define FL_ASSIGN_SCALAR_OP(OP, FUN)                       \
        FL_ASSIGN_OP_TYPE(OP, FUN, const double&);         \
        FL_ASSIGN_OP_TYPE(OP, FUN, const float&);          \
        FL_ASSIGN_OP_TYPE(OP, FUN, const int&);            \
        FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned&);       \
        FL_ASSIGN_OP_TYPE(OP, FUN, const bool&);           \
        FL_ASSIGN_OP_TYPE(OP, FUN, const char&);           \
        FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned char&);  \
        FL_ASSIGN_OP_TYPE(OP, FUN, const short&);          \
        FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned short&); \
        FL_ASSIGN_OP_TYPE(OP, FUN, const long&);           \
        FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned long&);  \
        FL_ASSIGN_OP_TYPE(OP, FUN, const long long&);      \
        FL_ASSIGN_OP_TYPE(OP, FUN, const unsigned long long&);

#define FL_ASSIGN_OP(OP, FUN)         \
        FL_ASSIGN_TENSOR_OP(OP, FUN); \
        FL_ASSIGN_SCALAR_OP(OP, FUN);

// (operator, function name on impl)
FL_ASSIGN_SCALAR_OP(operator=, assign);
FL_ASSIGN_OP(operator+=, inPlaceAdd);
FL_ASSIGN_OP(operator-=, inPlaceSubtract);
FL_ASSIGN_OP(operator*=, inPlaceMultiply);
FL_ASSIGN_OP(operator/=, inPlaceDivide);
#undef FL_ASSIGN_OP_TYPE
#undef FL_ASSIGN_TENSOR_OP
#undef FL_ASSIGN_SCALAR_OP
#undef FL_ASSIGN_OP

// Move assignment operator when `this` is a lvalue, e.g., `x = std::move(y)`.
// In such cases, we let `this` take over the tensor data of `other`.
Tensor& Tensor::operator=(Tensor&& other) & {
    if(this == &other)
        return *this;
    this->impl_ = std::move(other.impl_);
    return *this;
}

// Move assignment operator when `this` is a rvalue, e.g., `x(0) =
// std::move(y)`. In such cases, we copy the data from `other` to `this`.
Tensor& Tensor::operator=(Tensor&& other) && {
    if(this == &other) 
        return *this;

    this->impl_->assign(other);
    return *this;
}

// Copy assignment operator when `this` is a lvalue, e.g., `x = y`.
// In such cases, we let `this` take over the _cloned_ data from `other`.
Tensor& Tensor::operator=(Tensor const& other) & {
    if(this == &other)
        return *this;

    this->impl_ = other.impl_->clone();
    return *this;
}

// Copy assignment operator when `this` is a lvalue, e.g., `x(0) = y`.
// In such cases, we copy the data from `other` to `this`.
Tensor& Tensor::operator=(Tensor const& other) && {
    if(this == &other)
        return *this;

    this->impl_->assign(other);
    return *this;
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define FL_CREATE_FUN_LITERAL_TYPE(TYPE)                                                 \
        template<> FL_API Tensor fromScalar(TYPE value, const dtype type) {              \
            return defaultTensorBackend().fromScalar(value, type);                       \
        }                                                                                \
        template<> FL_API Tensor full(const Shape& dims, TYPE value, const dtype type) { \
            return defaultTensorBackend().full(dims, value, type);                       \
        }
FL_CREATE_FUN_LITERAL_TYPE(const double&);
FL_CREATE_FUN_LITERAL_TYPE(const float&);
FL_CREATE_FUN_LITERAL_TYPE(const int&);
FL_CREATE_FUN_LITERAL_TYPE(const unsigned&);
FL_CREATE_FUN_LITERAL_TYPE(const char&);
FL_CREATE_FUN_LITERAL_TYPE(const unsigned char&);
FL_CREATE_FUN_LITERAL_TYPE(const long&);
FL_CREATE_FUN_LITERAL_TYPE(const unsigned long&);
FL_CREATE_FUN_LITERAL_TYPE(const long long&);
FL_CREATE_FUN_LITERAL_TYPE(const unsigned long long&);
FL_CREATE_FUN_LITERAL_TYPE(const bool&);
FL_CREATE_FUN_LITERAL_TYPE(const short&);
FL_CREATE_FUN_LITERAL_TYPE(const unsigned short&);
#undef FL_CREATE_FUN_LITERAL_TYPE

Tensor identity(Dim const dim, dtype const type) { return defaultTensorBackend().identity(dim, type); }

#define FL_ARRANGE_FUN_DEF(TYPE)                                                              \
        template<> FL_API Tensor arrange(TYPE start, TYPE end, TYPE step, const dtype type) { \
            return fl::arrange({static_cast<long>((end - start) / step)}, 0, type) *          \
                   step +                                                                    \
                   start;                                                                    \
        }
FL_ARRANGE_FUN_DEF(const double&);
FL_ARRANGE_FUN_DEF(const float&);
FL_ARRANGE_FUN_DEF(const int&);
FL_ARRANGE_FUN_DEF(const unsigned&);
FL_ARRANGE_FUN_DEF(const long&);
FL_ARRANGE_FUN_DEF(const unsigned long&);
FL_ARRANGE_FUN_DEF(const long long&);
FL_ARRANGE_FUN_DEF(const unsigned long long&);

Tensor arrange(Shape const& shape, Dim const seqDim, dtype const type) {
    return defaultTensorBackend().arange(shape, seqDim, type);
}

Tensor iota(Shape const& dims, Shape const& tileDims, dtype const type) {
    return defaultTensorBackend().iota(dims, tileDims, type);
}

/************************ Shaping and Indexing *************************/

Tensor reshape(Tensor const& tensor, Shape const& shape) { return tensor.backend().reshape(tensor, shape); }

Tensor transpose(Tensor const& tensor, Shape const& axes /* = {} */) {
    return tensor.backend().transpose(tensor, axes);
}

Tensor tile(Tensor const& tensor, Shape const& shape) { return tensor.backend().tile(tensor, shape); }

Tensor concatenate(std::vector<Tensor> const& tensors, unsigned const axis) {
    if(tensors.empty())
        throw std::invalid_argument("concatenate: called on empty set of tensors");

    // Check all backends match
    TensorBackendType const b = tensors.front().backendType();
    bool const matches =
        std::all_of(
            tensors.begin(),
            tensors.end(),
            [b](Tensor const& t) { return t.backendType() == b; }
        );
    if(!matches)
        throw std::invalid_argument(
            "concatenate: tried to concatenate tensors of different backends"
        );

    return tensors.front().backend().concatenate(tensors, axis);
}

Tensor nonzero(Tensor const& tensor) { return tensor.backend().nonzero(tensor); }

Tensor pad(
    Tensor const& input,
    std::vector<std::pair<int, int>> const& padWidths,
    PadType const type
) { return input.backend().pad(input, padWidths, type); }

/************************** Unary Operators ***************************/
Tensor exp(Tensor const& tensor) { return tensor.backend().exp(tensor); }

Tensor log(Tensor const& tensor) { return tensor.backend().log(tensor); }

Tensor negative(Tensor const& tensor) { return tensor.backend().negative(tensor); }

Tensor logicalNot(Tensor const& tensor) { return tensor.backend().logicalNot(tensor); }

Tensor log1p(Tensor const& tensor) { return tensor.backend().log1p(tensor); }

Tensor sin(Tensor const& tensor) { return tensor.backend().sin(tensor); }

Tensor cos(Tensor const& tensor) { return tensor.backend().cos(tensor); }

Tensor sqrt(Tensor const& tensor) { return tensor.backend().sqrt(tensor); }

Tensor tanh(Tensor const& tensor) { return tensor.backend().tanh(tensor); }

Tensor floor(Tensor const& tensor) { return tensor.backend().floor(tensor); }

Tensor ceil(Tensor const& tensor) { return tensor.backend().ceil(tensor); }

Tensor rint(Tensor const& tensor) { return tensor.backend().rint(tensor); }

Tensor absolute(Tensor const& tensor) { return tensor.backend().absolute(tensor); }

Tensor sigmoid(Tensor const& tensor) { return tensor.backend().sigmoid(tensor); }

Tensor erf(Tensor const& tensor) { return tensor.backend().erf(tensor); }

Tensor flip(Tensor const& tensor, unsigned const dim) { return tensor.backend().flip(tensor, dim); }

Tensor clip(Tensor const& tensor, Tensor const& low, Tensor const& high) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(tensor, low, high);
    return tensor.backend().clip(tensor, low, high);
}

Tensor clip(Tensor const& tensor, Tensor const& low, double const& high) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(tensor, low);
    return tensor.backend().clip(tensor, low, high);
}

Tensor clip(Tensor const& tensor, double const& low, Tensor const& high) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(tensor, high);
    return tensor.backend().clip(tensor, low, high);
}

Tensor clip(Tensor const& tensor, double const& low, double const& high) {
    return tensor.backend().clip(tensor, low, high);
}

Tensor roll(Tensor const& tensor, int const shift, unsigned const axis) {
    return tensor.backend().roll(tensor, shift, axis);
}

Tensor isnan(Tensor const& tensor) { return tensor.backend().isnan(tensor); }

Tensor isinf(Tensor const& tensor) { return tensor.backend().isinf(tensor); }

Tensor sign(Tensor const& tensor) { return tensor.backend().sign(tensor); }

Tensor tril(Tensor const& tensor) { return tensor.backend().tril(tensor); }

Tensor triu(Tensor const& tensor) { return tensor.backend().triu(tensor); }

Tensor where(Tensor const& condition, Tensor const& x, Tensor const& y) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(condition, x, y);
    return condition.backend().where(condition, x, y);
}

Tensor where(Tensor const& condition, Tensor const& x, double const& y) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(condition, x);
    return condition.backend().where(condition, x, y);
}

Tensor where(Tensor const& condition, double const& x, Tensor const& y) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(condition, y);
    return condition.backend().where(condition, x, y);
}

void topk(
    Tensor& values,
    Tensor& indices,
    Tensor const& input,
    unsigned const k,
    Dim const axis,
    SortMode const sortMode /* = SortMode::Descending */
) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(values, indices, input);
    input.backend().topk(values, indices, input, k, axis, sortMode);
}

Tensor sort(Tensor const& input, Dim const axis, SortMode const sortMode) {
    return input.backend().sort(input, axis, sortMode);
}

void sort(
    Tensor& values,
    Tensor& indices,
    Tensor const& input,
    Dim const axis,
    SortMode const sortMode /* = SortMode::Descending */
) { return values.backend().sort(values, indices, input, axis, sortMode); }

Tensor argsort(Tensor const& input, Dim const axis, SortMode const sortMode) {
    return input.backend().argsort(input, axis, sortMode);
}

/************************** Binary Operators ***************************/
#define FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, TYPE)     \
        Tensor FUNC(TYPE lhs, const Tensor& rhs) {        \
            return rhs.backend().FUNC(lhs, rhs);          \
        }                                                 \
        Tensor FUNC(const Tensor& lhs, TYPE rhs) {        \
            return lhs.backend().FUNC(lhs, rhs);          \
        }                                                 \
        Tensor operator OP(TYPE lhs, const Tensor& rhs) { \
            return FUNC(lhs, rhs);                        \
        }                                                 \
        Tensor operator OP(const Tensor& lhs, TYPE rhs) { \
            return FUNC(lhs, rhs);                        \
        }

#define FL_BINARY_OP_LITERALS_DEF(OP, FUNC)                                 \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const bool&);               \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const int&);                \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const unsigned&);           \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const char&);               \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const unsigned char&);      \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const long&);               \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const unsigned long&);      \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const long long&);          \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const unsigned long long&); \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const double&);             \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const float&);              \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const short&);              \
        FL_BINARY_OP_LITERAL_TYPE_DEF(OP, FUNC, const unsigned short&);

#define FL_BINARY_OP_DEF(OP, FUNC)                                 \
        Tensor FUNC(const Tensor& lhs, const Tensor& rhs) {        \
            FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);              \
            return lhs.backend().FUNC(lhs, rhs);                   \
        }                                                          \
        Tensor operator OP(const Tensor& lhs, const Tensor& rhs) { \
            return FUNC(lhs, rhs);                                 \
        }                                                          \
        FL_BINARY_OP_LITERALS_DEF(OP, FUNC);

FL_BINARY_OP_DEF(+, add);
FL_BINARY_OP_DEF(-, sub);
FL_BINARY_OP_DEF(*, mul);
FL_BINARY_OP_DEF(/, div);
FL_BINARY_OP_DEF(==, eq);
FL_BINARY_OP_DEF(!=, neq);
FL_BINARY_OP_DEF(<, lessThan);
FL_BINARY_OP_DEF(<=, lessThanEqual);
FL_BINARY_OP_DEF(>, greaterThan);
FL_BINARY_OP_DEF(>=, greaterThanEqual);
FL_BINARY_OP_DEF(||, logicalOr);
FL_BINARY_OP_DEF(&&, logicalAnd);
FL_BINARY_OP_DEF(%, mod);
FL_BINARY_OP_DEF(&, bitwiseAnd);
FL_BINARY_OP_DEF(|, bitwiseOr);
FL_BINARY_OP_DEF(^, bitwiseXor);
FL_BINARY_OP_DEF(<<, lShift);
FL_BINARY_OP_DEF(>>, rShift);

#undef FL_BINARY_OP_DEF
#undef FL_BINARY_OP_LITERALS_DEF
#undef FL_BINARY_OP_LITERAL_TYPE_DEF

Tensor minimum(Tensor const& lhs, Tensor const& rhs) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
    return lhs.backend().minimum(lhs, rhs);
}

Tensor maximum(Tensor const& lhs, Tensor const& rhs) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
    return lhs.backend().maximum(lhs, rhs);
}

Tensor minimum(Tensor const& lhs, double const& rhs) { return lhs.backend().minimum(lhs, rhs); }

Tensor minimum(double const& lhs, Tensor const& rhs) { return rhs.backend().minimum(lhs, rhs); }

Tensor maximum(Tensor const& lhs, double const& rhs) { return lhs.backend().maximum(lhs, rhs); }

Tensor maximum(double const& lhs, Tensor const& rhs) { return rhs.backend().maximum(lhs, rhs); }

Tensor power(Tensor const& lhs, Tensor const& rhs) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
    return lhs.backend().power(lhs, rhs);
}

Tensor power(Tensor const& lhs, double const& rhs) { return lhs.backend().power(lhs, rhs); }

Tensor power(double const& lhs, Tensor const& rhs) { return rhs.backend().power(lhs, rhs); }

/******************************* BLAS ********************************/
Tensor matmul(
    Tensor const& lhs,
    Tensor const& rhs,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp
) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(lhs, rhs);
    return lhs.backend().matmul(lhs, rhs, lhsProp, rhsProp);
}

/************************** Reductions ***************************/

Tensor amin(
    Tensor const& input,
    std::vector<int> const& axes /* = {} */,
    bool const keepDims /* = false */
) { return input.backend().amin(input, axes, keepDims); }

Tensor amax(
    Tensor const& input,
    std::vector<int> const& axes /* = {} */,
    bool const keepDims /* = false */
) { return input.backend().amax(input, axes, keepDims); }

void min(
    Tensor& values,
    Tensor& indices,
    Tensor const& input,
    unsigned const axis,
    bool const keepDims
) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(values, indices, input);
    return input.backend().min(values, indices, input, axis, keepDims);
}

void max(
    Tensor& values,
    Tensor& indices,
    Tensor const& input,
    unsigned const axis,
    bool const keepDims /* = false */
) {
    FL_TENSOR_BACKENDS_MATCH_CHECK(values, indices, input);
    return input.backend().max(values, indices, input, axis, keepDims);
}

Tensor sum(
    Tensor const& input,
    std::vector<int> const& axes /* = {} */,
    bool const keepDims /* = false */
) { return input.backend().sum(input, axes, keepDims); }

Tensor cumsum(Tensor const& input, unsigned const axis) { return input.backend().cumsum(input, axis); }

Tensor argmax(
    Tensor const& input,
    unsigned const axis,
    bool const keepDims /* = false */
) { return input.backend().argmax(input, axis, keepDims); }

Tensor argmin(
    Tensor const& input,
    unsigned const axis,
    bool const keepDims /* = false */
) { return input.backend().argmin(input, axis, keepDims); }

Tensor mean(
    Tensor const& input,
    std::vector<int> const& axes /* = {} */,
    bool const keepDims /* = false */
) { return input.backend().mean(input, axes, keepDims); }

Tensor median(
    Tensor const& input,
    std::vector<int> const& axes /* = {} */,
    bool const keepDims /* = false */
) { return input.backend().median(input, axes, keepDims); }

Tensor var(
    Tensor const& input,
    std::vector<int> const& axes /* = {} */,
    bool const bias,
    bool const keepDims /* = false */
) { return input.backend().var(input, axes, bias, keepDims); }

Tensor std(
    Tensor const& input,
    std::vector<int> const& axes /* = {} */,
    bool const keepDims /* = false */
) { return input.backend().std(input, axes, keepDims); }

Tensor norm(
    Tensor const& input,
    std::vector<int> const& axes /* = {} */,
    double p /* = 2 */,
    bool const keepDims /* = false */
) { return input.backend().norm(input, axes, p, keepDims); }

Tensor countNonzero(
    Tensor const& input,
    std::vector<int> const& axes /* = {} */,
    bool const keepDims /* = false */
) { return input.backend().countNonzero(input, axes, keepDims); }

Tensor any_of(
    Tensor const& input,
    std::vector<int> const& axes /* = {} */,
    bool const keepDims /* = false */
) { return input.backend().any(input, axes, keepDims); }

Tensor all_of(
    Tensor const& input,
    std::vector<int> const& axes /* = {} */,
    bool const keepDims /* = false */
) { return input.backend().all(input, axes, keepDims); }

/************************** Utilities ***************************/

std::ostream& operator<<(std::ostream& ostr, Tensor const& t) {
    t.operator<<(ostr);
    return ostr;
}

void print(Tensor const& tensor) { tensor.backend().print(tensor); }

bool allClose(
    fl::Tensor const& a,
    fl::Tensor const& b,
    double const absTolerance
) {
    if(a.type() != b.type())
        return false;
    if(a.shape() != b.shape())
        return false;
    if(a.elements() == 0 && b.elements() == 0)
        return true;

    auto const diff = fl::amax(fl::abs(a - b)).asType(dtype::f64).scalar<double>();

    return diff < absTolerance;
}

bool isInvalidArray(Tensor const& tensor) {
    return fl::any_of(fl::isnan(tensor)).asScalar<bool>()
        || fl::any_of(fl::isinf(tensor)).asScalar<bool>();
}

std::string tensorBackendTypeToString(TensorBackendType const type) {
    switch(type) {
        case TensorBackendType::Stub: return "Stub";
        case TensorBackendType::Tracer: return "Tracer";
        case TensorBackendType::ArrayFire: return "ArrayFire";
    }
    throw std::runtime_error("Unreachable -- unrecognized tensor backend type");
}

std::ostream& operator<<(std::ostream& os, TensorBackendType const type) {
    os << tensorBackendTypeToString(type);
    return os;
}

namespace detail {

    std::unique_ptr<TensorAdapterBase> releaseAdapter(Tensor&& t) { return t.releaseAdapter(); }

    std::unique_ptr<TensorAdapterBase> releaseAdapterUnsafe(Tensor& t) { return t.releaseAdapter(); }

    bool areTensorTypesEqual(Tensor const& a, Tensor const& b) { return a.type() == b.type(); }

} // namespace detail

} // namespace fl

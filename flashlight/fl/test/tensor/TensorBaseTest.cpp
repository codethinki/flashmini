/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

#include <format>

using namespace ::testing;
using namespace fl;


TEST(TensorBaseTest, FullTypeMismatch) {
    Shape const shape{2, 2};

    // Case where everything matches
    auto const x0 = fl::full<double>(shape, 1.0, fl::dtype::f64);
    ASSERT_EQ(x0.type(), fl::dtype::f64);
    ASSERT_EQ(x0.shape(), shape);


    auto x0Span = x0.host<double>();
    for(double val : x0Span)
        ASSERT_EQ(val, 1.0);

    auto const x1 = fl::full<int>(shape, 0, fl::dtype::f64);
    ASSERT_EQ(x1.type(), fl::dtype::f64);
    ASSERT_EQ(x1.shape(), shape);

    auto x1Span = x1.host<double>();
    for(double val : x1Span)
        ASSERT_EQ(val, 0.0);

    auto const x2 = fl::full<double>(shape, 1.0, fl::dtype::s32);
    ASSERT_EQ(x2.type(), fl::dtype::s32);
    ASSERT_EQ(x2.shape(), shape);

    auto const x2Span = x2.host<int>();
    for(int val : x2Span)
        ASSERT_EQ(val, 1);
}
TEST(TensorBaseTest, ArangeTypeMismatch) {
    // Case where everything matches
    auto const y0 = fl::arange<double>(0.0, 4.0, 1.0, fl::dtype::f64);
    ASSERT_EQ(y0.type(), fl::dtype::f64);
    ASSERT_EQ(y0.shape(), Shape({4}));
    ASSERT_EQ(y0.scalar<double>(), 0.0);

    // Emitting int literals while requesting f64 tensor creation.
    auto const y1 = fl::arange<int>(0, 4, 1, fl::dtype::f64);
    ASSERT_EQ(y1.type(), fl::dtype::f64);
    ASSERT_EQ(y1.shape(), Shape({4}));
    ASSERT_EQ(y1.scalar<double>(), 0.0);

    // Emitting double literals while requesting s32 tensor creation.
    auto const y2 = fl::arange<double>(0.0, 4.0, 1.0, fl::dtype::s32);
    ASSERT_EQ(y2.type(), fl::dtype::s32);
    ASSERT_EQ(y2.shape(), Shape({4}));
    ASSERT_EQ(y2.scalar<int>(), 0);
}

TEST(TensorBaseTest, Concatenate) {
    auto a = fl::full({3, 3}, 1.f);
    auto b = fl::full({3, 3}, 2.f);
    auto c = fl::full({3, 3}, 3.f);
    ASSERT_TRUE(
        allClose(fl::concatenate(0, a, b, c), fl::concatenate({a, b, c}))
    );
    auto const out = fl::concatenate(0, a, b, c);
    ASSERT_EQ(out.shape(), (Shape{9, 3}));

    // Empty tenors
    ASSERT_EQ(fl::concatenate(0, Tensor{}, Tensor{}).shape(), Shape{0});
    ASSERT_EQ(fl::concatenate(2, Tensor{}, Tensor{}).shape(), (Shape{0, 1, 1}));
    ASSERT_EQ(
        fl::concatenate(1, fl::rand({5, 5}), Tensor{}).shape(),
        (Shape{5, 5})
    );
}

TEST(TensorBaseTest, ConcatenateMany) {
    for(int n = 1; n <= 30; ++n) {
        std::vector<Tensor> tensors{};
        std::vector<int> expectedData{};
        long long totalSize = 0;

        for(size_t i = 0; i < n; ++i) {
            // Variable width: i + 1 elements
            auto const width = i + 1;

            // Variable content: start at i * 10
            auto const startVal = i * 10;
            auto t = fl::arange<int>(startVal, startVal + width, 1, fl::dtype::s32);
            tensors.push_back(t);

            for(size_t j = 0; j < width; ++j)
                expectedData.push_back(startVal + j);
            totalSize += width;
        }

        auto result = fl::concatenate(tensors, /* axis = */ 0);
        auto expectedTensor = Tensor::fromVector<int>({totalSize}, expectedData);

        ASSERT_EQ(result.shape(), Shape({totalSize}));
        ASSERT_TRUE(allClose(result, expectedTensor));
    }
}


TEST(TensorBaseTest, ConcatenateDuplicateTensors) {
    auto t1 = fl::full({2, 2}, 1.0f, fl::dtype::f32);
    auto t2 = fl::full({2, 2}, 2.0f, fl::dtype::f32);

    auto result = fl::concatenate({t1, t2, t1, t2}, /* axis = */ 1);
    auto expected = fl::concatenate({t1.copy(), t2.copy(), t1.copy(), t2.copy()}, /* axis = */ 1);

    ASSERT_TRUE(allClose(result, expected));
}

TEST(TensorBaseTest, ConcatenateViews) {
    std::vector const data{
        0.1f,
        0.2f,
        0.3f,
        0.4f,
        0.5f,
        1.1f,
        1.2f,
        1.3f,
        1.4f,
        1.5f
    };
    auto const t = fl::Tensor::fromVector({5, 2}, data);

    auto const vertTiled = fl::concatenate(
        0,
        fl::reshape(t(0, fl::span), {1, 2}),
        t,
        fl::reshape(t(t.dim(0) - 1, fl::span), {1, 2})
    );
    auto vTiled0 = vertTiled(fl::span, 0);
    auto vTiled1 = vertTiled(fl::span, 1);

    auto const result = fl::concatenate({vTiled1, vTiled0, vTiled0, vTiled1, vTiled1, vTiled0}, 1);

    auto const expectedSymmetricPad = fl::concatenate(
        {vTiled1.copy(), vTiled0.copy(), vTiled0.copy(), vTiled1.copy(), vTiled1.copy(), vTiled0.copy()},
        1
    );

    ASSERT_TRUE(allClose(result, expectedSymmetricPad));
}



TEST(TensorBaseTest, DefaultConstruction) {
    Tensor const t{};
    ASSERT_EQ(t.shape(), Shape({0}));
    ASSERT_EQ(t.type(), fl::dtype::f32);

    Tensor const u{{1, 2, 3}};
    ASSERT_EQ(u.shape(), Shape({1, 2, 3}));
    ASSERT_EQ(u.type(), fl::dtype::f32);
    Tensor const x({0, 3});
    ASSERT_EQ(x.shape(), Shape({0, 3}));

    Tensor const q(fl::dtype::f64);
    ASSERT_EQ(q.shape(), Shape({0}));
    ASSERT_EQ(q.type(), fl::dtype::f64);

    Tensor const v({4, 5, 6}, fl::dtype::u64);
    ASSERT_EQ(v.shape(), Shape({4, 5, 6}));
    ASSERT_EQ(v.type(), fl::dtype::u64);
}

TEST(TensorBaseTest, CopyConstruction) {
    Shape const shape{2, 2};
    constexpr auto initialValue = 0;
    constexpr auto afterIncrement = 23;

    auto x = fl::full(shape, initialValue);
    auto const y = x; // actual copy (implementation may be CoW)

    ASSERT_TRUE(allClose(x, fl::full(shape, initialValue)));
    ASSERT_TRUE(allClose(y, fl::full(shape, initialValue)));
    x += afterIncrement; // affects both tensors
    ASSERT_TRUE(allClose(x, fl::full(shape, afterIncrement)));
    ASSERT_TRUE(allClose(y, fl::full(shape, initialValue)));
}

TEST(TensorBaseTest, MoveConstruction) {
    Shape const shape{2, 2};
    constexpr auto initialValue = 0;
    constexpr auto afterMove = 42;

    auto x = fl::full(shape, initialValue);
    auto const y = x(span, span); // view of x

    auto z = std::move(x); // `z` takes over `x`'s data
    // TODO the following line (or any read to `y`, as it seems) promotes view to
    // copy; to avoid this, we must update impl of `assign`
    // ASSERT_TRUE(allClose(y, fl::full(shape, 0)));
    ASSERT_TRUE(allClose(z, fl::full(shape, initialValue)));

    z += afterMove; // `y` is now a view of `z`, so it's affected
    ASSERT_TRUE(allClose(y, fl::full(shape, afterMove)));
    ASSERT_TRUE(allClose(z, fl::full(shape, afterMove)));
}

TEST(TensorBaseTest, AssignmentOperatorLvalueWithRvalue) {
    Shape const shape{2, 2};
    constexpr auto initialValue = 0;
    constexpr auto assignedValue = 42;
    constexpr auto expectedAfterIncrement = 43;

    auto const x = fl::full(shape, initialValue);
    auto y = x(span, span);

    // view as a lvalue cannot be used to update original tensor
    y = fl::full(shape, assignedValue); // `x` isn't affected
    y += 1; // `x` isn't affected
    ASSERT_TRUE(allClose(x, fl::full(shape, initialValue)));
    ASSERT_TRUE(allClose(y, fl::full(shape, expectedAfterIncrement)));
}

TEST(TensorBaseTest, AssignmentOperatorLvalueWithLvalue) {
    Shape const shape{2, 2};
    constexpr auto initialValue = 0;
    constexpr auto value1 = 1;
    constexpr auto expectedAfterAssignment = 2;

    auto const x = fl::full(shape, initialValue);
    auto y = x(span, span);
    auto const z = fl::full(shape, value1);

    y = z; // `x` is a copy of `z` now (impl may be CoW)
    y += value1; // `z` isn't affected
    ASSERT_TRUE(allClose(x, fl::full(shape, initialValue)));
    ASSERT_TRUE(allClose(y, fl::full(shape, expectedAfterAssignment)));
    ASSERT_TRUE(allClose(z, fl::full(shape, value1)));
}

TEST(TensorBaseTest, AssignmentOperatorRvalueWithRvalue) {
    Shape const shape{2, 2};
    constexpr auto initialValue = 0;
    constexpr auto assignValue = 1;

    auto const type = dtype::f32;
    auto const x = fl::full(shape, initialValue, type);
    auto const y = x(span, span);

    x(0, span) = fl::full({2}, assignValue); // `x` is updated by copying from rhs data
    auto const res = fl::Tensor::fromVector<float>(shape, {1, 0, 1, 0}, type);
    ASSERT_TRUE(allClose(x, res));
    ASSERT_TRUE(allClose(y, res));
}

TEST(TensorBaseTest, AssignmentOperatorRvalueWithLvalue) {
    Shape const shape{2, 2};
    constexpr auto initialValue = 0;
    constexpr auto vectorValue = 1;
    constexpr auto expectedAfterIncrement = 2;

    auto const type = dtype::f32;
    auto x = fl::full(shape, initialValue, type);
    auto const y = x(span, span); // view of `x`
    auto const z = fl::full({2}, vectorValue, type);

    x(span, 1) = z; // `x` is updated by copying from `z`'s data
    x += vectorValue; // `z` isn't affected
    auto const res = fl::Tensor::fromVector<float>(shape, {1, 1, 2, 2}, type);
    ASSERT_TRUE(allClose(x, res));
    ASSERT_TRUE(allClose(y, res));
    ASSERT_TRUE(allClose(z, fl::full({2}, vectorValue, type)));
}

TEST(TensorBaseTest, Metadata) {
    int size = 9;
    auto const t = fl::rand({size, size});
    ASSERT_EQ(t.elements(), size * size);
    ASSERT_FALSE(t.isEmpty());
    ASSERT_EQ(t.bytes(), size * size * sizeof(float));

    Tensor const e;
    ASSERT_EQ(e.elements(), 0);
    ASSERT_TRUE(e.isEmpty());
    ASSERT_FALSE(e.isSparse());
    ASSERT_FALSE(e.isLocked());
}

TEST(TensorBaseTest, fromScalar) {
    constexpr auto scalarValue = 3.14;
    auto const type = fl::dtype::f32;

    Tensor const a = fromScalar(scalarValue, type);
    ASSERT_EQ(a.type(), type);
    ASSERT_EQ(a.elements(), 1);
    ASSERT_EQ(a.ndim(), 0);
    ASSERT_FALSE(a.isEmpty());
    ASSERT_EQ(a.shape(), Shape({}));
}

TEST(TensorBaseTest, string) {
    // Different backends might print tensors differently - check for consistency
    // across two identical tensors
    auto const a = fl::full({3, 4, 5}, 6.f);
    auto const b = fl::full({3, 4, 5}, 6.f);
    ASSERT_EQ(a.toString(), b.toString());

    std::stringstream ssa, ssb;
    ssa << a;
    ssb << b;
    ASSERT_EQ(ssa.str(), ssb.str());
}

TEST(TensorBaseTest, AssignmentOperators) {
    auto a = fl::full({3, 3}, 1.f);
    a += 2;
    ASSERT_TRUE(allClose(a, fl::full({3, 3}, 3.f)));
    a -= 1;
    ASSERT_TRUE(allClose(a, fl::full({3, 3}, 2.f)));
    a *= 8;
    ASSERT_TRUE(allClose(a, fl::full({3, 3}, 16.f)));
    a /= 4;
    ASSERT_TRUE(allClose(a, fl::full({3, 3}, 4.f)));

    a = fl::full({4, 4}, 7.f);
    ASSERT_TRUE(allClose(a, fl::full({4, 4}, 7.f)));
    auto const b = a;
    ASSERT_TRUE(allClose(b, fl::full({4, 4}, 7.f)));
    a = 6.;
    ASSERT_TRUE(allClose(a, fl::full({4, 4}, 6.f)));

    a = fl::full({5, 6, 7}, 8.f);
    ASSERT_TRUE(allClose(a, fl::full({5, 6, 7}, 8.f)));
}

TEST(TensorBaseTest, CopyOperators) {
    auto a = fl::full({3, 3}, 1.f);
    auto const b = a;
    a += 1;
    ASSERT_TRUE(allClose(b, fl::full({3, 3}, 1.f)));
    ASSERT_TRUE(allClose(a, fl::full({3, 3}, 2.f)));

    auto const c = a.copy();
    a += 1;
    ASSERT_TRUE(allClose(a, fl::full({3, 3}, 3.f)));
    ASSERT_TRUE(allClose(c, fl::full({3, 3}, 2.f)));
}

TEST(TensorBaseTest, ConstructFromData) {
    // Tensor::fromVector
    constexpr auto vectorSize = 100;
    float fillValue = 3.f;
    std::vector<float> vec(vectorSize, fillValue);
    fl::Shape bigShape = {10, 10};
    ASSERT_TRUE(allClose(fl::Tensor::fromVector(bigShape, vec), fl::full(bigShape, fillValue)));

    ASSERT_TRUE(
        allClose(
            fl::Tensor::fromBuffer(bigShape, vec.data(), fl::MemoryLocation::Host),
            fl::full(bigShape, fillValue)
        )
    );

    std::vector<float> ascending = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto t = fl::Tensor::fromVector({3, 4}, ascending);
    ASSERT_EQ(t.type(), fl::dtype::f32);
    for(int i = 0; i < ascending.size(); ++i)
        ASSERT_FLOAT_EQ(t(i % 3, i / 3).scalar<float>(), ascending[i]);

    // TODO: add fixtures/check stuff
    std::vector<int> intV = {1, 2, 3};
    ASSERT_EQ(fl::Tensor::fromVector({3}, intV).type(), fl::dtype::s32);
    ASSERT_EQ(
        fl::Tensor::fromVector<float>({5}, {0., 1., 2., 3., 4.}).type(),
        fl::dtype::f32
    );

    std::vector<float> flat = {0, 1, 2, 3, 4, 5, 6, 7};
    unsigned size = flat.size();
    ASSERT_EQ(fl::Tensor::fromVector(flat).shape(), Shape({size}));

    // Tensor::fromArray
    constexpr unsigned arrFSize = 5;
    std::array<float, arrFSize> arrF = {1, 2, 3, 4, 5};
    auto tArrF = Tensor::fromArray(arrF);
    ASSERT_EQ(tArrF.type(), fl::dtype::f32);
    ASSERT_EQ(tArrF.shape(), Shape({arrFSize}));
    auto tArrD = Tensor::fromArray({arrFSize}, arrF, fl::dtype::f64);
    ASSERT_EQ(tArrD.type(), fl::dtype::f64);

    constexpr unsigned arrISize = 8;
    std::array<unsigned, arrISize> arrI = {1, 2, 3, 4, 5, 6, 7, 8};
    auto tArrI = Tensor::fromArray(arrI);
    ASSERT_EQ(tArrI.type(), fl::dtype::u32);
    ASSERT_EQ(tArrI.shape(), Shape({arrISize}));
    auto tArrIs = Tensor::fromArray({2, 4}, arrI);
    ASSERT_EQ(tArrIs.shape(), Shape({2, 4}));
}

TEST(TensorBaseTest, reshape) {
    auto const a = fl::full({4, 4}, 3.f);
    auto const b = fl::reshape(a, Shape({8, 2}));
    ASSERT_EQ(b.shape(), Shape({8, 2}));
    ASSERT_TRUE(allClose(a, fl::reshape(b, {4, 4})));

    ASSERT_THROW(fl::reshape(a, {}), std::exception);
}

TEST(TensorBaseTest, transpose) {
    // TODO: expand to check els
    ASSERT_TRUE(
        allClose(fl::transpose(fl::full({3, 4}, 3.f)), fl::full({4, 3}, 3.f))
    );
    ASSERT_TRUE(
        allClose(
            fl::transpose(fl::full({4, 5, 6, 7}, 3.f), {2, 0, 1, 3}),
            fl::full({6, 4, 5, 7}, 3.f)
        )
    );
    ASSERT_THROW(fl::transpose(fl::rand({3, 4, 5}), {0, 1}), std::exception);
    ASSERT_THROW(
        fl::transpose(fl::rand({2, 4, 6, 8}), {1, 0, 2}),
        std::exception
    );
    ASSERT_THROW(
        fl::transpose(fl::rand({2, 4, 6, 8}), {1, 0, 2, 4}),
        std::exception
    );

    auto a = fl::rand({4});
    ASSERT_TRUE(allClose(fl::transpose(a), a));

    ASSERT_EQ(fl::transpose(fl::rand({5, 6, 7})).shape(), Shape({7, 6, 5}));
    ASSERT_EQ(fl::transpose(fl::rand({5, 6, 1, 7})).shape(), Shape({7, 1, 6, 5}));
    ASSERT_EQ(fl::transpose(fl::rand({1, 1})).shape(), Shape({1, 1}));
    ASSERT_EQ(
        fl::transpose(fl::rand({7, 2, 1, 3}), {0, 2, 1, 3}).shape(),
        Shape({7, 1, 2, 3})
    );
}

TEST(TensorBaseTest, tile) {
    auto const a = fl::full({4, 4}, 3.f);
    auto const tiled = fl::tile(a, {2, 2});
    ASSERT_EQ(tiled.shape(), Shape({8, 8}));
    ASSERT_TRUE(allClose(tiled, fl::full({8, 8}, 3.f)));
    ASSERT_EQ(fl::tile(a, {}).shape(), a.shape());

    auto const s = fl::fromScalar(3.14f);
    ASSERT_EQ(fl::tile(s, {3, 3}).shape(), Shape({3, 3}));
    ASSERT_EQ(fl::tile(s, {}).shape(), s.shape());
}



TEST(TensorBaseTest, nonzero) {
    std::vector<int> const idxs = {0, 1, 4, 9, 11, 23, 55, 82, 91};
    auto const a = fl::full({10, 10}, 1, fl::dtype::u32);
    for(const auto idx : idxs)
        a(idx / 10, idx % 10) = 0;
    auto const indices = fl::nonzero(a);
    int nnz = a.elements() - idxs.size();
    ASSERT_EQ(indices.shape(), Shape({nnz}));
    ASSERT_TRUE(
        allClose(a.flatten()(indices), fl::full({nnz}, 1, fl::dtype::u32))
    );
}

TEST(TensorBaseTest, flatten) {
    int size = 6;
    auto const a = fl::full({size, size, size}, 2.f);
    auto const flat = a.flatten();
    ASSERT_EQ(flat.shape(), Shape({size * size * size}));
    ASSERT_TRUE(allClose(flat, fl::full({size * size * size}, 2.f)));
}

TEST(TensorBaseTest, pad) {
    std::vector<float> data{
        {
            0.1f,
            0.2f,
            0.3f,
            0.4f,
            0.5f,
            1.1f,
            1.2f,
            1.3f,
            1.4f,
            1.5f
        }
    };


    auto const t = fl::Tensor::fromVector({5, 2}, data);
    //auto const t = fl::rand({5, 2});
    auto const actualZeroPad = fl::pad(t, {{1, 2}, {3, 4}});
    auto const expectedZeroPad = fl::concatenate(
        1,
        fl::full({8, 3}, 0.f),
        fl::concatenate(0, fl::full({1, 2}, 0.f), t, fl::full({2, 2}, 0.f)),
        fl::full({8, 4}, 0.f)
    );


    ASSERT_TRUE(allClose(actualZeroPad, expectedZeroPad));

    auto const actualEdgePad = fl::pad(t, {{1, 1}, {2, 2}}, PadType::Edge);
    auto const vertTiled = fl::concatenate(
        0,
        fl::reshape(t(0, fl::span), {1, 2}),
        t,
        fl::reshape(t(t.dim(0) - 1, fl::span), {1, 2})
    );
    auto vTiled0 = vertTiled(fl::span, 0);
    auto vTiled1 = vertTiled(fl::span, 1);

    auto const expectedEdgePad = fl::concatenate(
        1,
        fl::tile(vTiled0, {1, 3}),
        fl::tile(vTiled1, {1, 3})
    );

    ASSERT_TRUE(allClose(actualEdgePad, expectedEdgePad));


    auto const actualSymmetricPad = fl::pad(t, {{1, 1}, {2, 2}}, PadType::Symmetric);
    auto const expectedSymmetricPad = fl::concatenate(
        {vTiled1, vTiled0, vTiled0, fl::concatenate(1, vTiled1, vTiled1, vTiled0)},
        1
    );

    ASSERT_TRUE(allClose(actualSymmetricPad, expectedSymmetricPad));
}

TEST(TensorBaseTest, asType) {
    auto const a = fl::rand({3, 3});
    auto const size = 9;

    ASSERT_EQ(a.type(), dtype::f32);

    auto const aDouble = a.asType(dtype::f64);

    ASSERT_EQ(aDouble.type(), dtype::f64);

    auto const aData = a.host<float>();
    auto const bData = aDouble.host<double>();

    for(size_t i = 0; i < size; i++)
        ASSERT_NEAR(aData[i], bData[i], 1e-6);
}

TEST(TensorBaseTest, where) {
    constexpr auto threshold = 5;

    auto const a = Tensor::fromVector<int>({2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto const out = fl::where(a < threshold, a, a * 10);
    a(a >= threshold) *= 10;
    ASSERT_TRUE(allClose(out, a));
    auto const outC = fl::where(a < threshold, a, 3);
    a(a >= threshold) = 3;
    ASSERT_TRUE(allClose(outC, a));
    auto const outC2 = fl::where(a < threshold, 3, a);
    a(a < threshold) = 3;
    ASSERT_TRUE(allClose(outC2, a));

    // non b8-type vector throws
    EXPECT_THROW(
        fl::where((a < threshold).asType(fl::dtype::f32), a, a * 10),
        std::exception
    );
}

TEST(TensorBaseTest, topk) {
    constexpr auto k = 3;
    constexpr auto k4 = 4;

    auto const a = fl::arange({10, 2});
    Tensor values;
    Tensor indices;
    fl::topk(values, indices, a, /* k = */ k, /* axis = */ 0); // descending sort
    ASSERT_TRUE(
        allClose(values, Tensor::fromVector<float>({k, 2}, {9, 8, 7, 9, 8, 7}))
    );

    fl::topk(
        values,
        indices,
        a,
        /* k = */
        k4,
        /* axis = */
        0,
        fl::SortMode::Ascending
    );
    ASSERT_TRUE(
        allClose(
            values,
            Tensor::fromVector<float>({k4, 2}, {0, 1, 2, 3, 0, 1, 2, 3})
        )
    );
}

TEST(TensorBaseTest, sort) {
    Shape dims({10, 2});
    auto const a = fl::arange(dims);
    auto const sorted = fl::sort(a, /* axis = */ 0, SortMode::Descending);

    Tensor const expected({dims[0]}, a.type());
    for(int i = 0; i < dims[0]; ++i)
        expected(i) = dims[0] - i - 1;
    auto const tiled = fl::tile(expected, {1, 2});
    ASSERT_TRUE(allClose(sorted, tiled));

    ASSERT_TRUE(allClose(a, fl::sort(tiled, 0, SortMode::Ascending)));

    auto const b = fl::rand({10});
    Tensor values, indices;
    fl::sort(values, indices, b, /* axis = */ 0, SortMode::Descending);
    ASSERT_TRUE(
        allClose(values, fl::sort(b, /* axis = */ 0, SortMode::Descending))
    );
    ASSERT_TRUE(
        allClose(fl::argsort(b, /* axis = */ 0, SortMode::Descending), indices)
    );
}

TEST(TensorBaseTest, argsort) {
    Shape dims({10, 2});
    auto const a = fl::arange(dims);
    auto const sorted = fl::argsort(a, /* axis = */ 0, SortMode::Descending);

    Tensor const expected({dims[0]}, fl::dtype::u32);
    for(int i = 0; i < dims[0]; ++i)
        expected(i) = dims[0] - i - 1;
    auto const tiled = fl::tile(expected, {1, 2});
    ASSERT_TRUE(allClose(sorted, tiled));

    ASSERT_TRUE(allClose(tiled, fl::argsort(tiled, 0, SortMode::Ascending)));
}

template<typename ScalarArgType>
void assertScalarBehavior(fl::dtype type) {
    ScalarArgType scalar = 42; // small enough for any scalar type
    auto one = fl::full({1}, scalar, type);

    if(dtype_traits<ScalarArgType>::fl_type != type) {
        ASSERT_THROW(one.template scalar<ScalarArgType>(), std::logic_error)
            << "dtype: " << type
            << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();
        return;
    }

    if(
        (type == fl::dtype::f16) || (type == fl::dtype::f32)
        || (type == fl::dtype::f64)
    )
        ASSERT_FLOAT_EQ(one.template scalar<ScalarArgType>(), scalar)
            << "dtype: " << type
            << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();
    else
        ASSERT_EQ(one.template scalar<ScalarArgType>(), scalar)
            << "dtype: " << type
            << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();

    auto val = static_cast<ScalarArgType>(rand());
    auto a = fl::full({5, 6}, val, type);

    ASSERT_TRUE(allClose(fl::full({1}, a.template scalar<ScalarArgType>(), type), a(0, 0)))
        << "dtype: " << type
        << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();
}

TEST(TensorBaseTest, scalar) {
    auto const types = {
        fl::dtype::b8,
        fl::dtype::u8,
        fl::dtype::s16,
        fl::dtype::u16,
        fl::dtype::s32,
        fl::dtype::u32,
        fl::dtype::s64,
        fl::dtype::u64,
        fl::dtype::f16,
        fl::dtype::f32,
        fl::dtype::f64
    };
    for(auto const type : types) {
        assertScalarBehavior<char>(type);
        assertScalarBehavior<unsigned char>(type);
        assertScalarBehavior<short>(type);
        assertScalarBehavior<unsigned short>(type);
        assertScalarBehavior<int>(type);
        assertScalarBehavior<unsigned int>(type);
        assertScalarBehavior<long>(type);
        assertScalarBehavior<unsigned long>(type);
        assertScalarBehavior<long long>(type);
        assertScalarBehavior<unsigned long long>(type);
        assertScalarBehavior<float>(type);
        assertScalarBehavior<double>(type);
    }
}

TEST(TensorBaseTest, isContiguous) {
    // Contiguous by default
    auto const a = fl::rand({10, 10});
    ASSERT_TRUE(a.isContiguous());
}

TEST(TensorBaseTest, strides) {
    auto const t = fl::rand({10, 10});
    ASSERT_EQ(t.strides(), Shape({1, 10}));
}

TEST(TensorBaseTest, stream) {
    auto const t1 = fl::rand({10, 10});
    auto const t2 = -t1;
    auto const t3 = t1 + t2;
    ASSERT_EQ(&t1.stream(), &t2.stream());
    ASSERT_EQ(&t1.stream(), &t3.stream());
}

TEST(TensorBaseTest, asContiguousTensor) {
    auto const t = fl::rand({5, 6, 7, 8});
    auto const indexed = t(
        fl::range(1, 4, 2),
        fl::range(0, 6, 2),
        fl::range(0, 7, 3),
        fl::range(0, 5, 3)
    );

    auto const contiguous = indexed.asContiguousTensor();
    std::vector<Dim> strides;
    unsigned stride = 1;
    for(unsigned i = 0; i < contiguous.ndim(); ++i) {
        strides.push_back(stride);
        stride *= contiguous.dim(i);
    }
    ASSERT_EQ(contiguous.strides(), Shape(strides));
}

TEST(TensorBaseTest, raw_host) {
    auto const a = fl::rand({10, 10}, fl::dtype::f32);

    std::unique_ptr<float[]> ptr{static_cast<float*>(a.raw_host())};
    for(int i = 0; i < a.elements(); ++i)
        ASSERT_EQ(ptr[i], a.flatten()(i).scalar<float>());


    std::vector<float> existingBuffer(100);
    a.raw_host(existingBuffer.data());
    for(int i = 0; i < a.elements(); ++i)
        ASSERT_EQ(existingBuffer[i], a.flatten()(i).scalar<float>());

    ASSERT_EQ(Tensor().raw_host(), nullptr);
}

TEST(TensorBaseTest, host) {
    auto const a = fl::rand({10, 10});
    auto const vec = a.host<float>();

    for(int i = 0; i < a.elements(); ++i)
        ASSERT_EQ(vec[i], a.flatten()(i).scalar<float>());

    ASSERT_EQ(Tensor().host<float>().size(), 0);
}

TEST(TensorBaseTest, arange) {
    // Range/step overload
    ASSERT_TRUE(
        allClose(fl::arange(2, 10, 2), Tensor::fromVector<int>({2, 4, 6, 8}))
    );
    ASSERT_TRUE(
        allClose(fl::arange(0, 6), Tensor::fromVector<int>({0, 1, 2, 3, 4, 5}))
    );
    ASSERT_TRUE(
        allClose(
            fl::arange(0.f, 1.22f, 0.25f),
            Tensor::fromVector<float>({0.f, 0.25f, 0.5f, 0.75f})
        )
    );
    ASSERT_TRUE(
        allClose(
            fl::arange(0.f, 4.1f),
            Tensor::fromVector<float>({0.f, 1.f, 2.f, 3.f})
        )
    );

    // Shape overload
    auto const v = Tensor::fromVector<float>({0.f, 1.f, 2.f, 3.f});
    ASSERT_TRUE(allClose(fl::arange({4}), v));

    ASSERT_TRUE(allClose(fl::arange({4, 5}), fl::tile(v, {1, 5})));
    ASSERT_EQ(fl::arange({4, 5}, 1).shape(), Shape({4, 5}));
    ASSERT_TRUE(
        allClose(
            fl::arange({4, 5}, 1),
            fl::tile(
                fl::reshape(Tensor::fromVector<float>({0.f, 1.f, 2.f, 3.f, 4.f}), {1, 5}),
                {4}
            )
        )
    );
    ASSERT_EQ(fl::arange({2, 6}, 0, fl::dtype::f64).type(), fl::dtype::f64);
}

TEST(TensorBaseTest, iota) {
    ASSERT_TRUE(
        allClose(
            fl::iota({5, 3}, {1, 2}),
            fl::tile(fl::reshape(fl::arange({15}), {5, 3}), {1, 2})
        )
    );
    ASSERT_EQ(fl::iota({2, 2}, {2, 2}, fl::dtype::f64).type(), fl::dtype::f64);
    ASSERT_EQ(fl::iota({1, 10}, {5}).shape(), Shape({5, 10}));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    fl::init();
    return RUN_ALL_TESTS();
}

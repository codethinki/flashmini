/*
 * SPDX-License-Identifier: MIT
 *
 * Original code: Copyright (c) Meta Platforms, Inc. (see FLASHLIGHT_LICENSE)
 * Modifications: Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */

#pragma once

#include <initializer_list>
#include <limits>
#include <ostream>
#include <utility>
#include <vector>

#include "flashlight/fl/common/Defines.h"

namespace fl {

// The type of a dimension.
using dim_t = int64_t;
using Dim = dim_t;

/**
 * An object describing the dimensions of a tensor.
 *
 * The dimensions and sizes of a shape are explicit; where some tensor libraries
 * implement implicit dimensions (i.e. those that are 1 are ignored), Flashlight
 * Shapes can be of arbitrary size and 1-dimensions distinguish them.
 * Concretely, (3, 1) and (3) are distinct shapes. See ShapeTest for further
 * examples.
 *
 * Shapes dimensions should be >= 1 in size. Shapes with a zero dimension have
 * zero elements, even if other dimensions are of nonzero size. For example: a
 * Shape of (0) has zero elements, as does a Shape with dimensions (1, 2, 3, 0).
 *
 * Different tensor backends implement different shape and dimension semantics.
 * As such, these need to be converted back and forth to and from Flashlight
 * Shapes. Having a common set of behaviors in this API ensures that tensors and
 * their shapes can be freely-manipulated across tensor backends.
 *
 * Shape is an interface and can be derived from or implemented given specific
 * backing storage or handles.
 */
class FL_API Shape {
    // Storage for the dimension values. Defaults to an empty Shape {0}, whereas
    // {} is a scalar shape.
    std::vector<Dim> _dims;

    /**
     * Check if a dimension is valid (i.e. in bounds) given the current size of
     * the shape. If not valid, throws an exception.
     */
    void checkDimsOrThrow(size_t dim) const;

public:
    Shape() = default;
    ~Shape() = default;
    /**
     * Gives the maximum number of dimensions a tensor of a particular shape can
     * have.
     *
     * If the maximum size can be arbitrarily high, `std::numeric_limits<Dim>`
     * should be used.
     */
    static constexpr size_t kMaxDims = std::numeric_limits<size_t>::max();

    /**
     * Initialize a Shape via a vector.
     */
    explicit Shape(std::vector<Dim> d);

    /**
     * Initialize a Shape via an initializer list.
     */
    /* implicit */
    Shape(std::initializer_list<Dim> d);

    /**
     * @return the number of elements in a tensor that has the given shape.
     */
    Dim elements() const;

    /**
     * @return Number of dimensions in the shape.
     */
    Dim ndim() const;

    /**
     * Get the size of a given dimension in the number of arguments. Throws if the
     * given dimension is larger than the number of dimensions.
     *
     * @return the number of elements at the given dimension
     */
    Dim dim(size_t dim) const;

    /**
     * Returns a reference to the given index
     */
    Dim& operator[](size_t dim);
    Dim const& operator[](size_t dim) const;

    /**
     * Compares two shapes. Returns true if their dim vectors are equal.
     */
    bool operator==(Shape const& other) const;
    bool operator!=(Shape const& other) const;

    /**
     * Compare a shape to an initializer list.
     */
    bool operator==(std::initializer_list<Dim> const& other) const;
    bool operator!=(std::initializer_list<Dim> const& other) const;

    /**
     * Gets a reference to the underlying dims vector.
     */
    std::vector<Dim> const& get() const;
    std::vector<Dim>& get();

    /**
     * Returns a string representation of the Shape
     */
    std::string toString() const;
};

/**
 * Write a shape representation to an output stream.
 */
FL_API std::ostream& operator<<(std::ostream& ostr, Shape const& s);


/**
 * Composes two shapes with the given operation. 
 * @param first shape
 * @param second shape
 * @tparam Op to apply to elements
 * @tparam ExtendVal shapes of unequal size will be implicitly extended with this
 * @return element wise composition
 */
template<auto Op, Dim ExtendVal = 0>
FL_API Shape element_compose_op(Shape const& first, Shape const& second) {
    auto& large = first.ndim() < second.ndim() ? second : first;
    auto const outDim = large.ndim();
    auto const sharedDims = std::min(first.ndim(), second.ndim());

    std::vector<Dim> resultData(outDim);


    for(int i = 0; i < sharedDims; i++)
        resultData[i] = Op(first[i], second[i]);

    for(int i = sharedDims; i < outDim; i++)
        resultData[i] = Op(large[i], ExtendVal);

    return Shape{resultData};
}


/**
 * Performs element wise max.
 * @param first shape
 * @param second shape
 * @return element wise max composition
 * @details shapes of unequal size will be extended with 0
 */
FL_API inline Shape max(Shape const& first, Shape const& second) {
    constexpr auto max_op = [](Dim x, Dim y) { return std::max(x, y); };

    if(first.ndim() == 0)
        return second;
    if(second.ndim() == 0)
        return first;

    return element_compose_op<max_op>(first, second);
}

} // namespace fl

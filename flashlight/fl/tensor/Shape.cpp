/*
 * SPDX-License-Identifier: MIT
 *
 * Original code: Copyright (c) Meta Platforms, Inc. (see FLASHLIGHT_LICENSE)
 * Modifications: Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */

#include "flashlight/fl/tensor/Shape.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace fl {

Shape::Shape(std::vector<Dim> d) : _dims(std::move(d)) {}
Shape::Shape(std::initializer_list<Dim> d) : Shape(std::vector<Dim>(d)) {}

Dim const kEmptyShapeNumberOfElements = 1;

void Shape::checkDimsOrThrow(size_t const dim) const {
    if(dim > ndim() - 1) {
        std::stringstream ss;
        ss << "Shape index " << std::to_string(dim)
            << " out of bounds for shape with " << std::to_string(_dims.size())
            << " dimensions.";
        throw std::invalid_argument(ss.str());
    }
}

Dim Shape::elements() const {
    if(_dims.empty())
        return kEmptyShapeNumberOfElements;
    return std::accumulate(_dims.begin(), _dims.end(), static_cast<Dim>(1), std::multiplies<Dim>());
}

size_t Shape::ndim() const { return _dims.size(); }

Dim Shape::dim(size_t const dim) const {
    checkDimsOrThrow(dim);
    return _dims[dim];
}

Dim& Shape::operator[](size_t const dim) {
    checkDimsOrThrow(dim);
    return _dims[dim];
}

Dim const& Shape::operator[](size_t const dim) const {
    checkDimsOrThrow(dim);
    return _dims[dim];
}

bool Shape::operator==(Shape const& other) const { return _dims == other._dims; }

bool Shape::operator!=(Shape const& other) const { return !(this->operator==(other)); }

bool Shape::operator==(std::initializer_list<Dim> const& other) const {
    return _dims.size() == other.size()
        && std::equal(std::begin(_dims), std::end(_dims), std::begin(other));
}

bool Shape::operator!=(std::initializer_list<Dim> const& other) const { return !(this->operator==(other)); }

std::vector<Dim> const& Shape::get() const { return _dims; }

std::vector<Dim>& Shape::get() { return _dims; };

std::string Shape::toString() const {
    std::stringstream ss;
    ss << "(";
    for(size_t i = 0; i < ndim(); ++i)
        ss << dim(i) << (i == ndim() - 1 ? "" : ", ");
    ss << ")";
    return ss.str();
}

std::ostream& operator<<(std::ostream& ostr, Shape const& s) {
    ostr << s.toString();
    return ostr;
}

} // namespace fl

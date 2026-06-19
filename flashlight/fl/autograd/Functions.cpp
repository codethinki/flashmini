/*
 * SPDX-License-Identifier: MIT
 *
 * Original code: Copyright (c) Meta Platforms, Inc. (see FLASHLIGHT_LICENSE)
 * Modifications: Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/autograd/tensor/AutogradExtension.h"
#include "flashlight/fl/autograd/tensor/AutogradOps.h"
#include "flashlight/fl/common/DynamicBenchmark.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {
namespace detail {

    Tensor tileAs(Tensor const& input, Shape const& rdims) {
        // Scalar tensor
        if(input.ndim() == 0)
            return tile(input, rdims);

        Shape dims(std::vector<Dim>(rdims.ndim(), 1));
        Shape idims = input.shape();
        for(int i = 0; i < rdims.ndim(); i++) {
            int idimsSize = i + 1 > idims.ndim() ? 1 : idims[i];
            if(rdims[i] % idimsSize != 0) {
                std::stringstream ss;
                ss << "Invalid dims for tileAs for input dims " << idims
                    << " to output dims " << rdims;
                throw std::invalid_argument(ss.str());
            }
            dims[i] = rdims[i] / idimsSize;
        }
        return tile(input, dims);
    }

    Tensor sumAs(Tensor const& input, Shape const& rdims) {
        Shape idims = input.shape();
        auto result = input;
        for(int i = 0; i < input.ndim(); i++)
            if(i + 1 > rdims.ndim() || idims[i] != rdims[i])
                result = fl::sum(result, {i}, /* keepDims = */ true);

        return fl::reshape(result.asType(input.type()), rdims);
    }

    Shape expandedShapeFromReducedDims(
        Tensor const& input,
        std::vector<int> const& axes,
        bool keepDims /* = false */
    ) {
        // Fast path - tensor already retained its shape
        if(keepDims)
            return input.shape();
        // If we output a scalar,
        if(input.ndim() == 0)
            return {};

        unsigned preNDims = input.ndim() + axes.size();
        Shape newShape(std::vector<Dim>(preNDims, 1));
        unsigned axesIdx = 0;
        unsigned inputIdx = 0;
        for(unsigned i = 0; i < preNDims; ++i) {
            if(i == axes[axesIdx])
            // This dim was reduced over, leave as 1 in the new shape
                axesIdx++;
            else {
                // Dim wasn't reduced over - add the shape from the new tensor
                newShape[i] = input.dim(inputIdx);
                inputIdx++;
            }
        }
        return newShape;
    }

    // TODO: remove these/use a simple template
    Variable expandFromReduction(
        Variable const& input,
        std::vector<int> const& axes,
        bool keepDims
    ) {
        return moddims(
            input,
            expandedShapeFromReducedDims(input.tensor(), axes, keepDims)
        );
    }

    Tensor expandFromReduction(
        Tensor const& input,
        std::vector<int> const& axes,
        bool keepDims
    ) {
        auto o = expandedShapeFromReducedDims(input, axes, keepDims);
        return fl::reshape(
            input,
            expandedShapeFromReducedDims(input, axes, keepDims)
        );
    }

    bool areVariableTypesEqual(Variable const& a, Variable const& b) { return a.type() == b.type(); }

} // namespace detail

Variable operator+(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    auto result = lhs.tensor() + rhs.tensor();
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(Variable(gradOutput.tensor(), false));
        inputs[1].addGrad(Variable(gradOutput.tensor(), false));
    };
    return Variable(result, {lhs.withoutData(), rhs.withoutData()}, gradFunc);
}

Variable operator+(Variable const& lhs, double const& rhsVal) {
    auto result = (lhs.tensor() + rhsVal).asType(lhs.type());
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(Variable(gradOutput.tensor(), false));
    };
    return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator+(double const& lhsVal, Variable const& rhs) { return rhs + lhsVal; }

Variable operator-(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    auto result = lhs.tensor() - rhs.tensor();
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(Variable(gradOutput.tensor(), false));
        inputs[1].addGrad(Variable(negate(gradOutput).tensor(), false));
    };
    return Variable(result, {lhs.withoutData(), rhs.withoutData()}, gradFunc);
}

Variable operator-(Variable const& lhs, double const& rhsVal) {
    auto result = (lhs.tensor() - rhsVal).asType(lhs.type());
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(Variable(gradOutput.tensor(), false));
    };
    return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator-(double const& lhsVal, Variable const& rhs) {
    auto result = (lhsVal - rhs.tensor()).asType(rhs.type());
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(Variable(negate(gradOutput).tensor(), false));
    };
    return Variable(result, {rhs.withoutData()}, gradFunc);
}

Variable operator*(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    auto result = lhs.tensor() * rhs.tensor();
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        if(inputs[0].isCalcGrad())
            inputs[0].addGrad(
                Variable(gradOutput.tensor() * inputs[1].tensor(), false)
            );
        if(inputs[1].isCalcGrad())
            inputs[1].addGrad(
                Variable(gradOutput.tensor() * inputs[0].tensor(), false)
            );
    };
    return Variable(
        result,
        {
            rhs.isCalcGrad() ? lhs : lhs.withoutData(),
            lhs.isCalcGrad() ? rhs : rhs.withoutData()
        },
        gradFunc
    );
}

Variable operator*(Variable const& lhs, double const& rhsVal) {
    auto result = (lhs.tensor() * rhsVal).asType(lhs.type());
    auto gradFunc = [rhsVal](std::vector<Variable>& inputs, Variable const& gradOutput) {
        inputs[0].addGrad(Variable(gradOutput.tensor() * rhsVal, false));
    };
    return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator*(double const& lhsVal, Variable const& rhs) { return rhs * lhsVal; }

Variable operator/(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    auto result = lhs.tensor() / rhs.tensor();
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto inputs1rec = reciprocal(inputs[1]);
        auto gradInput0 = gradOutput * inputs1rec;
        if(inputs[0].isCalcGrad())
            inputs[0].addGrad(Variable(gradInput0.tensor(), false));
        if(inputs[1].isCalcGrad())
            inputs[1].addGrad(
                Variable(
                    (gradInput0 * negate(inputs[0]) * inputs1rec).tensor(),
                    false
                )
            );
    };
    return Variable(
        result,
        {rhs.isCalcGrad() ? lhs : lhs.withoutData(), rhs},
        gradFunc
    );
}

Variable operator/(Variable const& lhs, double const& rhsVal) {
    auto result = (lhs.tensor() / rhsVal).asType(lhs.type());
    auto gradFunc =
        [rhsVal](std::vector<Variable>& inputs, Variable const& gradOutput) {
        inputs[0].addGrad(Variable((gradOutput / rhsVal).tensor(), false));
    };
    return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator/(double const& lhsVal, Variable const& rhs) {
    auto result = (lhsVal / rhs.tensor()).asType(rhs.type());
    auto gradFunc = [lhsVal](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(
            Variable(
                (gradOutput * (-lhsVal) / (inputs[0] * inputs[0])).tensor(),
                false
            )
        );
    };
    return Variable(result, {rhs}, gradFunc);
}

Variable operator>(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    auto result = lhs.tensor() > rhs.tensor();
    return Variable(result, false);
}

Variable operator>(Variable const& lhs, double const& rhsVal) {
    auto result = (lhs.tensor() > rhsVal).asType(lhs.type());
    return Variable(result, false);
}

Variable operator>(double const& lhsVal, Variable const& rhs) {
    auto result = (lhsVal > rhs.tensor()).asType(rhs.type());
    return Variable(result, false);
}

Variable operator<(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    auto result = lhs.tensor() < rhs.tensor();
    return Variable(result, false);
}

Variable operator<(Variable const& lhs, double const& rhsVal) {
    auto result = (lhs.tensor() < rhsVal).asType(lhs.type());
    return Variable(result, false);
}

Variable operator<(double const& lhsVal, Variable const& rhs) {
    auto result = (lhsVal < rhs.tensor()).asType(rhs.type());
    return Variable(result, false);
}

Variable operator>=(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    auto result = lhs.tensor() >= rhs.tensor();
    return Variable(result, false);
}

Variable operator>=(Variable const& lhs, double const& rhsVal) {
    auto result = (lhs.tensor() >= rhsVal).asType(lhs.type());
    return Variable(result, false);
}

Variable operator>=(double const& lhsVal, Variable const& rhs) {
    auto result = (lhsVal >= rhs.tensor()).asType(rhs.type());
    return Variable(result, false);
}

Variable operator<=(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    auto result = lhs.tensor() <= rhs.tensor();
    return Variable(result, false);
}

Variable operator<=(Variable const& lhs, double const& rhsVal) {
    auto result = (lhs.tensor() <= rhsVal).asType(lhs.type());
    return Variable(result, false);
}

Variable operator<=(double const& lhsVal, Variable const& rhs) {
    auto result = (lhsVal <= rhs.tensor()).asType(rhs.type());
    return Variable(result, false);
}

Variable operator&&(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    auto result = lhs.tensor() && rhs.tensor();
    return Variable(result, false);
}

Variable operator!(Variable const& input) {
    auto result = (!input.tensor()).asType(input.type());
    return Variable(result, false);
}

Variable max(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    auto result = fl::maximum(lhs.tensor(), rhs.tensor());
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto mask = Variable(
            (inputs[0].tensor() > inputs[1].tensor()).asType(gradOutput.type()),
            false
        );
        inputs[0].addGrad(Variable((mask * gradOutput).tensor(), false));
        inputs[1].addGrad(Variable((!mask * gradOutput).tensor(), false));
    };
    return Variable(result, {lhs, rhs}, gradFunc);
}

Variable max(Variable const& lhs, double const& rhsVal) {
    auto result = fl::maximum(lhs.tensor(), rhsVal).asType(lhs.type());
    auto gradFunc =
        [rhsVal](std::vector<Variable>& inputs, Variable const& gradOutput) {
        auto mask = Variable(
            (inputs[0].tensor() > rhsVal).asType(gradOutput.type()),
            false
        );
        inputs[0].addGrad(Variable((mask * gradOutput).tensor(), false));
    };
    return Variable(result, {lhs}, gradFunc);
}

Variable max(double const& lhsVal, Variable const& rhs) { return max(rhs, lhsVal); }

Variable min(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    auto result = fl::minimum(lhs.tensor(), rhs.tensor());
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto mask = Variable(
            (inputs[0].tensor() < inputs[1].tensor()).asType(gradOutput.type()),
            false
        );
        inputs[0].addGrad(Variable((mask * gradOutput).tensor(), false));
        inputs[1].addGrad(Variable((!mask * gradOutput).tensor(), false));
    };
    return Variable(result, {lhs, rhs}, gradFunc);
}

Variable min(Variable const& lhs, double const& rhsVal) {
    auto result = fl::minimum(lhs.tensor(), rhsVal).asType(lhs.type());
    auto gradFunc =
        [rhsVal](std::vector<Variable>& inputs, Variable const& gradOutput) {
        auto mask = Variable(
            (inputs[0].tensor() < rhsVal).asType(gradOutput.type()),
            false
        );
        inputs[0].addGrad(Variable((mask * gradOutput).tensor(), false));
    };
    return Variable(result, {lhs}, gradFunc);
}

Variable min(double const& lhsVal, Variable const& rhs) { return min(rhs, lhsVal); }

Variable negate(Variable const& input) {
    auto result = (0.0 - input.tensor()).asType(input.type());
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(Variable(negate(gradOutput).tensor(), false));
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable reciprocal(Variable const& input) {
    auto result = 1.0 / FL_ADJUST_INPUT_TYPE(input.tensor());
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto res = reciprocal(inputs[0]);
        inputs[0].addGrad(
            Variable((negate(gradOutput) * res * res).tensor(), false)
        );
    };
    return Variable(result, {input}, gradFunc);
}

Variable exp(Variable const& input) {
    auto result = fl::exp(FL_ADJUST_INPUT_TYPE(input.tensor()));
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(
            Variable(gradOutput.tensor() * fl::exp(inputs[0].tensor()), false)
        );
    };
    return Variable(result, {input}, gradFunc);
}

Variable log(Variable const& input) {
    auto result = fl::log(FL_ADJUST_INPUT_TYPE(input.tensor()));
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(
            Variable((gradOutput.tensor() / inputs[0].tensor()), false)
        );
    };
    return Variable(result, {input}, gradFunc);
}

Variable log1p(Variable const& input) {
    auto result = fl::log1p(FL_ADJUST_INPUT_TYPE(input.tensor()));
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(
            Variable((gradOutput.tensor() / (1.0 + inputs[0].tensor())), false)
        );
    };
    return Variable(result, {input}, gradFunc);
}

Variable pow(Variable const& input, double p) {
    auto result = fl::power(FL_ADJUST_INPUT_TYPE(input.tensor()), p);
    auto gradFunc = [p](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        Tensor grad =
            p * fl::power(inputs[0].tensor(), p - 1) * gradOutput.tensor();
        inputs[0].addGrad(Variable(grad, false));
    };
    return Variable(result, {input}, gradFunc);
}

Variable sin(Variable const& input) {
    auto result = fl::sin(input.tensor());
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(
            Variable((gradOutput.tensor() * cos(inputs[0].tensor())), false)
        );
    };
    return Variable(result, {input}, gradFunc);
}

Variable cos(Variable const& input) {
    auto result = fl::cos(input.tensor());
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(
            Variable(
                (gradOutput.tensor() * negative(sin(inputs[0].tensor()))),
                false
            )
        );
    };
    return Variable(result, {input}, gradFunc);
}

Variable tanh(Variable const& input) {
    auto result = fl::tanh(input.tensor());
    auto gradFunc =
        [result](std::vector<Variable>& inputs, Variable const& gradOutput) {
        auto grad =
            Variable((1.0 - result * result) * gradOutput.tensor(), false);
        inputs[0].addGrad(Variable(grad.tensor(), false));
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable clamp(Variable const& input, double const lo, double const hi) {
    auto result = fl::clip(input.tensor(), lo, hi);
    auto gradFunc = [lo, hi, result](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        Tensor gradMask = gradOutput.tensor();
        gradMask = fl::where((result > lo) && (result < hi), gradMask, 0);
        inputs[0].addGrad(Variable(gradMask, false));
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable sqrt(Variable const& input) {
    auto result = fl::sqrt(input.tensor());
    auto gradFunc = [result](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto output = Variable(result, false);
        inputs[0].addGrad(Variable((gradOutput / (2 * output)).tensor(), false));
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable sigmoid(Variable const& input) {
    auto result = fl::sigmoid(input.tensor());
    auto gradFunc =
        [result](std::vector<Variable>& inputs, Variable const& gradOutput) {
        auto grad = gradOutput.tensor() * result * (1 - result);
        inputs[0].addGrad(Variable(grad, false));
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable swish(Variable const& input, double beta) { return input * sigmoid(beta * input); }

Variable erf(Variable const& input) {
    auto result = fl::erf(FL_ADJUST_INPUT_TYPE(input.tensor()));
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto x = inputs[0].tensor();
        auto grad = gradOutput.tensor() * 2 / std::sqrt(M_PI) * fl::exp(-(x * x));
        inputs[0].addGrad(Variable(grad, false));
    };
    return Variable(result, {input}, gradFunc);
}

Variable transpose(Variable const& input, Shape const& dims /* = {} */) {
    auto result = fl::transpose(input.tensor(), dims);
    auto gradFunc = [inputDims = input.shape(), ndim = input.ndim(), dims](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        Shape reverseShape = dims;

        if(dims.ndim()) {
            // Reverse vec if transposing all dims (empty arg)
            auto dVec = dims.get();
            std::reverse(dVec.begin(), dVec.end());
            reverseShape = Shape(dVec);
        }

        for(unsigned i = 0; i < reverseShape.ndim(); ++i)
            reverseShape[dims[i]] = i;

        inputs[0].addGrad(
            Variable(fl::transpose(gradOutput.tensor(), reverseShape), false)
        );
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable tileAs(Variable const& input, Shape const& rdims) {
    auto result = detail::tileAs(input.tensor(), rdims);

    Shape inDims = input.shape();
    auto gradFunc = [inDims](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(
            Variable(
                sumAs(gradOutput, inDims).tensor().asType(inputs[0].type()),
                false
            )
        );
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable tileAs(Variable const& input, Variable const& reference) { return tileAs(input, reference.shape()); }

Variable sumAs(Variable const& input, Shape const& rdims) {
    auto result = detail::sumAs(FL_ADJUST_INPUT_TYPE(input.tensor()), rdims);
    auto idims = input.tensor().shape();
    auto gradFunc =
        [idims](std::vector<Variable>& inputs, Variable const& gradOutput) {
        inputs[0].addGrad(Variable(tileAs(gradOutput, idims).tensor(), false));
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable sumAs(Variable const& input, Variable const& reference) { return sumAs(input, reference.shape()); }

Variable concatenate(std::vector<Variable> const& concatInputs, int dim) {
    if(concatInputs.empty())
        throw std::invalid_argument("cannot concatenate zero variables");

    if(concatInputs.size() == 1)
        return concatInputs[0];
    // All Variables must be of the same type
    fl::dtype type = concatInputs[0].type();
    for(auto& var : concatInputs)
        if(var.type() != type)
            throw std::invalid_argument(
                "concatenate: all input Variables must be of the same type"
            );
    // All Variables must have the same number of dims
    unsigned numDims = concatInputs[0].ndim();
    for(auto& var : concatInputs)
        if(numDims != var.ndim())
            throw std::invalid_argument(
                "concatenate: all input Variables must "
                "have the same number of dimensions"
            );

    // All Variables must have the same size when indexed along the dim not being
    // concatenated along
    auto dims = concatInputs[0].shape();
    int concatSize = dims[dim];
    for(int i = 1; i < concatInputs.size(); i++) {
        concatSize += concatInputs[i].dim(dim);
        for(int d = 0; d < numDims; d++)
            if(dim != d && concatInputs[i].dim(d) != dims[d])
                throw std::invalid_argument(
                    "mismatch in dimension not being concatenated"
                );
    }
    dims[dim] = concatSize;
    Tensor result(dims, concatInputs[0].type());
    std::vector<fl::Index> slice(numDims, fl::span);
    int start = 0;
    for(auto const& input : concatInputs) {
        slice[dim] = fl::range({start, start + input.dim(dim)});
        result(slice) = input.tensor();
        start += input.dim(dim);
    }

    std::vector<Variable> inputsNoData;
    std::vector<Shape> inDims;

    for(auto const& in : concatInputs) {
        inputsNoData.push_back(in.withoutData());
        inDims.push_back(in.shape());
    }

    auto gradFunc = [dim, inDims, numDims](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        std::vector<fl::Index> sx(numDims, fl::span);
        int s = 0;
        for(size_t i = 0; i < inputs.size(); ++i) {
            sx[dim] = fl::range(s, s + inDims[i][dim]);
            inputs[i].addGrad(Variable(gradOutput.tensor()(sx), false));
            s += inDims[i][dim];
        }
    };

    return Variable(result, inputsNoData, gradFunc);
}

std::vector<Variable> split(Variable const& input, int64_t splitSize, int dim) {
    if(splitSize <= 0)
        throw std::invalid_argument("split size must be a positive integer");
    auto dimSize = input.dim(dim);
    std::vector splitSizes(dimSize / splitSize, splitSize);

    if(dimSize % splitSize > 0)
        splitSizes.push_back(dimSize % splitSize);
    return split(input, splitSizes, dim);
}

std::vector<Variable> split(Variable const& input, std::vector<int64_t> const& splitSizes, int dim) {
    if(dim >= input.ndim())
        throw std::invalid_argument(
            "split: passed dim is larger than the number of dimensions "
            "of the input."
        );
    auto dimSize = input.dim(dim);
    auto N = splitSizes.size();

    std::vector<Variable> outputs(N);
    std::vector<fl::Index> sel(input.ndim(), fl::span);
    int start = 0;
    for(int i = 0; i < N; ++i) {
        if(splitSizes[i] <= 0)
            throw std::invalid_argument("elements in split sizes has to be positive");
        int end = start + splitSizes[i];
        sel[dim] = fl::range(start, end);
        outputs[i] = input(sel);
        start = end;
    }
    if(start != dimSize)
        throw std::invalid_argument("sum of split sizes must match split dim");
    return outputs;
}

Variable tile(Variable const& input, Shape const& dims) {
    Tensor result = fl::tile(input.tensor(), dims);
    Shape idims = input.shape();
    auto gradFunc =
        [idims](std::vector<Variable>& inputs, Variable const& gradOutput) {
        inputs[0].addGrad(
            Variable(
                sumAs(gradOutput, idims).tensor().asType(inputs[0].type()),
                false
            )
        );
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable sum(
    Variable const& input,
    std::vector<int> const& axes,
    bool keepDims /* = false*/
) {
    auto result = FL_ADJUST_INPUT_TYPE(input.tensor());
    result = fl::sum(result, axes, keepDims);

    Shape indims = input.shape();
    auto gradFunc = [indims, axes, keepDims](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(
            Variable(
                detail::tileAs(
                    detail::expandFromReduction(gradOutput.tensor(), axes, keepDims),
                    indims
                ),
                false
            )
        );
    };
    return Variable(result.asType(input.type()), {input.withoutData()}, gradFunc);
}

Variable mean(
    Variable const& input,
    std::vector<int> const& axes,
    bool keepDims /* = false*/
) {
    auto result = FL_ADJUST_INPUT_TYPE(input.tensor());
    result = mean(result, axes, keepDims);

    Shape idims = input.shape();
    auto gradFunc = [idims, axes, keepDims](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        Shape odims = gradOutput.shape();
        Dim count = 1;
        for(int i = 0; i < idims.ndim(); i++) {
            Dim odimSize = i + 1 > odims.ndim() ? 1 : odims[i];
            count *= idims[i] / odimSize;
        }
        auto grad =
            detail::tileAs(
                detail::expandFromReduction(gradOutput.tensor(), axes, keepDims),
                idims
            )
            / count;
        inputs[0].addGrad(
            Variable(
                detail::tileAs(
                    detail::expandFromReduction(gradOutput.tensor(), axes, keepDims),
                    idims
                )
                / count,
                false
            )
        );
    };

    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable var(
    Variable const& in,
    std::vector<int> const& axes,
    bool const isbiased /* = false */,
    bool keepDims /* = false*/
) {
    Tensor input = FL_ADJUST_INPUT_TYPE(in.tensor());
    auto result = sum(input * input, axes, keepDims);

    auto avg = fl::mean(input, axes, keepDims);
    auto n = 1;
    for(auto ax : axes)
        n *= input.dim(ax);
    if(!isbiased && n == 1)
        throw std::invalid_argument(
            "cannot compute unbiased variance with only one sample"
        );
    auto val = 1.0 / (isbiased ? n : n - 1);
    result = val * (result - n * avg * avg);

    auto gradFunc =
        [val, axes](std::vector<Variable>& inputs, Variable const& gradOutput) {
        Shape expandedDims = inputs[0].shape();
        Shape tileDims = inputs[0].shape();
        for(auto ax : axes) {
            tileDims[ax] = inputs[0].dim(ax);
            expandedDims[ax] = 1;
        }

        inputs[0].addGrad(
            Variable(
                ((2 * val * tileAs(moddims(gradOutput, expandedDims), tileDims))
                    * (inputs[0]
                        - tileAs(moddims(mean(inputs[0], axes), expandedDims), tileDims)))
                .tensor(),
                false
            )
        );
    };
    return Variable(result, {in}, gradFunc);
}

Variable norm(
    Variable const& input,
    std::vector<int> const& axes,
    double p /* = 2 */,
    bool keepDims /* = false */
) {
    if(p <= 0)
        throw std::out_of_range("Lp norm: p must be > 0");
    auto result = fl::power(fl::abs(FL_ADJUST_INPUT_TYPE(input.tensor())), p);
    result = fl::sum(result, axes, /* keepDims = */ keepDims);

    Tensor sumap = detail::expandFromReduction(result, axes, keepDims);
    result = fl::power(result, 1 / p);
    fl::eval(result);

    auto gradFunc = [sumap, p, axes, keepDims](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        // correct, but less precise: auto gvar = Variable(fl::power(result, p - 1),
        // false);
        auto gvar = Variable(fl::power(sumap, 1 - 1 / p), false);
        auto normGrad =
        (inputs[0].tensor() * fl::pow(fl::abs(inputs[0]), p - 2).tensor()
            * detail::tileAs(
                detail::expandFromReduction(gradOutput.tensor(), axes, keepDims)
                / gvar.tensor(),
                inputs[0].shape()
            ));
        inputs[0].addGrad(Variable(normGrad, false));
    };
    return Variable(result, {input}, gradFunc);
}

Variable normalize(
    Variable const& in,
    std::vector<int> const& axes,
    double p /* = 2 */,
    double eps /* = 1e-12 */
) {
    auto input = FL_ADJUST_INPUT_TYPE(in);
    Variable norm = fl::norm(input, axes, p);
    Variable invscale = max(norm, eps);
    return input / tileAs(invscale, input);
}

Variable matmul(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    // lhs:Input[0] -- [M, N]
    // rhs:Input[1] -- [N, K]
    // matmul(lhs, rhs)
    // -- matmul([M, N], [N, K]) --  [M, K]
    // result:gradOutput -- [M, K]
    auto result = fl::matmul(lhs.tensor(), rhs.tensor());
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        if(inputs[0].isCalcGrad()) {
            Tensor _lhs = gradOutput.tensor();
            if(_lhs.ndim() == 1)
                _lhs = fl::reshape(_lhs, {1, _lhs.dim(0)});
            Tensor _rhs = inputs[1].tensor();
            if(_rhs.ndim() == 1)
                _rhs = fl::reshape(_rhs, {_rhs.dim(0), 1});

            // matmulNT(gradOutput, inputs[1])
            // -- matmulNT([M, K], [N, K])
            // -- matmul([M, K], [K, N]) -- [M, K]
            auto val = fl::matmul(
                _lhs,
                _rhs,
                /* lhsProp = */
                MatrixProperty::None,
                /* rhsProp = */
                MatrixProperty::Transpose
            );
            inputs[0].addGrad(Variable(detail::sumAs(val, inputs[0].shape()), false));
        }
        if(inputs[1].isCalcGrad()) {
            Tensor _lhs = inputs[0].tensor();
            if(_lhs.ndim() == 1)
                _lhs = fl::reshape(_lhs, {1, _lhs.dim(0)});
            Tensor _rhs = gradOutput.tensor();
            if(_rhs.ndim() == 1)
                _rhs = fl::reshape(_rhs, {_rhs.dim(0), 1});

            // matmulTN(inputs[0], gradOutput)
            // -- matmulTN([M, N], [M, K])
            // -- matmul([N, M], [M, K]) -- [N, K]
            auto val = fl::matmul(
                _lhs,
                _rhs,
                /* lhsProp = */
                MatrixProperty::Transpose
            );
            inputs[1].addGrad(Variable(detail::sumAs(val, inputs[1].shape()), false));
        }
    };
    return Variable(result, {lhs, rhs}, gradFunc);
}

Variable matmulTN(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    // lhs:Input[0] -- [N, M]
    // rhs:Input[1] -- [N, K]
    // matmulTN(lhs, rhs)
    // -- matmulTN([N, M], [N, K])
    // -- matmul([M, N], [N, K]) -- [M, K]
    // result:gradOutput -- [M, K]
    auto result = fl::matmul(
        lhs.tensor(),
        rhs.tensor(),
        /* lhsProp = */
        MatrixProperty::Transpose
    );
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        if(inputs[0].isCalcGrad()) {
            // matmulNT(inputs[1], gradOutput)
            // -- matmulNT([N, K], [M, K])
            // -- matmul([N, K], [K, M]) -- [N, M]
            auto val = fl::matmul(
                inputs[1].tensor(),
                gradOutput.tensor(),
                /* lhsProp = */
                MatrixProperty::None,
                /* rhsProp = */
                MatrixProperty::Transpose
            );
            inputs[0].addGrad(Variable(detail::sumAs(val, inputs[0].shape()), false));
        }
        if(inputs[1].isCalcGrad()) {
            // matmul(inputs[0], gradOutput)
            // -- matmulNT([N, M], [M, K]) -- [N, K]
            auto val = fl::matmul(inputs[0].tensor(), gradOutput.tensor());
            inputs[1].addGrad(Variable(detail::sumAs(val, inputs[1].shape()), false));
        }
    };
    return Variable(result, {lhs, rhs}, gradFunc);
}

Variable matmulNT(Variable const& lhs, Variable const& rhs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
    // lhs:Input[0] -- [M, N]
    // rhs:Input[1] -- [K, N]
    // matmulNT(lhs, rhs)
    // -- matmulNT([M, N], [K, N])
    // -- matmul([M, N], [N, K]) -- [M, K]
    // result:gradOutput -- [M, K]
    auto result = fl::matmul(
        lhs.tensor(),
        rhs.tensor(),
        /* lhsProp = */
        MatrixProperty::None,
        /* rhsProp = */
        MatrixProperty::Transpose
    );
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        if(inputs[0].isCalcGrad()) {
            // matmul(gradOutput, inputs[1])
            // -- matmul([M, K], [K, N]) -- [M, N]
            auto val = fl::matmul(gradOutput.tensor(), inputs[1].tensor());
            inputs[0].addGrad(Variable(detail::sumAs(val, inputs[0].shape()), false));
        }
        if(inputs[1].isCalcGrad()) {
            // matmulTN(gradOutput, inputs[0])
            // -- matmulTN([M, K], [M, N])
            // -- matmul([K, M], [M, N]) -- [K, N]
            auto val = fl::matmul(
                gradOutput.tensor(),
                inputs[0].tensor(),
                /* lhsProp = */
                MatrixProperty::Transpose
            );
            inputs[1].addGrad(Variable(detail::sumAs(val, inputs[1].shape()), false));
        }
    };
    return Variable(result, {lhs, rhs}, gradFunc);
}

Variable abs(Variable const& input) {
    auto result = fl::abs(input.tensor());
    auto gradFunc = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        // Convert it into -1, 0, 1
        auto sign = fl::sign(inputs[0].tensor());
        inputs[0].addGrad(Variable((sign * gradOutput.tensor()), false));
    };
    return Variable(result, {input}, gradFunc);
}

Variable flat(Variable const& input) {
    auto result = input.tensor().flatten();
    Shape idims = input.shape();
    auto gradFunc =
        [idims](std::vector<Variable>& inputs, Variable const& gradOutput) {
        inputs[0].addGrad(Variable(reshape(gradOutput.tensor(), idims), false));
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable moddims(Variable const& input, Shape const& dims) {
    if(input.ndim() == 0)
        return input;
    Shape inferDims = dims;
    unsigned maxNDims =
        std::max(input.ndim(), static_cast<unsigned>(dims.ndim()));

    // Check for inferred dims that are beyond the input's number of dims
    for(int i = 0; i < maxNDims; ++i)
        if(i >= input.ndim() && inferDims[i] == 0)
            throw std::invalid_argument(
                "moddims: tried to infer dimension " + std::to_string(i)
                + " which exceeds the number of dimensions of the input."
            );

    // Infer any 0 dim
    for(int i = 0; i < maxNDims; ++i)
        if(i < inferDims.ndim() && inferDims[i] == 0)
            inferDims[i] = input.dim(i);

    // Infer any -1 dim
    int nInfer = 0;
    for(int i = 0; i < maxNDims; i++)
        if(i < inferDims.ndim() && inferDims[i] == -1) {
            nInfer++;
            inferDims[i] = -(input.elements() / inferDims.elements());
        }

    if(nInfer > 1)
        throw std::invalid_argument("moddims: too many dimensions infer");

    if(inferDims.elements() != input.elements())
        throw std::invalid_argument("moddims: mismatched # of elements");

    auto result = fl::reshape(input.tensor(), inferDims);

    Shape inDims = input.shape();
    auto gradFunc = [inDims](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        inputs[0].addGrad(Variable(moddims(gradOutput, inDims).tensor(), false));
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable softmax(Variable const& input, int const dim) {
    Tensor inputArr = FL_ADJUST_INPUT_TYPE(input.tensor());
    auto maxvals = amax(inputArr, {dim}, /* keepDims = */ true);
    Shape tiledims(std::vector<Dim>(input.ndim(), 1));
    tiledims[dim] = input.dim(dim);

    auto expInput = fl::exp(inputArr - fl::tile(maxvals, tiledims));
    auto result = expInput
        / fl::tile(fl::sum(expInput, {dim}, /* keepDims = */ true), tiledims);

    fl::eval(result);
    auto gradFunc = [dim, tiledims, result](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto rbyg = gradOutput.tensor() * result;
        auto gradSm = rbyg
            - result
            * fl::tile(fl::sum(rbyg, {dim}, /* keepDims = */ true), tiledims);
        inputs[0].addGrad(Variable(gradSm.asType(inputs[0].type()), false));
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable logSoftmax(Variable const& input, int const dim) {
    Tensor inputArr = FL_ADJUST_INPUT_TYPE(input.tensor());
    auto maxvals = amax(inputArr, {dim}, /* keepDims = */ true);
    // TODO{fl::Tensor}{rewrite}
    Shape tiledims(std::vector<Dim>(input.ndim(), 1));
    tiledims[dim] = input.dim(dim);
    auto result = inputArr
        - fl::tile(
            fl::log(
                fl::sum(
                    fl::exp(inputArr - fl::tile(maxvals, tiledims)),
                    {dim},
                    /* keepDims = */
                    true
                )
            )
            + maxvals,
            tiledims
        );

    fl::eval(result);
    auto gradFunc = [dim, tiledims, result](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto gradLsm = gradOutput.tensor()
            - fl::exp(result)
            * fl::tile(
                fl::sum(gradOutput.tensor(), {dim}, /* keepDims = */ true),
                tiledims
            );
        inputs[0].addGrad(Variable(gradLsm.asType(inputs[0].type()), false));
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable binaryCrossEntropy(Variable const& inputs, Variable const& targets) {
    auto targetsTyped = targets.asType(inputs.type());
    return negate(
        targetsTyped * log(inputs) + (1 - targetsTyped) * log(1 - inputs)
    );
}

Variable categoricalCrossEntropy(
    Variable const& in,
    Variable const& targets,
    ReduceMode reduction /* =ReduceMode::MEAN */,
    int ignoreIndex /* = -1 */
) {
    auto input = FL_ADJUST_INPUT_TYPE(in);
    // input -- [C, X1, X2, X3]
    // target -- [X1, X2, X3, 1]
    if(input.ndim() != targets.ndim() + 1)
        throw std::invalid_argument(
            "dimension mismatch in categorical cross entropy: "
            "target must have one fewer dimension than input"
        );
    for(int i = 1; i < input.ndim(); i++)
        if(input.dim(i) != targets.dim(i - 1))
            throw std::invalid_argument(
                "dimension mismatch in categorical cross entropy"
            );

    int C = input.dim(0);
    int X = targets.elements();
    if(
        fl::any_of(
            ((targets.tensor() < 0) || (targets.tensor() >= C))
            && (targets.tensor() != ignoreIndex)
        )
        .scalar<char>()
    )
        throw std::invalid_argument(
            "target contains elements out of valid range [0, num_categories) "
            "in categorical cross entropy"
        );

    auto x = fl::reshape(input.tensor(), Shape({C, X}));
    auto y = fl::reshape(targets.tensor(), Shape({1, X}));

    auto A = fl::arange(Shape({C, X}));
    auto B = fl::tile(y, Shape({C}));
    auto mask = -(A == B); // [C X]

    auto result = mask * x;
    auto ignoreMask = (y == ignoreIndex).flatten(); // [X, 1]
    result = fl::sum(result, {0}).flatten(); // [X, 1]
    result(ignoreMask) = 0.;

    Tensor denominator;
    if(reduction == ReduceMode::NONE) {
        result = fl::reshape(result, targets.shape()); // [X1 X2 X3]
    }
    else if(reduction == ReduceMode::MEAN) {
        denominator = fl::sum((!ignoreMask).asType(fl::dtype::s32), {0});
        result = fl::sum(result, {0}) / denominator; // [1]
    }
    else if(reduction == ReduceMode::SUM) {
        result = fl::sum(result, {0}); // [1]
    }
    else
        throw std::invalid_argument(
            "unknown reduction method for categorical cross entropy"
        );

    auto inputDims = input.shape();
    auto gradFunc = [C, X, mask, ignoreMask, denominator, reduction, inputDims](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        Tensor grad = gradOutput.tensor();
        if(reduction == ReduceMode::NONE)
            grad = fl::reshape(grad, {X});
        else if(reduction == ReduceMode::MEAN)
            grad = fl::tile(grad / denominator, {X});
        else if(reduction == ReduceMode::SUM)
            grad = fl::tile(grad, {X});
        // [1 X]
        grad(ignoreMask) = 0.;
        grad = fl::reshape(grad, {1, X});
        grad = fl::tile(grad, {C}) * mask;
        inputs[0].addGrad(Variable(fl::reshape(grad, inputDims), false));
    };

    return Variable(result, {input.withoutData(), targets}, gradFunc);
}

Variable weightedCategoricalCrossEntropy(
    Variable const& input,
    Variable const& targets,
    Variable const& weight,
    int ignoreIndex /* = -1 */
) {
    // input -- [C, X1, X2, X3]
    // target -- [X1, X2, X3]
    if(input.ndim() < targets.ndim() - 1)
        throw std::invalid_argument(
            "weightedCategoricalCrossEntropy: input must have one more than the "
            "number of target dimensions minus 1"
        );

    for(int i = 1; i < targets.ndim() - 2; i++)
        if(input.dim(i) != targets.dim(i - 1))
            throw std::invalid_argument(
                "weightedCategoricalCrossEntropy: dimension mismatch in categorical cross entropy"
            );
    if(weight.dim(0) != input.dim(0))
        throw std::invalid_argument(
            "weightedCategoricalCrossEntropy: dimension mismatch in categorical cross entropy"
        );

    int C = input.dim(0);
    int X = targets.elements();
    if(
        fl::any_of((targets.tensor() < 0) || (targets.tensor() >= C))
        .scalar<char>()
    )
        throw std::invalid_argument(
            "weightedCategoricalCrossEntropy: target contains elements out of valid range "
            "[0, num_categories) in categorical cross entropy"
        );

    auto x = fl::reshape(input.tensor(), {C, X});
    auto y = fl::reshape(targets.tensor(), {1, X});

    auto A = fl::arange({C, X});
    auto B = fl::tile(y, {C});
    auto mask = -(A == B); // [C X]

    auto weightSum = (-mask) * fl::tile(weight.tensor(), {1, X});
    Variable denominator = {fl::sum(weightSum, {0, 1}), false};

    auto result = mask * x;
    result = result * weight.tensor();

    auto ignoreMask = (y != ignoreIndex).asType(fl::dtype::s32); // [1, X]
    result = ignoreMask * fl::sum(result, {0}, /* keepDims = */ true); // [1, X]
    result = fl::sum(result, {1}, /* keepDims = */ true) / denominator.tensor();

    auto inputDims = input.shape();
    auto gradFunc = [C, X, mask, ignoreMask, denominator, inputDims](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto grad = gradOutput.tensor();
        grad = fl::tile(grad / denominator.tensor(), {1, X});

        auto weightTensor = inputs[2].tensor();
        grad *= ignoreMask;
        grad = fl::tile(grad, {C}) * mask;
        grad = fl::reshape(grad, inputDims);
        grad = grad * weightTensor;
        inputs[0].addGrad(Variable(fl::reshape(grad, inputDims), false));
    };

    return Variable(result, {input.withoutData(), targets, weight}, gradFunc);
}

Variable reorder(Variable const& input, Shape const& shape) {
    auto result = fl::transpose(input.tensor(), shape);
    if(!result.isContiguous())
        result = result.asContiguousTensor();

    std::vector<std::pair<Dim, int>> dimGrad(shape.ndim());
    for(unsigned i = 0; i < shape.ndim(); ++i)
        dimGrad[i] = {shape.dim(i), i};

    std::sort(dimGrad.begin(), dimGrad.end());

    auto gradFunc =
        [dimGrad](std::vector<Variable>& inputs, Variable const& gradOutput) {
        Shape reordered(std::vector<Dim>(dimGrad.size()));
        for(unsigned i = 0; i < dimGrad.size(); ++i)
            reordered[i] = dimGrad[i].second;

        inputs[0].addGrad(
            Variable(fl::transpose(gradOutput.tensor(), reordered), false)
        );
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable linear(Variable const& input, Variable const& weight) {
    auto dummyBias = Variable(Tensor().asType(input.type()), false);
    return linear(input, weight, dummyBias);
}

Variable linear(Variable const& in, Variable const& wt, Variable const& bs) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(in, wt, bs);
    auto input = FL_ADJUST_INPUT_TYPE(in);
    auto weight = FL_ADJUST_INPUT_TYPE(wt);
    auto bias = FL_ADJUST_INPUT_TYPE(bs);

    Shape to2d({input.dim(0), input.elements() / input.dim(0)});
    auto to4d = input.shape();
    to4d[0] = weight.tensor().dim(0);

    auto output =
        reshape(fl::matmul(weight.tensor(), reshape(input.tensor(), to2d)), to4d);

    auto hasBias = bias.elements() > 0;
    if(hasBias) {
        auto tiledims = output.shape();
        tiledims[0] = 1;
        output = output + tile(bias.tensor(), tiledims);
    }

    auto gradFunc = [hasBias](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto& in = inputs[0];
        auto& wt = inputs[1];
        Tensor wtTensor = wt.tensor();
        Tensor gradOutputTensor = gradOutput.tensor();

        auto nframes = in.elements() / in.dim(0);

        if(hasBias && inputs[2].isCalcGrad()) {
            auto& bs = inputs[2];
            auto biasGrad = sumAs(gradOutput, bs).tensor();
            bs.addGrad(Variable(biasGrad, false));
        }
        if(in.isCalcGrad()) {
            Shape to2dout({wtTensor.dim(0), nframes});
            auto inGrad =
                moddims(matmulTN(wt, moddims(gradOutput, to2dout)), in.shape())
                .tensor();
            in.addGrad(Variable(inGrad, false));
        }
        if(wt.isCalcGrad()) {
            Shape to2din({wtTensor.dim(1), nframes});
            Shape to2dout({wtTensor.dim(0), nframes});
            auto wtGrad =
                matmulNT(moddims(gradOutput, to2dout), moddims(in, to2din)).tensor();
            wt.addGrad(Variable(wtGrad, false));
        }
    };
    if(hasBias)
        return Variable(output, {input, weight, bias}, gradFunc);
    return Variable(output, {input, weight}, gradFunc);
}

Variable conv2d(
    Variable const& input,
    Variable const& weights,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks
) {
    auto dummyBias = Variable(Tensor(input.type()), false);
    return conv2d(
        input,
        weights,
        dummyBias,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        groups,
        benchmarks
    );
}

Variable conv2d(
    Variable const& in,
    Variable const& wt,
    Variable const& bs,
    int sx,
    int sy,
    int px,
    int py,
    int dx,
    int dy,
    int groups,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks
) {
    FL_VARIABLE_DTYPES_MATCH_CHECK(in, wt, bs);

    auto payload = detail::createAutogradPayload(in, wt, bs);

    bool hasBias = !bs.isEmpty();

    auto input = FL_ADJUST_INPUT_TYPE(in);
    auto weights = FL_ADJUST_INPUT_TYPE(wt);
    auto bias = FL_ADJUST_INPUT_TYPE(bs);

    Tensor output = detail::conv2d(
        input.tensor(),
        weights.tensor(),
        bias.tensor(),
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        groups,
        payload
    );

    auto gradFunc =
        [sx, sy, px, py, dx, dy, hasBias, groups, benchmarks, payload](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        // Create benchmarks if needed
        auto& autogradExtension =
            inputs[0].tensor().backend().getExtension<AutogradExtension>();

        std::shared_ptr<DynamicBenchmark> dataBench;
        std::shared_ptr<DynamicBenchmark> filterBench;
        std::shared_ptr<DynamicBenchmark> biasBench;
        if(benchmarks && DynamicBenchmark::getBenchmarkMode()) {
            if(!benchmarks->bwdFilterBenchmark) {
                benchmarks->bwdFilterBenchmark =
                    autogradExtension.createBenchmarkOptions();
                filterBench = benchmarks->bwdFilterBenchmark;
            }
            if(!benchmarks->bwdDataBenchmark) {
                benchmarks->bwdDataBenchmark =
                    autogradExtension.createBenchmarkOptions();
                dataBench = benchmarks->bwdDataBenchmark;
            }
            if(!benchmarks->bwdBiasBenchmark) {
                benchmarks->bwdBiasBenchmark =
                    autogradExtension.createBenchmarkOptions();
                biasBench = benchmarks->bwdBiasBenchmark;
            }
        }

        // Bias gradients
        Tensor bs;
        bool const computeBiasGrad =
            inputs.size() > 2 && inputs[2].isCalcGrad();
        if(hasBias && computeBiasGrad) {
            bs = inputs[2].tensor();
            // auto biasGrad =
            // bs.backend().getExtension<AutogradExtension>().conv2dBackwardBias(
            // gradOutput.tensor(), bs, biasBench, payload);

            // inputs[2].addGrad(Variable(biasGrad, false)); // bias
        }

        auto& in = inputs[0].tensor();
        auto& wt = inputs[1].tensor();

        // Data (input) gradients
        if(inputs[0].isCalcGrad()) {
            auto dataGrad =
                in.backend().getExtension<AutogradExtension>().conv2dBackwardData(
                    gradOutput.tensor(),
                    in,
                    wt,
                    sx,
                    sy,
                    px,
                    py,
                    dx,
                    dy,
                    groups,
                    dataBench,
                    payload
                );

            inputs[0].addGrad(Variable(dataGrad, false)); // input/data
        }

        // Filter (weight) and bias gradients
        if(inputs[1].isCalcGrad() || computeBiasGrad) {
            auto [filterGrad, biasGrad] = wt.backend()
                                            .getExtension<AutogradExtension>()
                                            .conv2dBackwardFilterBias(
                                                gradOutput.tensor(),
                                                in,
                                                wt,
                                                bs,
                                                sx,
                                                sy,
                                                px,
                                                py,
                                                dx,
                                                dy,
                                                groups,
                                                filterBench,
                                                biasBench,
                                                payload
                                            );
            if(inputs[1].isCalcGrad()) {
                inputs[1].addGrad(Variable(filterGrad, false)); // filter/weight
            }
            if(computeBiasGrad)
                inputs[2].addGrad(Variable(biasGrad, false));
        }
    };
    if(hasBias)
        return Variable(output, {input, weights, bias}, gradFunc);
    return Variable(output, {input, weights}, gradFunc);
}

Variable pool2d(
    Variable const& input,
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py,
    PoolingMode mode /* = PoolingMode::MAX */
) {
    auto payload = detail::createAutogradPayload(input);
    Tensor output =
        fl::detail::pool2d(input.tensor(), wx, wy, sx, sy, px, py, mode, payload);

    auto gradFunc = [wx, wy, sx, sy, px, py, mode, output, payload](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto& in = inputs[0];
        if(!in.isCalcGrad())
            return;

        in.addGrad(
            Variable(
                in.tensor().backend().getExtension<AutogradExtension>().pool2dBackward(
                    gradOutput.tensor(),
                    in.tensor(),
                    output,
                    wx,
                    wy,
                    sx,
                    sy,
                    px,
                    py,
                    mode,
                    payload
                ),
                false
            )
        );
    };
    return Variable(output, {input}, gradFunc);
}

Variable batchnorm(
    Variable const& _input,
    Variable const& weight,
    Variable const& bias,
    Variable& runningMean,
    Variable& runningVar,
    std::vector<int> const& axes,
    bool train,
    double momentum,
    double epsilon
) {
    auto payload = detail::createAutogradPayload(_input, weight, bias);
    auto input = FL_ADJUST_INPUT_TYPE(_input);

    Tensor saveMean, saveVar;
    Tensor output = fl::detail::batchnorm(
        saveMean,
        saveVar,
        input.tensor(),
        weight.tensor(),
        bias.tensor(),
        runningMean.tensor(),
        runningVar.tensor(),
        axes,
        train,
        momentum,
        epsilon,
        payload
    );

    auto gradFunc =
        [saveMean = std::move(saveMean),
            saveVar = std::move(saveVar),
            train,
            axes,
            epsilon,
            payload](std::vector<Variable>& inputs, Variable const& _gradOutput) {
        auto& in = inputs[0];
        auto& wt = inputs[1];
        auto& bs = inputs[2];

        auto gradOutput = detail::adjustInputType(_gradOutput, "batchnorm");

        if(!in.isCalcGrad() && !wt.isCalcGrad() && !bs.isCalcGrad())
            return;

        auto [gradIn, gradWt, gradBs] =
            in.tensor()
              .backend()
              .getExtension<AutogradExtension>()
              .batchnormBackward(
                  gradOutput.tensor(),
                  saveMean,
                  saveVar,
                  detail::adjustInputType(in.tensor(), "batchnorm"),
                  wt.tensor(),
                  axes,
                  train,
                  epsilon,
                  payload
              );

        in.addGrad(Variable(gradIn.asType(in.type()), false));
        wt.addGrad(Variable(gradWt.asType(wt.type()), false));
        if(!bs.isEmpty())
            bs.addGrad(Variable(gradBs.asType(bs.type()), false));
    };
    return Variable(output, {input, weight, bias}, gradFunc);
}

Variable gatedlinearunit(Variable const& input, int const dim) {
    if(dim >= input.ndim())
        throw std::invalid_argument(
            "gatedlinearunit - passed dim is great than the "
            "number of dimensions of the input."
        );

    auto inDims = input.shape();
    auto inType = input.type();
    auto inSize = inDims[dim];
    if(inSize % 2 == 1)
        throw std::invalid_argument("halving dimension must be even for GLU");

    std::vector<fl::Index> fhalf(input.ndim(), fl::span);
    std::vector<fl::Index> shalf(input.ndim(), fl::span);
    fhalf[dim] = fl::range(inSize / 2);
    shalf[dim] = fl::range(inSize / 2, inSize);

    Tensor fhalfout = input.tensor()(fhalf);
    Tensor shalfout = input.tensor()(shalf);

    // Temporary workaround for indexing bug present in ArrayFire 3.6.1.
    fhalfout = fl::reshape(fhalfout, fhalfout.shape());
    shalfout = fl::reshape(shalfout, shalfout.shape());
    shalfout = fl::sigmoid(shalfout);

    auto gradFunc = [fhalf, shalf, fhalfout, shalfout, inDims, inType](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto gradGlu = Tensor(inDims, inType);
        gradGlu(fhalf) = shalfout * gradOutput.tensor();
        gradGlu(shalf) =
            shalfout * (1.0 - shalfout) * fhalfout * gradOutput.tensor();
        inputs[0].addGrad(Variable(gradGlu, false));
    };
    return Variable(fhalfout * shalfout, {input.withoutData()}, gradFunc);
}

std::tuple<Variable, Variable, Variable> rnn(
    Variable const& input,
    Variable const& hiddenState,
    Variable const& cellState,
    Variable const& weights,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    bool bidirectional,
    float dropProb
) {
    auto payload =
        detail::createAutogradPayload(input, hiddenState, cellState, weights);

    Tensor output, hiddenOut, cellStateOut;
    std::tie(output, hiddenOut, cellStateOut) = detail::rnn(
        input.tensor(),
        hiddenState.tensor(),
        cellState.tensor(),
        weights.tensor(),
        hiddenSize,
        numLayers,
        mode,
        bidirectional,
        dropProb,
        payload
    );

    auto gradData = std::make_shared<detail::RNNGradData>();

    auto gradFunc = [output,
            numLayers,
            hiddenSize,
            mode,
            bidirectional,
            dropProb,
            gradData,
            payload](
        std::vector<Variable>& inputs,
        Variable const& /* gradOutput */
    
    ) {
        auto& input = inputs[0];
        auto& hiddenState = inputs[1];
        auto& cellState = inputs[2];
        auto& weights = inputs[3];

        if(
            !(input.isCalcGrad() || hiddenState.isCalcGrad()
                || cellState.isCalcGrad() || weights.isCalcGrad())
        )
            return;

        auto [dy, dhy, dcy, dweights] =
            input.tensor().backend().getExtension<AutogradExtension>().rnnBackward(
                input.tensor(),
                hiddenState.tensor(),
                cellState.tensor(),
                weights.tensor(),
                gradData,
                output,
                numLayers,
                hiddenSize,
                mode,
                bidirectional,
                dropProb,
                payload
            );

        input.addGrad(Variable(dy.asType(input.type()), false));
        hiddenState.addGrad(Variable(dhy.asType(hiddenState.type()), false));
        cellState.addGrad(Variable(dcy.asType(cellState.type()), false));
        weights.addGrad(Variable(dweights.asType(weights.type()), false));
    };

    Variable dummy(Tensor(), {input, hiddenState, cellState, weights}, gradFunc);

    auto dyGradFunc =
        [gradData](std::vector<Variable>& inputs, Variable const& gradOutput) {
        if(!inputs[0].isGradAvailable())
            inputs[0].addGrad(Variable(Tensor(), false));
        gradData->dy = gradOutput.tensor().asContiguousTensor();
    };

    auto dhyGradFunc =
        [gradData](std::vector<Variable>& inputs, Variable const& gradOutput) {
        if(!inputs[0].isGradAvailable())
            inputs[0].addGrad(Variable(Tensor(), false));
        gradData->dhy = gradOutput.tensor().asContiguousTensor();
    };

    auto dcyGradFunc =
        [gradData](std::vector<Variable>& inputs, Variable const& gradOutput) {
        if(!inputs[0].isGradAvailable())
            inputs[0].addGrad(Variable(Tensor(), false));
        gradData->dcy = gradOutput.tensor().asContiguousTensor();
    };

    Variable yv(output, {dummy}, dyGradFunc); // output
    Variable hyv(hiddenOut, {dummy}, dhyGradFunc); // hidden state output
    Variable cyv(cellStateOut, {dummy}, dcyGradFunc); // cell state output
    return std::make_tuple(yv, hyv, cyv);
}

Variable embedding(Variable const& input, Variable const& embeddings) {
    // TODO{fl::Tensor}{4-dims} - relax this
    if(input.ndim() >= 4)
        throw std::invalid_argument{"embedding input must have 3 or fewer dims"};

    auto const idxs = input.tensor().flatten();
    auto inDims = input.shape();
    std::vector<Dim> rDims(input.ndim() + 1);
    rDims[0] = embeddings.dim(0);
    for(Dim i = 1; i < input.ndim() + 1; i++)
        rDims[i] = inDims[i - 1];

    Shape const resultDims{rDims};
    auto const result = fl::reshape(embeddings.tensor()(fl::span, idxs), resultDims);

    auto grad_func = [](
        std::vector<Variable>& inputs,
        Variable const& gradOutput
    ) {
        auto& w = inputs[1];
        if(!w.isCalcGrad())
            return;

        auto const ip = inputs[0].tensor().flatten();
        auto size = static_cast<Dim>(ip.elements());
        auto const deltas = fl::reshape(gradOutput.tensor(), {w.dim(0), size});

        // Sparse Tensor
        auto const sp = Tensor{
            static_cast<Dim>(ip.elements()),
            w.dim(1),
            fl::full({size}, 1, deltas.type()),
            fl::arange({size + 1}, 0, fl::dtype::s32),
            ip.asType(fl::dtype::s32),
            fl::StorageType::CSR
        };

        auto const grad = transpose(
            fl::matmul(
                sp,
                transpose(deltas),
                /* lhsProp = */
                MatrixProperty::Transpose
            )
        );
        w.addGrad(Variable{grad, false});
    };

    return Variable{result, {input, embeddings}, grad_func};
}

Variable padding(
    Variable const& input,
    std::vector<std::pair<int, int>> pad,
    double val
) {
    if(pad.size() > input.ndim())
        throw std::invalid_argument{
            "padding: number of padding dimensions exceeds number "
            "of input dimensions"
        };

    Shape opDims = input.shape();
    std::vector<fl::Index> inSeq(input.ndim(), fl::span);
    for(int i = 0; i < pad.size(); ++i) {
        opDims[i] += (pad[i].first + pad[i].second);
        inSeq[i] = fl::range(pad[i].first, opDims[i] - pad[i].second);
    }
    Tensor result = fl::full(opDims, val, input.type());
    result(inSeq) = input.tensor();

    auto gradFunc =
        [inSeq](std::vector<Variable>& inputs, Variable const& gradOutput) {
        inputs[0].addGrad(Variable(gradOutput.tensor()(inSeq), false));
    };
    return Variable(result, {input.withoutData()}, gradFunc);
}

Variable dropout(Variable const& input, double p) {
    if(p > 0.0) {
        auto mask = Variable(
            (fl::rand(input.shape(), input.type()) > p).asType(input.type()),
            false
        );
        return 1.0 / (1.0 - p) * mask * input;
    }
    else
        return input;
}

Variable relu(Variable const& input) { return max(input, 0.0); }

Variable gelu(Variable const& in) {
    auto input = FL_ADJUST_INPUT_TYPE(in);
    return 0.5 * input
        * (1.0
            + fl::tanh(0.7978845608 * (input + 0.044715 * input * input * input)));
}

fl::Variable relativePositionEmbeddingRotate(fl::Variable const& input) {
    if(input.ndim() != 3)
        throw std::invalid_argument(
            "relativePositionEmbeddingRotate - "
            "input tensor must have 3 dimensions"
        );

    auto data = input.tensor();
    int d0 = data.dim(0);
    int d1 = data.dim(1);
    int d2 = data.dim(2);
    data = fl::concatenate(
        /* axis = */ 0,
        data,
        fl::full({d1, d1, d2}, 0.0, data.type())
    );
    data = fl::reshape(data, {(d0 + d1) * d1, 1, d2});
    data = data(fl::range(0, (d1 + d0 - 1) * d1));
    data = fl::reshape(data, {d0 + d1 - 1, d1, d2});
    auto gradFunc = [d0, d1, d2](
        std::vector<fl::Variable>& inputs,
        fl::Variable const& gradOutput
    ) {
        auto gradData = gradOutput.tensor();
        gradData = fl::reshape(gradData, {(d0 + d1 - 1) * d1, 1, d2});
        gradData = fl::concatenate(
            0,
            gradData,
            fl::full({d1, 1, d2}, 0.0, gradData.type())
        );
        gradData = reshape(gradData, {d0 + d1, d1, d2});
        gradData = Variable(gradData, false)(fl::range(0, d0)).tensor();
        inputs[0].addGrad(fl::Variable(gradData, false));
    };
    return fl::Variable(data, {input}, gradFunc);
}

fl::Variable multiheadAttention(
    fl::Variable const& query,
    fl::Variable const& key,
    fl::Variable const& value,
    fl::Variable const& posEmb,
    fl::Variable const& mask,
    fl::Variable const& padMask,
    int32_t const nHeads,
    double const pDropout,
    int32_t const offset /* = 0 */
) {
    if(query.ndim() != 3)
        throw std::invalid_argument(
            "multiheadAttention - query input tensor should be 3 dimensions: "
            "Time x (nHeads * headDim) x B"
        );
    if(key.ndim() != 3)
        throw std::invalid_argument(
            "multiheadAttention - key input tensor should be 3 dimensions: "
            "Time x (nHeads * headDim) x B"
        );
    if(value.ndim() != 3)
        throw std::invalid_argument(
            "multiheadAttention - value input tensor should be 3 dimensions: "
            "Time x (nHeads * headDim) x B"
        );

    int32_t bsz = query.dim(2);
    int32_t modelDim = query.dim(1);
    int32_t headDim = modelDim / nHeads;

    auto q = moddims(query, {-1, headDim, nHeads * bsz});
    auto k = moddims(key, {-1, headDim, nHeads * bsz});
    auto v = moddims(value, {-1, headDim, nHeads * bsz});

    q = q / std::sqrt(float(headDim));
    auto scores = matmulNT(q, k);
    if(!posEmb.isEmpty()) {
        int n = posEmb.dim(0) / 2 - offset;
        auto pscores =
            relativePositionEmbeddingRotate(matmulNT(posEmb.asType(q.type()), q));
        scores =
            scores + transpose(pscores(fl::range(n, n + k.dim(0))), {1, 0, 2});
    }
    if(!mask.isEmpty())
        scores = scores + tileAs(mask.asType(scores.type()), scores);
    if(!padMask.isEmpty()) {
        if(padMask.dim(0) != query.dim(0))
            throw std::invalid_argument(
                "multiheadAttention: invalid padding mask size"
            );
        auto padMaskTile = moddims(padMask, {1, padMask.dim(0), 1, bsz});
        padMaskTile =
            tileAs(padMaskTile, {padMask.dim(0), padMask.dim(0), nHeads, bsz});
        scores = scores
            + moddims(
                padMaskTile.asType(scores.type()),
                {padMask.dim(0), padMask.dim(0), nHeads * bsz}
            );
    }

    auto attn = dropout(softmax(scores, 1), pDropout);
    auto result = matmul(attn.asType(v.type()), v);
    result = moddims(result, {-1, headDim * nHeads, bsz});
    return result;
}

} // namespace fl

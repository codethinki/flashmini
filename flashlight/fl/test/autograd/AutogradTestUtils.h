/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gtest/gtest.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

namespace fl {
namespace detail {

    class AutogradTestF16 : public ::testing::Test {
        void SetUp() override {
            // Ensures all operations will be in f16
            OptimMode::get().setOptimLevel(OptimLevel::O3);
        }

        void TearDown() override { OptimMode::get().setOptimLevel(OptimLevel::DEFAULT); }
    };

    using JacobianFunc = std::function<Variable (Variable&)>;
    inline bool jacobianTestImpl(
        JacobianFunc const& func,
        Variable& input,
        double precision = 1E-5,
        float perturbation = 1E-4,
        std::vector<Variable*> const& zeroGradientVariables = {}
    ) {
        auto const outBase = func(input);
        auto const outElements = outBase.elements();
        auto const inElements = input.elements();

        auto const fwdJacobian = Tensor({outElements, inElements}, input.type());

        for(int i = 0; i < inElements; ++i) {
            auto orig = input.tensor().flatten()(i);
            input.tensor().flat(i) = orig - perturbation;
            auto outA = func(input).tensor();

            input.tensor().flat(i) = orig + perturbation;
            auto outB = func(input).tensor();

            input.tensor().flat(i) = orig;


            fwdJacobian(fl::span, i) = fl::reshape((outB - outA), {static_cast<Dim>(outA.elements())}) * 0.5 /
                perturbation;
        }

        auto const bwdJacobian = Tensor({outElements, inElements}, input.type());
        auto const outD = Variable(fl::full(outBase.shape(), 0, outBase.type()), false);

        for(int i = 0; i < outD.elements(); ++i) {
            outD.tensor().flat(i) = 1; // element in 1D view
            input.zeroGrad();
            for(auto* var : zeroGradientVariables)
                var->zeroGrad();
            auto out = func(input);
            out.backward(outD);

            bwdJacobian(i) = fl::reshape(input.grad().tensor(), {inElements});
            outD.tensor().flat(i) = 0;
        }

        return allClose(fwdJacobian, bwdJacobian, precision);
    }

}
} // namespace fl

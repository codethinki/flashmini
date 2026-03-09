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


        if(fwdJacobian.type() == dtype::f64 && fwdJacobian.elements() == 1280) {
            auto const flat_tensor_print = [](Tensor t, std::string_view name) {
                std::span host{t.host<double>(), t.elements()};


                std::cout << std::format("{}:\n[", name);
                for(size_t i = 0; i < host.size();) {
                    size_t c = 1;
                    while(i + c < host.size() && host[i] == host[i + c])
                        c++;

                    if(c == 1)
                        std::cout << host[i];
                    else
                        std::cout << std::format("({})_{}", host[i], c);
                    if(i + c < host.size() - 1)
                        std::cout << ", ";

                    i += c;
                }

                std::cout << "]\n\n\n";
            };


            flat_tensor_print(fwdJacobian, "fwd");
            flat_tensor_print(bwdJacobian, "bwd");
        }

        return allClose(fwdJacobian, bwdJacobian, precision);
    }

}
} // namespace fl

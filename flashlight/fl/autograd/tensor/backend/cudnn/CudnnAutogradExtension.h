/*
 * SPDX-License-Identifier: MIT
 *
 * Original code: Copyright (c) Meta Platforms, Inc. (see FLASHLIGHT_LICENSE)
 * Modifications: Copyright (c) 2026 Lukas Thomann (see LICENSE)
 */

#pragma once

#include <unordered_map>

#include "flashlight/fl/autograd/tensor/AutogradExtension.h"

namespace fl {

class DynamicBenchmark;

class CudnnAutogradExtension : public AutogradExtension {
    // TODO(jacobkahn): implement getCudnnHandle

public:
    bool isDataTypeSupported(fl::dtype const& dtype) const override;

    enum class KernelMode { F32 = 0, F32_ALLOW_CONVERSION = 1, F16 = 2 };

    std::shared_ptr<fl::DynamicBenchmark> createBenchmarkOptions() override;

    /**************************** Forward ****************************/
    Tensor conv2d(
        Tensor const& input,
        Tensor const& weights,
        Tensor const& bias,
        int sx,
        int sy,
        int px,
        int py,
        int dx,
        int dy,
        int groups,
        std::shared_ptr<detail::AutogradPayload> payload
    ) override;

    Tensor pool2d(
        Tensor const& input,
        int wx,
        int wy,
        int sx,
        int sy,
        int px,
        int py,
        PoolingMode mode,
        std::shared_ptr<detail::AutogradPayload> payload
    ) override;

    Tensor batchnorm(
        Tensor& saveMean,
        Tensor& saveVar,
        Tensor const& input,
        Tensor const& weight,
        Tensor const& bias,
        Tensor& runningMean,
        Tensor& runningVar,
        std::vector<int> const& axes,
        bool train,
        double momentum,
        double epsilon,
        std::shared_ptr<detail::AutogradPayload> payload
    ) override;

    std::tuple<Tensor, Tensor, Tensor> rnn(
        Tensor const& input,
        Tensor const& hiddenState,
        Tensor const& cellState,
        Tensor const& weights,
        int hiddenSize,
        int numLayers,
        RnnMode mode,
        bool bidirectional,
        float dropProb,
        std::shared_ptr<detail::AutogradPayload> autogradPayload
    ) override;

    /**************************** Backward ****************************/
    // ]----- Convolution
    Tensor conv2dBackwardData(
        Tensor const& gradOutput,
        Tensor const& input,
        Tensor const& weight,
        int sx,
        int sy,
        int px,
        int py,
        int dx,
        int dy,
        int groups,
        std::shared_ptr<DynamicBenchmark> dataGradBenchmark,
        std::shared_ptr<detail::AutogradPayload> payload
    ) override;

    std::pair<Tensor, Tensor> conv2dBackwardFilterBias(
        Tensor const& gradOutput,
        Tensor const& input,
        Tensor const& weights,
        Tensor const& bias,
        int sx,
        int sy,
        int px,
        int py,
        int dx,
        int dy,
        int groups,
        std::shared_ptr<DynamicBenchmark> filterBench,
        std::shared_ptr<DynamicBenchmark> biasBench,
        std::shared_ptr<detail::AutogradPayload> autogradPayload
    ) override;

    // ]----- pool2D
    Tensor pool2dBackward(
        Tensor const& gradOutput,
        Tensor const& input,
        Tensor const& poolOutput,
        int wx,
        int wy,
        int sx,
        int sy,
        int px,
        int py,
        PoolingMode mode,
        std::shared_ptr<detail::AutogradPayload> payload
    ) override;

    // ]----- batchnorm
    std::tuple<Tensor, Tensor, Tensor> batchnormBackward(
        Tensor const& gradOutput,
        Tensor const& saveMean,
        Tensor const& saveVar,
        Tensor const& input,
        Tensor const& weight,
        std::vector<int> const& axes,
        bool train,
        float epsilon,
        std::shared_ptr<detail::AutogradPayload> payload
    ) override;

    // ]----- rnn
    std::tuple<Tensor, Tensor, Tensor, Tensor> rnnBackward(
        Tensor const& input,
        Tensor const& hiddenState,
        Tensor const& cellState,
        Tensor const& weights,
        std::shared_ptr<detail::RNNGradData> gradData,
        Tensor const& output,
        int numLayers,
        int hiddenSize,
        RnnMode mode,
        bool bidirectional,
        float dropProb,
        std::shared_ptr<detail::AutogradPayload> autogradPayload
    ) override;

private:

    static void checkHiddenStateDims(int hiddenSize, Tensor const& hiddenState, int batchSize, int totalLayers);
    static void checkCellStateDims(
        int hiddenSize,
        RnnMode mode,
        Tensor const& cellState,
        int batchSize,
        int totalLayers
    );

};

} // namespace fl

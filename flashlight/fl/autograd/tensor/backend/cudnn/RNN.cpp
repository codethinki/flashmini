/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnAutogradExtension.h"

#include <cudnn.h>

#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnRnnUtils.h"
#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnUtils.h"


namespace fl {
namespace {
    struct CudnnRnnAutogradPayload : public detail::AutogradPayloadData {
        Tensor reserveSpace;
    };
} // namespace

void CudnnAutogradExtension::checkHiddenStateDims(
    int const hiddenSize,
    Tensor const& hiddenState,
    int batchSize,
    int totalLayers
) {
    auto const& hxDims = hiddenState.shape();
    int const hxHiddenSize = static_cast<int>(hxDims[0]);
    int const hxBatchSize = hiddenState.ndim() < 2 ? 1 : static_cast<int>(hxDims[1]);
    int const hxTotalLayers = hiddenState.ndim() < 3 ? 1 : static_cast<int>(hxDims[2]);

    if(
        hxHiddenSize != hiddenSize || hxBatchSize != batchSize
        || hxTotalLayers != totalLayers
    )
        throw std::invalid_argument("invalid hidden state dims for RNN");
}
void CudnnAutogradExtension::checkCellStateDims(
    int const hiddenSize,
    RnnMode const mode,
    Tensor const& cellState,
    int batchSize,
    int totalLayers
) {
    if(mode != RnnMode::LSTM || cellState.dim(0) != hiddenSize
        || cellState.dim(1) != batchSize || cellState.dim(2) != totalLayers)
        throw std::invalid_argument("invalid cell state dims for RNN");
}

std::tuple<Tensor, Tensor, Tensor> CudnnAutogradExtension::rnn(
    Tensor const& input,
    Tensor const& hiddenState,
    Tensor const& cellState,
    Tensor const& weights,
    int const hiddenSize,
    int const numLayers,
    RnnMode const mode,
    bool const bidirectional,
    float const dropProb,
    std::shared_ptr<detail::AutogradPayload> autogradPayload
) {
    FL_TENSOR_DTYPES_MATCH_CHECK(input, hiddenState, cellState, weights);

    bool const train = (autogradPayload != nullptr);
    auto const payload = std::make_shared<CudnnRnnAutogradPayload>();
    if(train)
        autogradPayload->data = payload;

    auto const x = input.asContiguousTensor();

    auto const cHiddenState = hiddenState.asContiguousTensor();
    auto const cCellState = cellState.asContiguousTensor();

    DropoutDescriptor dropout{dropProb};

    auto const& dims = max(x.shape(), {1, 1, 1});


    auto const inputSize = static_cast<int>(dims[0]);
    auto batchSize = static_cast<int>(dims[1]);
    auto seqLength = static_cast<int>(dims[2]);


    RNNDescriptor const rnnDesc{
        input.type(),
        inputSize,
        hiddenSize,
        numLayers,
        mode,
        bidirectional,
        dropout
    };


    int totalLayers = numLayers * (bidirectional ? 2 : 1);
    int outSize = hiddenSize * (bidirectional ? 2 : 1);

    if(!cHiddenState.isEmpty())
        checkHiddenStateDims(hiddenSize, cHiddenState, batchSize, totalLayers);

    if(!cCellState.isEmpty())
        checkCellStateDims(hiddenSize, mode, cCellState, batchSize, totalLayers);


    Shape const hDims = {1, hiddenSize, batchSize, totalLayers};
    TensorDescriptor const hxDesc{x.type(), hDims};
    TensorDescriptor const cxDesc{x.type(), hDims};

    Tensor y{{outSize, batchSize, seqLength}, input.type()};

    Tensor hy{{hiddenSize, batchSize, totalLayers}, x.type()};

    Tensor cy{};
    if(mode == RnnMode::LSTM)
        cy = Tensor{hy.shape(), x.type()};

    cudnn_rnn_forward(
        batchSize,
        seqLength,
        train,
        rnnDesc,
        x,
        y,
        weights,
        cxDesc,
        hxDesc,
        hy,
        cy,
        cHiddenState,
        cCellState,
        payload->reserveSpace // output
    );

    return std::make_tuple(y, hy, cy);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> CudnnAutogradExtension::rnnBackward(
    Tensor const& input,
    Tensor const& hiddenState,
    Tensor const& cellState,
    Tensor const& weights,
    std::shared_ptr<detail::RNNGradData> const gradData,
    Tensor const& output,
    int const numLayers,
    int const hiddenSize,
    RnnMode const mode,
    bool const bidirectional,
    float const dropProb,
    std::shared_ptr<detail::AutogradPayload> autogradPayload
) {
    if(!autogradPayload)
        throw std::invalid_argument(
            "CudnnAutogradExtension::rnnBackward given null detail::AutogradPayload"
        );
    auto const payload = std::static_pointer_cast<CudnnRnnAutogradPayload>(autogradPayload->data);

    auto const x = input.asContiguousTensor();
    auto& y = output;

    auto const& dims = x.shape();
    int const inputSize = dims[0];
    int const batchSize = dims.ndim() < 2 ? 1 : dims[1];
    int const seqLength = dims.ndim() < 3 ? 1 : dims[2];
    int const totalLayers = numLayers * (bidirectional ? 2 : 1);

    DropoutDescriptor dropout{dropProb};
    RNNDescriptor const rnnDesc{input.type(), inputSize, hiddenSize, numLayers, mode, bidirectional, dropout};

    Shape const hDims = {1, hiddenSize, batchSize, totalLayers};
    TensorDescriptor const hxDesc{x.type(), hDims};
    TensorDescriptor const cxDesc{x.type(), hDims};

    Tensor dhx{hiddenState.shape(), hiddenState.type()};
    Tensor dcx{cellState.shape(), cellState.type()};

    Tensor dx{input.shape(), input.type()};
    Tensor dw = fl::full(weights.shape(), 0, weights.type());

    auto& dy = gradData->dy;
    if(dy.isEmpty())
        dy = fl::full(y.shape(), 0.0, y.type());
    auto const& dhy = gradData->dhy;
    auto const& dcy = gradData->dcy;

    cudnn_rnn_backward(
        batchSize,
        seqLength,
        rnnDesc,
        x,
        y,
        dy,
        weights,
        cxDesc,
        hxDesc,
        dhy,
        dcy,
        hiddenState,
        cellState,
        dx,
        dhx,
        dcx,
        dw,
        payload->reserveSpace
    );

    return std::make_tuple(dx, dhx, dcx, dw);
}



} // namespace fl

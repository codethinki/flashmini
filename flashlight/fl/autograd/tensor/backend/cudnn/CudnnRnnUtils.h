#pragma once
#include "CudnnUtils.h"

namespace fl {
void cudnn_rnn_forward(
    int batchSize,
    int seqLength,
    bool train,
    RNNDescriptor const& rnnDesc,

    Tensor const& x,
    Tensor const& y,
    Tensor const& weights,
    TensorDescriptor const& cxDesc,
    TensorDescriptor const& hxDesc,
    Tensor const& hy,
    Tensor const& cy,
    Tensor const& hiddenState,
    Tensor const& cellState,

    Tensor& reserveSpace // out
);
void cudnn_rnn_backward(
    int batchSize,
    int seqLength,
    RNNDescriptor const& rnnDesc,

    Tensor const& x,
    Tensor const& y,
    Tensor const& dy,
    Tensor const& weights,
    TensorDescriptor const& cxDesc,
    TensorDescriptor const& hxDesc,
    Tensor const& dhy,
    Tensor const& dcy,
    Tensor const& hiddenState,
    Tensor const& cellState,
    Tensor const& dx,
    Tensor const& dhx,
    Tensor const& dcx,
    Tensor const& dw,

    Tensor const& reserveSpace
);
}

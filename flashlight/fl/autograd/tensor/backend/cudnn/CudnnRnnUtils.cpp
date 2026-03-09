#include "CudnnRnnUtils.h"

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Compute.h"


namespace fl {
namespace {
    struct temp_space_sizes {
        size_t size;
        size_t reserveSize;
    };

    temp_space_sizes rnn_temp_space_sizes(
        cudnnHandle_t handle,
        RNNDescriptor const& rnnDescriptor,
        RNNDataDescriptor const& xDescriptor,
        cudnnForwardMode_t mode
    ) {
        temp_space_sizes sizes{};

        CUDNN_CHECK_ERR(
            cudnnGetRNNTempSpaceSizes(
                handle,
                rnnDescriptor.get(),
                mode,
                xDescriptor.get(),
                &sizes.size,
                &sizes.reserveSize
            )
        );

        return sizes;
    }

    size_t rnn_weight_space_size(
        cudnnHandle_t handle,
        RNNDescriptor const& rnnDescriptor
    ) {
        size_t size = 0;

        CUDNN_CHECK_ERR(
            cudnnGetRNNWeightSpaceSize(handle,rnnDescriptor.get(),&size)
        );
        return size;
    }

    std::optional<Tensor> create_dev_seq_lengths(int batchSize, int seqLength) {
        //see cudnn docs for cudnnRNNForward as explanation
#if CUDNN_VERSION >= 8901
        return std::nullopt;
#else
        return fl::full({batchSize}, seqLength, fl::dtype::s32);
#endif
    }

}
}

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
    Tensor& reserveSpace
) {
    RNNDataDescriptor xDesc{x.type(), x.shape()};
    RNNDataDescriptor yDesc{y.type(), y.shape()};

    auto handle = getCudnnHandle();

    size_t weightSpaceSize = rnn_weight_space_size(handle, rnnDesc);

    if(weightSpaceSize != weights.bytes())
        throw std::invalid_argument("invalid # of parameters or wrong input shape for RNN");

    auto const forwardMode = train ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;

    auto [workspaceSize, reserveSize] = rnn_temp_space_sizes(handle, rnnDesc, xDesc, forwardMode);

    Tensor workspace({static_cast<int64_t>(workspaceSize)}, fl::dtype::b8);
    // Space must be reused between forward and backward for cuDNN

    reserveSpace = Tensor{{static_cast<int64_t>(reserveSize)}, fl::dtype::b8};

    auto devSeqLengths = create_dev_seq_lengths(batchSize, seqLength);

    auto const& cudnnStream = getCudnnStream();

    {
        auto contiguousX = x.asContiguousTensor();
        auto contiguousWeights = weights.asContiguousTensor();
        DevicePtr xRaw(contiguousX);
        DevicePtr hxRaw(hiddenState);
        DevicePtr cxRaw(cellState);
        DevicePtr weightSpaceRaw(contiguousWeights);
        DevicePtr yRaw(y);
        DevicePtr hyRaw(hy);
        DevicePtr cyRaw(cy);
        DevicePtr workspaceRaw(workspace);
        DevicePtr reserveSpaceRaw(reserveSpace);

        std::optional<DevicePtr> devSeqLengthsRaw{};

        if(devSeqLengths)
            devSeqLengthsRaw.emplace(*devSeqLengths);

        // ensure cudnn compute stream waits greaterThanEqual(&on input/output tensor streams

        std::vector waits{
            contiguousX,
            hiddenState,
            cellState,
            contiguousWeights,
            y,
            hy,
            cy,
            workspace,
            reserveSpace,
        };
        if(devSeqLengths)
            waits.push_back(*devSeqLengths);

        relativeSync(cudnnStream, waits);


        CUDNN_CHECK_ERR(
            cudnnRNNForward(
                handle,
                rnnDesc.get(),
                forwardMode,
                devSeqLengthsRaw ? devSeqLengthsRaw->getAs<int32_t const>() : nullptr,

                xDesc.get(),
                xRaw.get(),
                yDesc.get(),
                yRaw.get(),

                hxDesc.get(),
                hxRaw.get(),
                hyRaw.get(),
                cxDesc.get(),
                cxRaw.get(),
                cyRaw.get(),

                weightSpaceSize,
                weightSpaceRaw.get(),

                workspaceSize,
                workspaceRaw.get(),

                reserveSize,
                reserveSpaceRaw.get()
            )
        );
    }

    // ensure output tensor streams wait on cudnn compute stream
    relativeSync({y, hy, cy}, cudnnStream);
}

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
) {
    auto handle = getCudnnHandle();
    auto const& cudnnStream = getCudnnStream();

    RNNDataDescriptor xDesc{x.type(), x.shape()};
    RNNDataDescriptor yDesc{y.type(), y.shape()};

    size_t weightSpaceSize = rnn_weight_space_size(handle, rnnDesc);
    auto [workspaceSize, reserveSize] = rnn_temp_space_sizes(handle, rnnDesc, xDesc, CUDNN_FWD_MODE_TRAINING);

    Tensor workspace({static_cast<int64_t>(workspaceSize)}, fl::dtype::b8);

    auto devSeqLengths = create_dev_seq_lengths(batchSize, seqLength);

    std::vector<Tensor> waits = {y, workspace, reserveSpace};
    if(devSeqLengths)
        waits.push_back(*devSeqLengths);

    // ensure cudnn compute stream waits on input/output tensor streams
    relativeSync(cudnnStream, waits);

    DevicePtr yRaw(y);
    DevicePtr workspaceRaw(workspace);
    DevicePtr reserveSpaceRaw(reserveSpace);

    std::optional<DevicePtr> devSeqLengthsRaw{};
    if(devSeqLengths)
        devSeqLengthsRaw.emplace(*devSeqLengths);

    {
        DevicePtr dyRaw(dy); // Has to be set to 0 if empty
        DevicePtr dhyRaw(dhy);
        DevicePtr dcyRaw(dcy);

        DevicePtr wRaw(weights);

        DevicePtr hxRaw(hiddenState);
        DevicePtr cxRaw(cellState);

        DevicePtr dxRaw(dx);
        DevicePtr dhxRaw(dhx);
        DevicePtr dcxRaw(dcx);

        // ensure cudnn compute stream waits on input/output tensor streams
        relativeSync(
            cudnnStream,
            {dy, dhy, dcy, weights, hiddenState, cellState, dx, dhx, dcx}
        );

        /* We need to update reserveSpace even if we just want the
         * weight gradients. */
        CUDNN_CHECK_ERR(
            cudnnRNNBackwardData_v8(
                handle,
                rnnDesc.get(),
                devSeqLengthsRaw ? devSeqLengthsRaw->getAs<int32_t const>() : nullptr,
                yDesc.get(),
                yRaw.get(),
                dyRaw.get(),
                xDesc.get(),
                dxRaw.get(),
                hxDesc.get(),
                hxRaw.get(),
                dhyRaw.get(),
                dhxRaw.get(),
                cxDesc.get(),
                cxRaw.get(),
                dcyRaw.get(),
                dcxRaw.get(),
                weightSpaceSize,
                wRaw.get(),
                workspaceSize,
                workspaceRaw.get(),
                reserveSpace.bytes(),
                reserveSpaceRaw.get()
            )
        );
    }

    {
        DevicePtr xRaw(x);
        DevicePtr dwRaw(dw);
        DevicePtr hxRaw(hiddenState);

        // ensure cudnn compute stream waits on input/output tensor streams
        relativeSync(cudnnStream, {x, dw, hiddenState});

        CUDNN_CHECK_ERR(
            cudnnRNNBackwardWeights_v8(
                handle,
                rnnDesc.get(),
                CUDNN_WGRAD_MODE_ADD,
                devSeqLengthsRaw ? devSeqLengthsRaw->getAs<int32_t const>() : nullptr,
                xDesc.get(),
                xRaw.get(),
                hxDesc.get(),
                hxRaw.get(),
                yDesc.get(),
                yRaw.get(),
                weightSpaceSize,
                dwRaw.get(),
                workspaceSize,
                workspaceRaw.get(),
                reserveSpace.bytes(),
                reserveSpaceRaw.get()
            )
        );
    }

    // ensure output tensor streams wait on cudnn compute stream
    relativeSync({dx, dhx, dcx, dw}, cudnnStream);
}
}

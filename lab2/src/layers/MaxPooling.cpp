#include "MaxPooling.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

const int U = 2;

// Compute the convultion for the layer data
void MaxPoolingLayer::computeNaive(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // The following line is an example of copying a single 32-bit floating point integer from the input layer data to the output layer data
    // getOutputData().get<fp32>(0) = dataIn.get<fp32>(0);LayerParams params = getInputParams();
    const auto& inputParams    = getInputParams();
    const auto& outputParams   = getOutputParams();
    // const auto& weightParams   = getWeightParams();
    // const auto& biasParams     = getBiasParams();
    // const auto& weightDims     = weightParams.dims;  // [R,S,C,K]
    const auto& inputDims      = inputParams.dims;   // [H,W,C]
    const auto& outputDims     = outputParams.dims;  // [P,Q,K]
    // const auto& biasDims       = biasParams.dims;    // [K]

    // size_t H = inputDims[0];
    // size_t W = inputDims[1];
    size_t C = inputDims[2];

    size_t R = 2;
    size_t S = 2;
    // size_t C = weightDims[2];  // channels
    // size_t K = weightDims[3];  // number of filters

    auto steps = [](const std::vector<size_t>& d) {
        std::vector<size_t> s(d.size(), 1);
        for (int i = int(d.size()) - 2; i >= 0; --i) s[i] = s[i + 1] * d[i + 1];
        return s;
    };

    const auto inputStep  = steps(inputDims);   // [H,W,C]
    const auto outputStep = steps(outputDims);  // [P,Q,K]
    // const auto weightStep = steps(weightDims);  // [R,S,C,K]
    // const auto biasStep   = steps(biasDims);    // [K]

    // const LayerData& weightData = getWeightData();
    // const LayerData& biasData   = getBiasData();
    LayerData& outputData       = getOutputData();

    const size_t P = outputDims[0];
    const size_t Q = outputDims[1];

    for (size_t c = 0; c < C; ++c) {
        for (size_t p = 0; p < P; ++p) {
            for (size_t q = 0; q < Q; ++q) {
                const size_t in_y0 = static_cast<size_t>(p) * static_cast<size_t>(U);
                const size_t in_x0 = static_cast<size_t>(q) * static_cast<size_t>(U);

                fp32 maxVal = ((const fp32*)dataIn.raw())[
                    (in_y0 * inputStep[0]) + (in_x0 * inputStep[1]) + (c * inputStep[2])
                ];

                for (size_t r = 0; r < R; ++r) {
                    const size_t pVal = in_y0 + r;
                    for (size_t s = 0; s < S; ++s) {
                        const size_t qVal = in_x0 + s;

                        const size_t inIdx = pVal * inputStep[0] + qVal * inputStep[1] + c * inputStep[2];
                        const fp32 v = ((const fp32*)dataIn.raw())[inIdx];
                        if (v > maxVal) maxVal = v;
                    }
                }

                const size_t outIdx = p * outputStep[0] + q * outputStep[1] + c * outputStep[2];
                outputData.get<fp32>((unsigned)outIdx) = maxVal;
            }
        }
    }
}

// Compute the convolution using threads
void MaxPoolingLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    
}

// Compute the convolution using a tiled approach
void MaxPoolingLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using SIMD
void MaxPoolingLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML
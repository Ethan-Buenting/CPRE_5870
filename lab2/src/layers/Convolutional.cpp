#include "Convolutional.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

// Compute the convultion for the layer data
// Compute the convultion for the layer data
void ConvolutionalLayer::computeNaive(const LayerData& dataIn) const {
    // Keep your variable names
    LayerParams params = getInputParams();
    const auto& inputParams    = getInputParams();
    const auto& outputParams   = getOutputParams();
    const auto& weightParams   = getWeightParams();
    const auto& biasParams     = getBiasParams();
    const auto& weightDims     = weightParams.dims;  // [R,S,C,K]
    const auto& inputDims      = inputParams.dims;   // [H,W,C]
    const auto& outputDims     = outputParams.dims;  // [P,Q,K]
    const auto& biasDims       = biasParams.dims;    // [K]

    // size_t H = inputDims[0];
    // size_t W = inputDims[1];
    size_t C = inputDims[2];

    size_t R = weightDims[0];  // filter rows
    size_t S = weightDims[1];  // filter cols
    // size_t C = weightDims[2];  // channels
    size_t K = weightDims[3];  // number of filters

    auto steps = [](const std::vector<size_t>& d) {
        std::vector<size_t> s(d.size(), 1);
        for (int i = int(d.size()) - 2; i >= 0; --i) s[i] = s[i + 1] * d[i + 1];
        return s;
    };

    const auto inputStep  = steps(inputDims);   // [H,W,C]
    const auto outputStep = steps(outputDims);  // [P,Q,K]
    const auto weightStep = steps(weightDims);  // [R,S,C,K]
    const auto biasStep   = steps(biasDims);    // [K]

    const LayerData& weightData = getWeightData();
    const LayerData& biasData   = getBiasData();
    LayerData& outputData       = getOutputData();

    const size_t P = outputDims[0];
    const size_t Q = outputDims[1];

    auto relu = [](fp32 g) -> fp32 { return g < 0.0f ? 0.0f : g; };

    for (size_t m = 0; m < K; ++m) {
        for (size_t p = 0; p < P; ++p) {
            for (size_t q = 0; q < Q; ++q) {
                fp32 sum = 0.0f;
                for (size_t c = 0; c < C; ++c) {
                    for (size_t r = 0; r < R; ++r) {
                        const size_t pVal = p + r; // Up + r  U=1
                        for (size_t s = 0; s < S; ++s) {
                            const size_t qVal = q + s; // Uq + s U=1
                            // I{n c (Up+r) (Uq+s)}
                            const size_t inIdx = pVal*inputStep[0]+qVal*inputStep[1]+c*inputStep[2];
                            // F{m c r s}
                            const size_t wIdx = r*weightStep[0]+s*weightStep[1]+c*weightStep[2]+m*weightStep[3];
                            sum += ((const fp32*)dataIn.raw())[inIdx]*((const fp32*)weightData.raw())[wIdx];
                        }
                    }
                }
                sum += ((const fp32*)biasData.raw())[m*biasStep[0]];
                sum = relu(sum);
                const size_t outIdx = p*outputStep[0]+q*outputStep[1]+m*outputStep[2];

                outputData.get<fp32>(static_cast<unsigned>(outIdx)) = sum;
            }
        }
    }
}


// Compute the convolution using threads
void ConvolutionalLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using a tiled approach
void ConvolutionalLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using SIMD
void ConvolutionalLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML

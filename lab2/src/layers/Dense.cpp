#include "Dense.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
void DenseLayer::computeNaive(const LayerData& dataIn) const {
    // Keep your variable names
    const auto& inputParams    = getInputParams();
    const auto& outputParams   = getOutputParams();

    const LayerData& weightData = getWeightData();
    const LayerData& biasData   = getBiasData();
    LayerData& outputData       = getOutputData();

    const size_t N = inputParams.flat_count();
    const size_t M = outputParams.flat_count();

    auto relu = [](fp32 g) -> fp32 { return g < 0.0f ? 0.0f : g; };

    for (size_t m = 0; m < M; ++m) {
        fp32 sum = 0.0f;

        for (size_t n = 0; n < N; ++n) {
            const size_t wIdx = n * M + m;
            sum += ((const fp32*)dataIn.raw())[n] * ((const fp32*)weightData.raw())[wIdx];
        }

        sum += ((const fp32*)biasData.raw())[m];
        if(activation == Activation::ReLU) { sum = relu(sum); }
        
        outputData.get<fp32>((unsigned)m) = sum;
    }

}


// Compute the convolution using threads
void DenseLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using a tiled approach
void DenseLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using SIMD
void DenseLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML

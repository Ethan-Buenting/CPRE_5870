#include "Softmax.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

const int U = 2;

// Compute the convultion for the layer data
void SoftmaxLayer::computeNaive(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // The following line is an example of copying a single 32-bit floating point integer from the input layer data to the output layer data
    // getOutputData().get<fp32>(0) = dataIn.get<fp32>(0);LayerParams params = getInputParams();
    // const auto& inputParams    = getInputParams();
    const auto& outputParams   = getOutputParams();
    // const auto& weightParams   = getWeightParams();
    // const auto& biasParams     = getBiasParams();
    // const auto& weightDims     = weightParams.dims;  // [R,S,C,K]
    // const auto& inputDims      = inputParams.dims;   // [H,W,C]
    // const auto& outputDims     = outputParams.dims;  // [P,Q,K]
    // const auto& biasDims       = biasParams.dims;    // [K]

 
    // auto steps = [](const std::vector<size_t>& d) {
    //     std::vector<size_t> s(d.size(), 1);
    //     for (int i = int(d.size()) - 2; i >= 0; --i) s[i] = s[i + 1] * d[i + 1];
    //     return s;
    // };

    // const auto inputStep  = steps(inputDims);   // [H,W,C]
    // const auto outputStep = steps(outputDims);  // [P,Q,K]
    // const auto weightStep = steps(weightDims);  // [R,S,C,K]
    // const auto biasStep   = steps(biasDims);    // [K]

    // const LayerData& weightData = getWeightData();
    // const LayerData& biasData   = getBiasData();
    LayerData& outputData = getOutputData();

    // const size_t P = outputDims[0];
    // const size_t Q = outputDims[1];
    const size_t N = outputParams.flat_count();

    const fp32* g = (const fp32*)(dataIn.raw());

    fp32 sum = 0.0f;
    for(size_t i=0; i<N; i++){
        fp32 e = std::exp(g[i]);
        outputData.get<fp32>(i) = e;
        sum += e;
    }

    fp32 invSum = 1.0 /sum;
    for(size_t i=0; i<N; i++){
        fp32 val = outputData.get<fp32>(i);
        outputData.get<fp32>(i) = (val * invSum);
    }


    
}

// Compute the convolution using threads
void SoftmaxLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    
}

// Compute the convolution using a tiled approach
void SoftmaxLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using SIMD
void SoftmaxLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML
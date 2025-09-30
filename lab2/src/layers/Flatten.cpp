#include "Flatten.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

const int U = 2;

// Compute the convultion for the layer data
void FlattenLayer::computeNaive(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // The following line is an example of copying a single 32-bit floating point integer from the input layer data to the output layer data
    // getOutputData().get<fp32>(0) = dataIn.get<fp32>(0);LayerParams params = getInputParams();
    const auto& inputParams    = getInputParams();
    LayerData& outputData = getOutputData();
    const size_t N = inputParams.flat_count();

    for (size_t i=0; i<N; i++){
        outputData.get<fp32>(static_cast<unsigned>(i)) = dataIn.get<fp32>(static_cast<unsigned>(i));
    }
    
}

// Compute the convolution using threads
void FlattenLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    
}

// Compute the convolution using a tiled approach
void FlattenLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using SIMD
void FlattenLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML
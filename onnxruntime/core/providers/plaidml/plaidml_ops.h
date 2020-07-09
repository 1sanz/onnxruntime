// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#pragma once

#include <functional>
#include <map>
#include <string>

#include "plaidml/edsl/edsl.h"

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace plaidml_ep {

using OpFunction = std::function<std::vector<plaidml::edsl::Tensor>(const std::vector<plaidml::edsl::Value>& args)>;

std::vector<plaidml::edsl::Tensor> abs(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> add(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> argmax(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> asin(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> acos(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> atan(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> asinh(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> ceil(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> clip(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> conv(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> cos(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> cosh(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> cumsum(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> div(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> exp(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> flatten(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> floor(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> less(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> log(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> matmul(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> mul(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> relu(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> reshape(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sigmoid(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sin(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sinh(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> softmax(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sqrt(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sub(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> tan(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> tanh(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> transpose(const std::vector<plaidml::edsl::Value>& args);

//---unsqueeze
//maxpool
//---LRN
//avgpool
//concat
//reshape
//batchnorm
//---asin
//---acos
//---atan
//---sinh
//---cosh
//---tanh


//TODO: for some os more information from the node is required this is not the best way to do this. 
// openvino approach is better - need to create a structure to hold context 
std::vector<plaidml::edsl::Tensor> _argmax(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
 std::vector<plaidml::edsl::Tensor> _argmin(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _concat(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _conv(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _LRN(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
std::vector<plaidml::edsl::Tensor> _softmax(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);
 std::vector<plaidml::edsl::Tensor> _unsqueeze(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);


// trig ops check in plaidml mlir implementation? write notes in layers doc 
// std::vector<plaidml::edsl::Tensor> reshape(const std::vector<plaidml::edsl::Value>& args);
//std::vector<plaidml::edsl::Tensor> hardsigmoid(const std::vector<plaidml::edsl::Value>& args);

//openvino supported ops 
    //   ---"Add",
    //   "BatchNormalization",
    //   ---"Conv",
    //   "GlobalAveragePool",
    //   ---"Relu",
    //   "Reshape",
    //   ---"Flatten",
    //   "Gemm",
    //   "MaxPool",
    //   "AveragePool",
    //   "Concat",
    //   "Dropout",
    //   "LRN",
    //   ---"Softmax",
    //   ---"Mul",
    //   "Sum",
    //   ---"Transpose",
    //   "Identity",
    //   ---"MatMul",
    //   "Pad",
    //   "Unsqueeze",
    //   "ImageScaler",
    //   "LeakyRelu",
    //   "GlobalMaxPool",
    //   ---"Div",
    //   ---"Sub"


// IRV10 ops done

// Abs
// Acos
// Add
// Asin
// Atan
// AvgPool
// BatchNormInference
// Broadcast
// Ceiling
// Clamp
// Concat
// Constant
// Convert
// Convolution
// ConvolutionBackpropData
// Cos
// Cosh
// DepthToSpace
// Divide
// Elu
// Erf
// Exp
// Floor
// GRN
// GroupConvolution
// GroupConvolutionBackpropData
// HardSigmoid
// Log
// LRN
// MatMul
// Maximum
// MaxPool
// Minimum
// Mod
// Multiply
// Negative
// NormalizeL2
// Parameter
// Power
// PReLU
// ReduceMax
// ReduceMean
// ReduceMin
// ReduceProd
// ReduceSum
// ReLU
// Reshape
// Result
// Selu
// Sigmoid
// Sign
// Sin
// Sinh
// Softmax
// SpaceToDepth
// Sqrt
// SquaredDifference
// Squeeze
// StridedSlice
// Subtract
// Tan
// Tanh
// Transpose
// Unsqueeze
std::vector<plaidml::edsl::Tensor> MakePlaidMLOp(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);

}  // namespace plaidml_ep
}  // namespace onnxruntime
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
std::vector<plaidml::edsl::Tensor> ceil(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> clip(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> conv(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> cos(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> cosh(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> cumsum(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> div(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> exp(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> less(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> log(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> mul(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> relu(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sigmoid(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sin(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sinh(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> softmax(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sqrt(const std::vector<plaidml::edsl::Value>& args);
std::vector<plaidml::edsl::Tensor> sub(const std::vector<plaidml::edsl::Value>& args);

std::vector<plaidml::edsl::Tensor> MakePlaidMLOp(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs);

}  // namespace plaidml_ep
}  // namespace onnxruntime
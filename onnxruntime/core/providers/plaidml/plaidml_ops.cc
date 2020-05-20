// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include "plaidml_ops.h"

#include "plaidml/edsl/edsl.h"
#include "plaidml/op/op.h"

namespace onnxruntime {
namespace plaidml_ep {

std::map<std::string, OpFunction> kSupportedOps = {
  {"Abs", abs},
  {"Add", add},
  {"Argmax", argmax},
  {"Ceil", ceil},
  {"Clip", clip},
  {"Conv", conv},
  {"Cos", cos},
  {"Cosh", cosh},
  {"Cumsum", cumsum},
  {"Div", div},
  {"Exp", exp},
  {"Less", less},
  {"Log", log},
  {"Mul", mul},
  {"Relu", relu},
  {"Sigmoid", sigmoid},
  {"Sin", sin},
  {"Sinh", sinh},
  {"Softmax", softmax},
  {"Sqrt", sqrt,},
  {"Sub", sub,},
};

std::vector<plaidml::edsl::Tensor> abs(const std::vector<plaidml::edsl::Value>& args) {
  const auto& X = args[0].as_tensor();
  return {plaidml::op::abs(X)};
}

std::vector<plaidml::edsl::Tensor> add(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A + B};
}

std::vector<plaidml::edsl::Tensor> argmax(const std::vector<plaidml::edsl::Value>& args) {
  //TODO: argument mismatch onnx -> (I, axis, keepdims)
  const auto I = args[0].as_tensor();
  const auto axes = args[1];
  return {plaidml::op::argmax(I,axes)};
}

std::vector<plaidml::edsl::Tensor> ceil(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::ceil(A)};
}

std::vector<plaidml::edsl::Tensor> clip(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& min_val = args[1].as_tensor();
  const auto& max_val = args[2].as_tensor();
  return {plaidml::op::clip(A,min_val,max_val)};
}

std::vector<plaidml::edsl::Tensor> conv(const std::vector<plaidml::edsl::Value>& args) {
  const auto I = args[0].as_tensor();
  const auto K = args[1].as_tensor();
  plaidml::edsl::TensorDim N, X, KX, CI, CO;
  plaidml::edsl::TensorIndex n, x, k, ci, co;
  I.bind_dims(N, X, CI);
  K.bind_dims(KX, CI, CO);
  auto O = plaidml::edsl::TensorOutput(N, X - KX + 1, CO);
  O(n, x, co) += I(n, x + k, ci) * K(k, ci, co);
  return {O};
}

std::vector<plaidml::edsl::Tensor> cos(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::cos(A)};
}

std::vector<plaidml::edsl::Tensor> cosh(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::cosh(A)};
}

std::vector<plaidml::edsl::Tensor> cumsum(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_int();
  return {plaidml::op::cumsum(A,B)};
}

std::vector<plaidml::edsl::Tensor> div(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A / B};
}

std::vector<plaidml::edsl::Tensor> exp(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::exp(A)};
}

std::vector<plaidml::edsl::Tensor> less(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A < B};
}

std::vector<plaidml::edsl::Tensor> log(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::log(A)};
}

std::vector<plaidml::edsl::Tensor> mul(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A * B};
}

std::vector<plaidml::edsl::Tensor> relu(const std::vector<plaidml::edsl::Value>& args) {
  const auto& X = args[0].as_tensor();
  return {plaidml::op::relu(X)};
}

std::vector<plaidml::edsl::Tensor> sigmoid(const std::vector<plaidml::edsl::Value>& args) {
  const auto& X = args[0].as_tensor();
  return {plaidml::op::sigmoid(X)};
}

std::vector<plaidml::edsl::Tensor> sin(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::sin(A)};
}

std::vector<plaidml::edsl::Tensor> sinh(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::sinh(A)};
}

std::vector<plaidml::edsl::Tensor> softmax(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_int();
  return {plaidml::op::softmax(A,B)};
}

std::vector<plaidml::edsl::Tensor> sqrt(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::sqrt(A)};
}

std::vector<plaidml::edsl::Tensor> sub(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A - B};
}

std::vector<plaidml::edsl::Tensor> MakePlaidMLOp(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {

  auto op_it = plaidml_ep::kSupportedOps.find(node.op_type());
  if (op_it == plaidml_ep::kSupportedOps.end()) {
    throw std::runtime_error("Asked to executed unsupported op " + node.op_type());
  }
  return op_it->second(inputs);
}

}  // namespace plaidml_ep
}  // namespace onnxruntime
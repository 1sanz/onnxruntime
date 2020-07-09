// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include "plaidml_ops.h"

#include "plaidml/edsl/edsl.h"
#include "plaidml/op/op.h"

//TODO: using value of pi to compute sin from cos (find a better way )
#define _USE_MATH_DEFINES
#include <cmath>

namespace onnxruntime {
namespace plaidml_ep {

std::map<std::string, OpFunction> kSupportedOps = {
  {"Abs", abs},
  {"Add", add},
  {"Argmax", argmax},
  {"Asin", asin},
  {"Acos", acos},
  {"Atan", atan},
  //{"Asinh", asinh},
  {"Ceil", ceil},
  {"Clip", clip},
  {"Conv", conv},
  {"Cos", cos},
  {"Cosh", cosh},
  {"Cumsum", cumsum},
  {"Div", div},
  {"Exp", exp},
  {"Flatten", flatten},
  {"Floor", floor},
  {"Less", less},
  {"Log", log},
  {"MatMul", matmul},
  {"Mul", mul},
  {"Relu", relu},
  {"Sigmoid", sigmoid},
  {"Sin", sin},
  {"Sinh", sinh},
  {"Softmax", softmax},
  {"Sqrt", sqrt,},
  {"Sub", sub,},
  {"Tan", tan,},
  {"Tanh", tanh,},
  {"Transpose", transpose,},
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
  // figure out how to pass node attributes as arguments
  const auto I = args[0].as_tensor();
  const auto axes = args[1];
  return {plaidml::op::argmax(I,axes)};
}

std::vector<plaidml::edsl::Tensor> asin(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::asin(A)};
}

// std::vector<plaidml::edsl::Tensor> asinh(const std::vector<plaidml::edsl::Value>& args) {
//   const auto A = args[0].as_tensor();
//   return {plaidml::edsl::asinh(A)};
// }

std::vector<plaidml::edsl::Tensor> acos(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::acos(A)};
}

std::vector<plaidml::edsl::Tensor> atan(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::atan(A)};
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
  // TODO: figure out how to translate node attributes into arguments 
  // return {plaidml::op::convolution(I,K)
  //           .deriv_mode(plaidml::op::ConvDerivMode::NONE)
  //           .group_layout(plaidml::op::GroupLayout::NONE)
  //           .autogroup_mode(plaidml::op::AutoGroupMode::UNGROUPED)
  //           .autopad_mode(plaidml::op::AutoPadMode::SAME_UPPER)
  //           .input_layout(plaidml::op::TensorLayout::NCX)
  //           .filter_layout(plaidml::op::TensorLayout::KCX)};
    return {plaidml::op::convolution(I,K)
            .input_layout(plaidml::op::TensorLayout::NCX)
            .filter_layout(plaidml::op::TensorLayout::KCX)};
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

std::vector<plaidml::edsl::Tensor> flatten(const std::vector<plaidml::edsl::Value>& args) {
    const auto X = args[0].as_tensor();
    std::vector<plaidml::edsl::TensorDim> X_dims(X.rank());
    X.bind_dims(X_dims);
    if (X_dims.empty()) {
      return {X};
    }
    plaidml::edsl::TensorDim product(1);
    for (size_t i = 1; i < X.rank(); i++) {
      product = product * X_dims[i];
    }
    return {reshape(X, {X_dims[0], product})};
}

std::vector<plaidml::edsl::Tensor> floor(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::floor(A)};
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

std::vector<plaidml::edsl::Tensor> matmul(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {plaidml::op::dot(A,B)};
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
  const auto& A = args[0].as_tensor();
  return {plaidml::edsl::cos(M_PI_2 - A)};
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

std::vector<plaidml::edsl::Tensor> tan(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::tan(A)};
}

std::vector<plaidml::edsl::Tensor> tanh(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::tanh(A)};
}

std::vector<plaidml::edsl::Tensor> transpose(const std::vector<plaidml::edsl::Value>& args){
  const auto& A = args[0].as_tensor();
  return {plaidml::op::transpose(A)};
}

std::vector<plaidml::edsl::Tensor> _unsqueeze(    
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  const auto& A = inputs[0].as_tensor();
  std::vector<int64_t> axes;
  auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      printf("more attributes %d \n",num_attributes);
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++){
        printf("attribute %s", attribute->name().c_str());
        if(attribute->name() == "axes"){
         auto  at_axes = attribute->ints();
         for(auto at_axis = at_axes.begin();at_axis<at_axes.end();at_axis++){
           axes.push_back(*at_axis);
         }
        }
      }
    }
  return {plaidml::op::unsqueeze(A,{axes})};
}

std::vector<plaidml::edsl::Tensor> _LRN(    
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  const auto& A = inputs[0].as_tensor();
  float alpha = 0.0001;
  float beta = 0.75;
  float bias = 1.0;
  int size = 1;

  auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      printf("more attributes %d \n",num_attributes);
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++){
        if(attribute->name() == "alpha"){
         alpha = attribute->f();
        }
        if(attribute->name()=="beta"){
          beta = attribute->f();
        }
        if(attribute->name()=="bias"){
          bias = attribute->f();
        }
        if(attribute->name()=="size"){
          size = attribute->i();
        }
      }
    }
  return {plaidml::op::lrn(A,{static_cast<int64_t>(size)}).alpha(alpha).beta(beta).epsilon(bias)};
}

std::vector<plaidml::edsl::Tensor> _softmax(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  
    const auto& A = inputs[0].as_tensor();
    int axis = 1;
    auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++)
      {
        if(attribute->name() == "axis")
        {
          const auto at_axis = attribute->i();
          axis = at_axis;
        }
      }
    }
  return {plaidml::op::softmax(A,axis)};
}

// TODO: fix inputs 
// std::vector<plaidml::edsl::Tensor> _concat(
//     const ONNX_NAMESPACE::NodeProto& node,
//     const std::vector<plaidml::edsl::Value>& inputs){
  
//     const std::vector<plaidml::edsl::Tensor> A = inputs[0];
//     int axis = 1;
//     auto num_attributes = node.attribute_size();
//     if(num_attributes>0){
//       auto attributes = node.attribute();
//       for(auto attribute = attributes.begin();attribute < attributes.end();attribute++)
//       {
//         if(attribute->name() == "axis")
//         {
//           const auto at_axis = attribute->i();
//           axis = at_axis;
//         }
//       }
//     }
//   return {plaidml::op::concatenate(A,axis)};
// }

std::vector<plaidml::edsl::Tensor> _argmax(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  
    const auto& A = inputs[0].as_tensor();
    int B=1;
    auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      printf("more attributes %d \n",num_attributes);
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++)
      {
        printf("attribute %s", attribute->name().c_str());
        if(attribute->name() == "axis")
        {
          const auto axis = attribute->i();
          printf("Axis value %lld\n", axis);

          B = axis;
        }
      }
    }
  return {plaidml::op::argmax(A,plaidml::edsl::make_tuple(B))};
}

std::vector<plaidml::edsl::Tensor> _argmin(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  
    const auto& A = inputs[0].as_tensor();
    int B=1;
    auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      printf("more attributes %d \n",num_attributes);
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++)
      {
        printf("attribute %s", attribute->name().c_str());
        if(attribute->name() == "axis")
        {
          const auto axis = attribute->i();
          printf("Axis value %lld\n", axis);

          B = axis;
        }
      }
    }
  //TODO: needs negative ?
  return {plaidml::op::argmax(A,plaidml::edsl::make_tuple(B))};
}

std::vector<plaidml::edsl::Tensor> _conv(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){

    const auto I = inputs[0].as_tensor();
    const auto K = inputs[1].as_tensor();
        //int B=1;
    auto num_attributes = node.attribute_size();

    //default auto_pad mode
    auto auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;
    int group =1;
    if(num_attributes>0){
      printf("more attributes %d \n",num_attributes);
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++)
      {
        printf("attribute %s", attribute->name().c_str());
        if(attribute->name() == "auto_pad"){
          const auto auto_pad = attribute->s();
          printf("Auto_pad value %s\n", auto_pad.c_str());
          if(auto_pad=="SAME_UPPER")auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;
          if(auto_pad=="SAME_LOWER")auto_pad_mode = plaidml::op::AutoPadMode::SAME_LOWER;
          if(auto_pad=="VALID")auto_pad_mode = plaidml::op::AutoPadMode::VALID;

        }
        if(attribute->name()=="group"){
          group = attribute->i();
        }
        if(attribute->name()=="dilations"){
          //if not present default is 1 along each spacial axis
          auto dilations_ints = attribute->ints();//returns repeated field 
        }
        if(attribute->name()=="kernel_shape"){
          //If not present, should be inferred from input W
          auto kernel_shape_ints = attribute->ints();
        }
        if(attribute->name()=="pads"){
          // This attribute cannot be used simultaneously with 
          // auto_pad attribute. If not present, 
          // the padding defaults to 0 along start and end of each spatial axis.
          auto pads_ints = attribute->ints();

        }
        if(attribute->name()=="strides"){
          //If not present, the stride defaults is 1 along each spatial axis
          auto strides_ints = attribute->ints();
        }
      }
    }
  // TODO: figure out how to translate node attributes into arguments 
  // return {plaidml::op::convolution(I,K)
  //           .deriv_mode(plaidml::op::ConvDerivMode::NONE)
  //           .group_layout(plaidml::op::GroupLayout::NONE)
  //           .autogroup_mode(plaidml::op::AutoGroupMode::UNGROUPED)
  //           .autopad_mode(plaidml::op::AutoPadMode::SAME_UPPER)
  //           .input_layout(plaidml::op::TensorLayout::NCX)
  //           .filter_layout(plaidml::op::TensorLayout::KCX)};
    return {plaidml::op::convolution(I,K)
            .input_layout(plaidml::op::TensorLayout::NCX)
            .filter_layout(plaidml::op::TensorLayout::KCX)
            .autopad_mode(auto_pad_mode)
            .groups(group)};

    }

std::vector<plaidml::edsl::Tensor> MakePlaidMLOp(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
    
  //check if this op is supported 
  //TODO: might mive op translation logic into op functions will need to pass in node 
  // maybe split into two hashmaps one for direct translation and another for rest of 
  // the ops which require more logic and need to node passed in as an argument. 
  const std::vector<plaidml::edsl::Value *> args;

  auto op_it = plaidml_ep::kSupportedOps.find(node.op_type());
  if (op_it == plaidml_ep::kSupportedOps.end()) {
    throw std::runtime_error("Asked to executed unsupported op " + node.op_type());
  }

  //elementwise operations 

  //variable number of arguments 

  //attributes as arguments 
    //softmax axis 
    //convolution types 

  //get node attributes 
  
  

  //check if this is softmax : 
    if(node.op_type()=="Softmax"){
      return _softmax(node,inputs);
    }
    if(node.op_type()=="Conv"){
      return _conv(node,inputs);
    }
    if(node.op_type()=="Argmax"){
      return _argmax(node,inputs);
    }
    // if(node.op_type()=="Unsqueeze"){
    //   return _unsqueeze(node,inputs);
    // }

    // auto axis = attributes["axis"].i();
    // inputs.push_back(plaidml::edsl::Value(axis));
  

  return op_it->second(inputs);
}

}  // namespace plaidml_ep
}  // namespace onnxruntime
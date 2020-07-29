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
  {"And", logical_and},
  {"Asin", asin},
  {"Acos", acos},
  {"Atan", atan},
  {"Ceil", ceil},
  {"Clip", clip},
  {"Cos", cos},
  {"Cosh", cosh},
  //{"CumSum", cumsum},
  {"Div", div},
  {"Equal", equal},
  {"Erf", erf},
  {"Exp", exp},
  //{"Flatten", flatten},
  {"Floor", floor},
  {"Greater", greater},
  {"Less", less},
  {"Log", log},
  {"Max", max},
  //{"MatMul", matmul},
  {"Min", min},
  {"Mul", mul},
  {"Neg", neg},
  {"Not", logical_not},
  {"Or", logical_or},
  {"Pow", pow},
  {"PRelu", prelu},
  {"Reciprocal", reciprocal},
  {"Relu", relu},
  {"Reshape", reshape},
  {"SampleOp", sample_op},
  {"Sigmoid", sigmoid},
  {"Sign", sign},
  {"Sin", sin},
  {"Sinh", sinh},
  {"Sqrt", sqrt,},
  {"Sub", sub,},
  {"Sum", sum,},
  {"Tan", tan,},
  {"Tanh", tanh,},
  {"Tile", tile,},
  {"Transpose", transpose,},
  {"Xor", logical_xor,},
};

std::map<std::string, _OpFunction> _kSupportedOps = 
{
  {"ArgMax", _argmax,},
  {"Conv",_conv},
  {"Concat",_concat},
  {"CumSum", _cumsum},
  {"Elu",_elu},
  {"Flatten", _flatten},
  {"HardSigmoid",_hard_sigmoid},
  {"LogSoftmax",_log_softmax},
  {"LRN",_LRN},
  {"Selu",_selu},
  {"Softmax",_softmax},
  {"Squeeze",_squeeze},
  {"Unsqueeze",_unsqueeze},

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

std::vector<plaidml::edsl::Tensor> logical_and(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A && B};
}

std::vector<plaidml::edsl::Tensor> logical_not(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  auto T = plaidml::edsl::Tensor(true);
  auto F = plaidml::edsl::Tensor(false);
  return {plaidml::edsl::select(I, F, T)};
}

std::vector<plaidml::edsl::Tensor> logical_or(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A || B};
}

std::vector<plaidml::edsl::Tensor> logical_xor(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A ^ B};
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

// std::vector<plaidml::edsl::Tensor> conv(const std::vector<plaidml::edsl::Value>& args) {
//   const auto I = args[0].as_tensor();
//   const auto K = args[1].as_tensor();
//   // TODO: figure out how to translate node attributes into arguments 
//   // return {plaidml::op::convolution(I,K)
//   //           .deriv_mode(plaidml::op::ConvDerivMode::NONE)
//   //           .group_layout(plaidml::op::GroupLayout::NONE)
//   //           .autogroup_mode(plaidml::op::AutoGroupMode::UNGROUPED)
//   //           .autopad_mode(plaidml::op::AutoPadMode::SAME_UPPER)
//   //           .input_layout(plaidml::op::TensorLayout::NCX)
//   //           .filter_layout(plaidml::op::TensorLayout::KCX)};
//     return {plaidml::op::convolution(I,K)
//             .input_layout(plaidml::op::TensorLayout::NCX)
//             .filter_layout(plaidml::op::TensorLayout::KCX)};
// }
  
std::vector<plaidml::edsl::Tensor> cos(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::cos(A)};
}

std::vector<plaidml::edsl::Tensor> cosh(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::cosh(A)};
}

// std::vector<plaidml::edsl::Tensor> cumsum(const std::vector<plaidml::edsl::Value>& args) {
//   const auto& A = args[0].as_tensor();
//   const auto& B = args[1].as_int();
//   return {plaidml::op::cumsum(A,B)};
// }

std::vector<plaidml::edsl::Tensor> div(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A / B};
}

std::vector<plaidml::edsl::Tensor> equal(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A == B};
}

std::vector<plaidml::edsl::Tensor> erf(const std::vector<plaidml::edsl::Value>& args){
  const auto& A = args[0].as_tensor();
  return {plaidml::edsl::erf(A)};
}

std::vector<plaidml::edsl::Tensor> exp(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::exp(A)};
}

// std::vector<plaidml::edsl::Tensor> flatten(const std::vector<plaidml::edsl::Value>& args) {
//     const auto X = args[0].as_tensor();
//     std::vector<plaidml::edsl::TensorDim> X_dims(X.rank());
//     X.bind_dims(X_dims);
//     if (X_dims.empty()) {
//       return {X};
//     }
//     plaidml::edsl::TensorDim product(1);
//     for (size_t i = 1; i < X.rank(); i++) {
//       product = product * X_dims[i];
//     }
//     return {reshape(X, {X_dims[0], product})};
// }

std::vector<plaidml::edsl::Tensor> floor(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::floor(A)};
}

std::vector<plaidml::edsl::Tensor> greater(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A > B};
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

std::vector<plaidml::edsl::Tensor> max(const std::vector<plaidml::edsl::Value>& args) {
  plaidml::edsl::Tensor result = args[0].as_tensor();
  if(args.size()==1)return {result};
  for(size_t i=1;i<args.size();i++){
    const auto& B  = args[i].as_tensor();
    result = plaidml::op::maximum(result,B);
  }
  return {result};
}

std::vector<plaidml::edsl::Tensor> matmul(const std::vector<plaidml::edsl::Value>& args) {
  auto A = args[0].as_tensor();
  auto B = args[1].as_tensor();
  //numpy style matmul 
  // if both arguments are 2-D multiply like normal matrices
  // if dims > 2 

  return {plaidml::op::dot(A,B)};
}

std::vector<plaidml::edsl::Tensor> min(const std::vector<plaidml::edsl::Value>& args) {
  plaidml::edsl::Tensor result = args[0].as_tensor();
  if(args.size()==1)return {result};
  for(size_t i=1;i<args.size();i++){
    const auto& B  = args[i].as_tensor();
    result = plaidml::op::minimum(result,B);
  }
  return {result};
}

std::vector<plaidml::edsl::Tensor> mul(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A * B};
}

std::vector<plaidml::edsl::Tensor> neg(const std::vector<plaidml::edsl::Value>& args){
  const auto& A = args[0].as_tensor();
  return {-A};
}

std::vector<plaidml::edsl::Tensor> pow(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  const auto& p = args[1].as_tensor();
  return {plaidml::edsl::pow(I,p)};
}

std::vector<plaidml::edsl::Tensor> prelu(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  const auto& slope = args[1].as_tensor();
  auto O = select(I < 0.0, slope * I, I);
  return {O};
}

std::vector<plaidml::edsl::Tensor> reciprocal(const std::vector<plaidml::edsl::Value>& args) {
  const auto& X = args[0].as_tensor();
  return {1/X};
}

std::vector<plaidml::edsl::Tensor> relu(const std::vector<plaidml::edsl::Value>& args) {
  const auto& X = args[0].as_tensor();
  return {plaidml::op::relu(X)};
}

std::vector<plaidml::edsl::Tensor> reshape(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  const auto& shape = args[1].as_tensor();//need to convert this into 
  return {plaidml::op::reshape(I,plaidml::edsl::make_tuple(shape))};
}

std::vector<plaidml::edsl::Tensor> sample_op(const std::vector<plaidml::edsl::Value>& args) {
  const auto& X = args[0].as_tensor();
  return {X};
}

std::vector<plaidml::edsl::Tensor> sign(const std::vector<plaidml::edsl::Value>& args) {
  const auto& I = args[0].as_tensor();
  auto Z = plaidml::edsl::Tensor(0.0);
  auto O = plaidml::edsl::Tensor(1.0);
  return {plaidml::edsl::select(I > 0, O, Z) + plaidml::edsl::select(I < 0, -1 * O, Z)};
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

std::vector<plaidml::edsl::Tensor> sqrt(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::sqrt(A)};
}

std::vector<plaidml::edsl::Tensor> sub(const std::vector<plaidml::edsl::Value>& args) {
  const auto& A = args[0].as_tensor();
  const auto& B = args[1].as_tensor();
  return {A - B};
}

std::vector<plaidml::edsl::Tensor> sum(const std::vector<plaidml::edsl::Value>& args) {
  plaidml::edsl::Tensor result = args[0].as_tensor();
  if(args.size()==1)return {result};
  for(size_t i=1;i<args.size();i++){
    const auto& B  = args[i].as_tensor();
    result = result + B;
  }
  return {result};
}

std::vector<plaidml::edsl::Tensor> tan(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::tan(A)};
}

std::vector<plaidml::edsl::Tensor> tanh(const std::vector<plaidml::edsl::Value>& args) {
  const auto A = args[0].as_tensor();
  return {plaidml::edsl::tanh(A)};
}

std::vector<plaidml::edsl::Tensor> tile(const std::vector<plaidml::edsl::Value>& args) {
  const auto& inputs = args[0].as_tensor();
  //const auto& repeats = args[1].as_tensor();// TODO: need to convert this to vector<int>
  
  std::vector<int> reps_int;
  // for (size_t i = 1; i < repeats.rank(); i++) {
  //   reps_int.push_back(repeats[i].as_int());
  // }
  return {plaidml::op::tile(inputs, reps_int)};
}

std::vector<plaidml::edsl::Tensor> transpose(const std::vector<plaidml::edsl::Value>& args){
  const auto& A = args[0].as_tensor();
  return {plaidml::op::transpose(A)};
}


std::vector<plaidml::edsl::Tensor> _argmax(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  
    const auto& A = inputs[0].as_tensor();
    int B=1;
    auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      //printf("more attributes %d \n",num_attributes);
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++)
      {
        //printf("attribute %s", attribute->name().c_str());
        if(attribute->name() == "axis")
        {
          const auto axis = attribute->i();
          //printf("Axis value %lld\n", axis);

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
      //printf("more attributes %d \n",num_attributes);
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++)
      {
        //printf("attribute %s", attribute->name().c_str());
        if(attribute->name() == "axis")
        {
          const auto axis = attribute->i();
          //printf("Axis value %lld\n", axis);

          B = axis;
        }
      }
    }
  //TODO: needs negative ?
  return {plaidml::op::argmax(A,plaidml::edsl::make_tuple(B))};
}


// TODO: fix inputs 
std::vector<plaidml::edsl::Tensor> _concat(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  
    std::vector<plaidml::edsl::Tensor> tensors;
    for(size_t i=0;i<inputs.size();i++){
        tensors.push_back(inputs[i].as_tensor());
    }
    if (tensors.empty()) {
       throw std::runtime_error("The concatenate op requires at least one input tensor");
     }
    //throw std::runtime_error("The concatenate op copout\n");
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
  return {plaidml::op::concatenate(tensors,axis)};
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
    auto auto_group_mode = plaidml::op::AutoGroupMode::UNGROUPED;
    bool has_group_layout = false;
    int group = 1;
    std::vector<int> dilations;
    bool has_defined_dilations = false;
    std::vector<int> strides;
    bool has_defined_strides = false;
    std::vector<int> pads;
    bool has_manual_pads = false;
    std::vector<int> kernel_shape;
    if(num_attributes>0){
      auto attributes = node.attribute();
      for(auto attribute :attributes){
        if(attribute.name() == "auto_pad"){
          const auto auto_pad = attribute.s();
          if(auto_pad=="SAME_UPPER")auto_pad_mode = plaidml::op::AutoPadMode::SAME_UPPER;
          if(auto_pad=="SAME_LOWER")auto_pad_mode = plaidml::op::AutoPadMode::SAME_LOWER;
          if(auto_pad=="VALID")auto_pad_mode = plaidml::op::AutoPadMode::VALID;

        }
        if(attribute.name()=="pads"){
          // This attribute cannot be used simultaneously with 
          // auto_pad attribute. If not present, 
          // the padding defaults to 0 along start and end of each spatial axis.
          //TODO: fix this
          auto pads_ints = attribute.ints();
          for(auto pad: pads_ints){
            pads.push_back(pad);
          }
          has_manual_pads = true;
          auto_pad_mode = plaidml::op::AutoPadMode::EXPLICIT;
        }
        if(attribute.name()=="group"){
          group = attribute.i();
          auto_group_mode = plaidml::op::AutoGroupMode::EXPLICIT;
          //group_layout = 
          has_group_layout = true;
        }
        if(attribute.name()=="kernel_shape"){
          //If not present, should be inferred from input W
          //TODO: figure out if these are needed 
          auto kernel_shape_ints = attribute.ints();
          for(auto kernel_shape_int: kernel_shape_ints){
            kernel_shape.push_back(kernel_shape_int);
          }
        }
        if(attribute.name()=="dilations"){
          //if not present default is 1 along each spacial axis
          auto dilations_ints = attribute.ints();//returns repeated field 
          for(auto dilation: dilations_ints){
            dilations.push_back(dilation);
          }
          has_defined_dilations = true;
        }

        if(attribute.name()=="strides"){
          //If not present, the stride defaults is 1 along each spatial axis
          auto strides_ints = attribute.ints();
          for(auto stride: strides_ints){
            strides.push_back(stride);
          }
          has_defined_strides = true;
        }
      }
    }
    auto result =  plaidml::op::convolution(I,K)
              .input_layout(plaidml::op::TensorLayout::NCX)
              .filter_layout(plaidml::op::TensorLayout::KCX)
              .autopad_mode(auto_pad_mode)
              .groups(group)
              .autogroup_mode(auto_group_mode);

  if(has_defined_strides)result = result.strides(strides);
  if(has_defined_dilations) result = result.dilations(dilations);
  if(has_manual_pads) result = result.manual_padding(pads);
  if(has_group_layout) result = result.group_layout(plaidml::op::GroupLayout::IN_C);
  return {result};
}

std::vector<plaidml::edsl::Tensor> _cumsum(    
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  const auto& I = inputs[0].as_tensor();
  //const auto& axis = inputs[0].as_tensor();//need to turn tensor into integer
  int exclusive = 0;
  int reverse = 0;
  int int_axis = 0; ///axis tensor can be int32 or int64
  auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++){
        if(attribute->name() == "exclusive"){
         exclusive = attribute->i();
        }
        if(attribute->name() == "reverse"){
         reverse = attribute->i();
        }
      }
    }
  return {plaidml::op::cumsum(I,int_axis)};//cumsum(Tensor,int)
}

std::vector<plaidml::edsl::Tensor> _elu(    
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  const auto& I = inputs[0].as_tensor();
  float alpha = 1.0;

  auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++){
        if(attribute->name() == "alpha"){
         alpha = attribute->f();
        }
      }
    }
  return {plaidml::edsl::select(I >= 0, I, alpha * (plaidml::edsl::exp(I) - 1))};
}

std::vector<plaidml::edsl::Tensor> _flatten(    
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
    const auto X = inputs[0].as_tensor();
    int axis = 0; //TODO: add check (input tensor must have rank>=axis)

    auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      auto attributes = node.attribute();
      for(auto attribute: attributes){
        if(attribute.name() == "axis"){
         axis = attribute.i();
        }
      }
    }
    if(axis == 0) axis = (int)X.rank();
    if(axis < 0) axis = (int)X.rank() + axis;
    if(axis > (int)X.rank() || axis<0 ){
      throw std::runtime_error("{PlaidML ERROR} invalid axis attribute in flatten \n");
    }
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

std::vector<plaidml::edsl::Tensor> _hard_sigmoid(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  
  const auto I = inputs[0].as_tensor();
  float alpha = 0.2;
  float beta = 0.5;


  auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++){
        if(attribute->name() == "alpha"){
         alpha = attribute->f();
        }
        if(attribute->name()=="beta"){
          beta = attribute->f();
        }
      }
    }
  auto O = plaidml::edsl::Tensor(1.0);
  auto Z = plaidml::edsl::Tensor(0.0);

  auto result = alpha * I + beta;
  result = plaidml::edsl::select(result > O, O, result);
  result = plaidml::edsl::select(result < Z, Z, result);

  //return edsl::make_tuple(I);
  return {result};
}

std::vector<plaidml::edsl::Tensor> _log_softmax(
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
  auto O = plaidml::op::softmax(A,axis);
  return {plaidml::edsl::log(O)};
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
  return {plaidml::op::lrn(A,{static_cast<int64_t>(size)}).alpha(alpha).beta(beta).epsilon(bias).axes({1})};
}

std::vector<plaidml::edsl::Tensor> _selu(    
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  const auto& I = inputs[0].as_tensor();
  float alpha = 1.67326;
  float gamma = 1.0507;


  auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++){
        if(attribute->name() == "alpha"){
         alpha = attribute->f();
        }
        if(attribute->name()=="gamma"){
          gamma = attribute->f();
        }
      }
    }
  return {gamma * plaidml::edsl::select(I > 0, I, alpha * (plaidml::edsl::exp(I) - 1))};
}

std::vector<plaidml::edsl::Tensor> _squeeze(    
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  const auto& A = inputs[0].as_tensor();
  std::vector<int> axes;
  auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++){
        if(attribute->name() == "axes"){
         auto  at_axes = attribute->ints();
         axes.assign(at_axes.begin(),at_axes.end());
        //  for(auto at_axis = at_axes.begin();at_axis<at_axes.end();at_axis++){
        //    axes.push_back(*at_axis);
        //  }
        }
      }
    }
    //throw std::runtime_error("squeeze op throwing up" + node.op_type());
  return {plaidml::op::squeeze(A,axes)};
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
         axes.assign(at_axes.begin(),at_axes.end());
        //  for(auto at_axis = at_axes.begin();at_axis<at_axes.end();at_axis++){
        //    axes.push_back(*at_axis);
        //  }
        }
      }
    }
  return {plaidml::op::unsqueeze(A,{axes})};
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

std::vector<plaidml::edsl::Tensor> _transpose(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs){
  
    const auto& A = inputs[0].as_tensor();
    std::vector<int64_t> axes;
    auto num_attributes = node.attribute_size();
    if(num_attributes>0){
      auto attributes = node.attribute();
      for(auto attribute = attributes.begin();attribute < attributes.end();attribute++)
      {
        if(attribute->name() == "perm"){
         auto  at_axes = attribute->ints();
         axes.assign(at_axes.begin(),at_axes.end());
        }
      }
    }
  return {plaidml::op::transpose(A,plaidml::edsl::make_tuple(axes))};
}
// std::vector<plaidml::edsl::Tensor> _crop(
//     const ONNX_NAMESPACE::NodeProto& node,
//     const std::vector<plaidml::edsl::Value>& inputs){
  
//     const auto& A = inputs[0].as_tensor();
//     int axis = 1;
//     auto num_attributes = node.attribute_size();
//     if(num_attributes>0){
//       auto attributes = node.attribute();
//       for(auto attribute = attributes.begin();attribute < attributes.end();attribute++)
//       {
//         if(attribute->name() == "border")
//         {
//           const auto at_axis = attribute->i();
//           axis = at_axis;
//         }
//         if(attribute->name() == "scale")
//         {
//           const auto at_axis = attribute->i();
//           axis = at_axis;
//         }
//       }
//     }
//   return {plaidml::op::softmax(A,axis)};
// }

std::vector<plaidml::edsl::Tensor> MakePlaidMLOp(
    const ONNX_NAMESPACE::NodeProto& node,
    const std::vector<plaidml::edsl::Value>& inputs) {
    
  //check if this op is supported 
  const std::vector<plaidml::edsl::Value *> args;

  printf("Executing {PlaidML OP}: %s\n\n", node.op_type().c_str());

  auto op_it = plaidml_ep::kSupportedOps.find(node.op_type());
  auto _op_it = plaidml_ep::_kSupportedOps.find(node.op_type());
  if (op_it == plaidml_ep::kSupportedOps.end() && _op_it==plaidml_ep::_kSupportedOps.end()) {
    throw std::runtime_error("Asked to executed unsupported op " + node.op_type());
  }
  //return op_it->second(inputs);
  if(op_it != plaidml_ep::kSupportedOps.end()){
    return op_it->second(inputs);
  }
  else{
    return _op_it->second(node, inputs);
  }
  
}

}  // namespace plaidml_ep
}  // namespace onnxruntime
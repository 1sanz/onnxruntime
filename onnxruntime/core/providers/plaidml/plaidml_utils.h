
#pragma once

// #include <onnx/onnx_pb.h>
// #include <vector>
// #include <string>
// #include <functional>
// #include <map>
// #include <string>

#include "plaidml/edsl/edsl.h"

#include "core/graph/onnx_protobuf.h"


namespace onnxruntime {
namespace plaidml_ep {

//TODO: PlaidML add error type enumeration 

//-----------------------------------------------helper functions-----------------------------
plaidml::DType get_precision(ONNX_NAMESPACE::DataType onnx_type); 
//-----------------------------------------------------------------------------------------------

}  // namespace plaidml_ep
}  // namespace onnxruntime




#include "plaidml_utils.h"
#include "core/graph/model.h"
#include "plaidml_ops.h"

namespace onnxruntime {
namespace plaidml_ep {

//TODO: PlaidML PlaidMLNode should inherit nodeproto ?
class PlaidMLNode {
  public:
   explicit PlaidMLNode(const ONNX_NAMESPACE::NodeProto& node);

    //attribute getters
    bool has_attribute(const std::string& name);
    //TODO: PlaidML change in to int64_t for consistency with nodeproto
    int get_attribute(const std::string& name,int default_value);
    float get_attribute(const std::string& name,float default_value);
    std::string get_attribute(const std::string& name,std::string default_value);
    std::vector<int> get_attribute(const std::string& name);
  
  private:
   const ONNX_NAMESPACE::NodeProto& _node;

 };

class PlaidMLProgram{
public:
  explicit PlaidMLProgram(const onnxruntime::Node* fused_node);
  std::vector<plaidml::edsl::Tensor> get_program_inputs();
  std::vector<plaidml::edsl::Tensor> get_program_outputs();
  plaidml::edsl::Value get_value_from_dictionary(const std::string& name);
  bool add_value_to_dictionary(const std::string& name, plaidml::edsl::Value val);
  
  std::shared_ptr<plaidml::edsl::Program> program;
private:
  
  std::vector<plaidml::edsl::Tensor> _program_inputs;
  std::vector<plaidml::edsl::Tensor> _program_outputs;
  const onnxruntime::Node * _fused_node;
  /* fused node input dictionary contains the input tensor name and plaidml placeholder*/
   std::map<std::string, plaidml::edsl::Value> _value_dictionary;
   std::map<std::string, plaidml::edsl::Tensor> _output_dictionary;
   //PlaidMLProgram _plaidml_program;

    std::vector<int64_t> get_input_shape(const onnxruntime::NodeArg* node);
    bool add_fused_node_inputs_to_tensor_dictionary();
    std::vector<plaidml::edsl::Value> get_local_input_values(const onnxruntime::Node * node);

};

}  // namespace plaidml_ep
}  // namespace onnxruntime
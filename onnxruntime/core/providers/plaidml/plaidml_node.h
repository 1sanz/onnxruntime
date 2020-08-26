#include "plaidml_utils.h"

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
    //const std::vector<plaidml::edsl::Value>& inputs;
   //initializers


 };

}  // namespace plaidml_ep
}  // namespace onnxruntime
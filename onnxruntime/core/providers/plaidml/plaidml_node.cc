#include "plaidml_utils.h"
#include "plaidml_node.h"

namespace onnxruntime {
namespace plaidml_ep {

PlaidMLNode::PlaidMLNode(const ONNX_NAMESPACE::NodeProto& node):_node(node){

}

bool PlaidMLNode::has_attribute(const std::string& name){
    auto num_attributes = _node.attribute_size();
    if(num_attributes>0){
        auto attributes = _node.attribute();
        for(auto attribute: attributes){
            if(attribute.name() == name){
                return true;
            }
        }
    }
    return false;
}

int PlaidMLNode::get_attribute(const std::string& name,int default_value){
     auto num_attributes = _node.attribute_size();
      if(num_attributes>0){
        auto attributes = _node.attribute();
        for(auto attribute: attributes){
            if(attribute.name() == name){
                return attribute.i();
            }
        }
      }
    return default_value;
}

float PlaidMLNode::get_attribute(const std::string& name,float default_value){
     auto num_attributes = _node.attribute_size();
      if(num_attributes>0){
        auto attributes = _node.attribute();
        for(auto attribute: attributes){
            if(attribute.name() == name){
                return attribute.f();
            }
        }
      }
    return default_value;
}

std::string PlaidMLNode::get_attribute(const std::string& name,std::string default_value){
     auto num_attributes = _node.attribute_size();
      if(num_attributes>0){
        auto attributes = _node.attribute();
        for(auto attribute: attributes){
            if(attribute.name() == name){
                return attribute.s();
            }
        }
      }
    return default_value;
}

std::vector<int> PlaidMLNode::get_attribute(const std::string& name){
     auto num_attributes = _node.attribute_size();
     std::vector<int> ret_val;
      if(num_attributes>0){
        auto attributes = _node.attribute();
        for(auto attribute: attributes){
            if(attribute.name() == name){
               auto att_ints = attribute.ints();
               ret_val.assign(att_ints.begin(),att_ints.end());
               return ret_val;
            }
        }
      }
    return ret_val;
}

}  // namespace plaidml_ep
}  // namespace onnxruntime
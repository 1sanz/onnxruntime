

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [4])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [4])
Z = helper.make_tensor_value_info('Z', TensorProto.BOOL, [4])
axis = helper.make_tensor_value_info('axis',TensorProto.INT32,[1])
min_val = helper.make_tensor_value_info('min_val',TensorProto.FLOAT,[1])
max_val = helper.make_tensor_value_info('max_val',TensorProto.FLOAT,[1])
#node_add = helper.make_node('Mul',['A','B'],['Z'],)

node_exp = onnx.helper.make_node(
    'Exp',
    inputs=['A'],
    outputs=['B'],
)

graph_exp_graph = helper.make_graph(
    [node_exp],
    "exp-graph",
    [A],
    [B],
)

model_exp = helper.make_model(graph_exp_graph,
                              producer_name='onnx-exp-example')
onnx.save(model_exp,'./onnxruntime/onnxruntime/test/testdata/plaidml/exp.onnx')
model_in = onnx.load('./onnxruntime/onnxruntime/test/testdata/plaidml/exp.onnx')
print('The graph in model:\n{}'.format(model_in.graph))

onnx.checker.check_model(model_in)
print('The model is checked!')


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


# Create one input (ValueInfoProto)
# X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])

# # Create second input (ValueInfoProto)
# Pads = helper.make_tensor_value_info('Pads', TensorProto.INT64, [4])



# # Create one output (ValueInfoProto)
# Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])


# A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [4])
# B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [4])
# Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [4])

# X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [4])
# Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4])
#node_add = helper.make_node('Mul',['A','B'],['Z'],)

# node_abs = onnx.helper.make_node(
#     'Abs',
#     inputs=['X'],
#     outputs=['Y'],
#     mode='constant', # Attributes
# )
# graph_abs_graph = helper.make_graph(
#     [node_abs],
#     "abs-graph",
#     [X],
#     [Y],
# )

# model_abs = helper.make_model(graph_abs_graph,
#                               producer_name='onnx-abs-example')
# onnx.save(model_abs,'./abs.onnx')
# model_in = onnx.load('./abs.onnx')
# print('The graph in model:\n{}'.format(model_in.graph))
# # Create a node (NodeProto)
# node_def = helper.make_node(
#     'Pad', # node name
#     ['X', 'Pads'], # inputs
#     ['Y'], # outputs
#     mode='constant', # Attributes
# )

# # Create the graph (GraphProto)
# graph_def = helper.make_graph(
#     [node_def],
#     "test-model",
#     [X, Pads],
#     [Y],
#     [helper.make_tensor('Pads', TensorProto.INT64, [4,], [0, 0, 1, 1,])],
# )

# # Create the model (ModelProto)
# model_def = helper.make_model(graph_def,
#                               producer_name='onnx-example')

# print('The ir_version in model: {}\n'.format(model_def.ir_version))
# print('The producer_name in model: {}\n'.format(model_def.producer_name))
# print('The graph in model:\n{}'.format(model_def.graph))
# onnx.checker.check_model(model_def)
# print('The model is checked!')



# Convolution with padding

x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1,1,5,5])
w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1,1,3,3])
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1,1,5,5])
node_with_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'w'],
    outputs=['y'],
    kernel_shape=[3, 3],
    # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
    pads=[1, 1, 1, 1],
)
graph_conv_graph = helper.make_graph(
    [node_with_padding],
    "conv-graph",
    [x,w],
    [y],
)

model_conv = helper.make_model(graph_conv_graph,
                              producer_name='onnx-conv-example')
onnx.save(model_conv,'./onnxruntime/onnxruntime/test/testdata/plaidml/conv.onnx')
model_in = onnx.load('./onnxruntime/onnxruntime/test/testdata/plaidml/conv.onnx')
print('The graph in model:\n{}'.format(model_in.graph))
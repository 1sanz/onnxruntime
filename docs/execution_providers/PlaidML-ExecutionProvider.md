# PlaidML Execution Provider

PlaidML Execution Provider uses a pre-release PlaidML version (plaidml-v1) to generate MLIR code for onnxruntime programs. PlaidML version 1 is scheduled to release soon. We plan to support Intel CPUs, Intel integrated GPUs and Intel<sup>®</sup> Movidius<sup>TM</sup> Vision Processing Units (VPUs). Please refer to [this](https://software.intel.com/en-us/openvino-toolkit/hardware) page for details on the Intel hardware. For details on MLIR see https://mlir.llvm.org


## Build instructions for ONNXRT with PlaidML (plaidml-v1)

PlaidML version 1 pre-release can be downloaded and build from source see: https://github.com/plaidml/plaidml

Build the PlaidML library to use with onnxruntime
```
git clone --recursive  --single-branch --plaidml-v1  https://github.com/plaidml/plaidml.git
cd plaidml
./configure
conda activate .cenv/
bazelisk test //...
bazelisk build plaidml:shlib
```

The dylib will be generated inside bazel-bin/plaidml/

downlaod onnxruntime plaidml branch and run the following commands to build the onnxruntime wheel with plaidml Execution Provider enabled.  
```
export TODO_TEMP_PLAIDML_DIR=/path/to/plaidml
export TODO_TEMP_PLAIDML_LIB_DIR=~/path/to/plaidml.dylib
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_plaidml
./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_plaidml --build_wheel
```

## TILE eDSL (C++ / Python)
PlaidML provides the TILE eDSL (Embedded Domain Specific Language) as a simple way to translate mathematical logic the code. The eDSL is designed to make machine learning code easier to write and read. For learning more about TILE eDSL see: https://plaidml.readthedocs.io/en/latest/usage/edsl.html#how-to-write-tile-code

## ONNX Layers supported using PlaidML

The table below shows the ONNX layers created using TILE eDSL wrapper functions that translate onnx nodes into eDSL and generate MLIR code for the specified device, to provide the most efficient ewxecution possible. Click (TODO: add link to plaidml exec provider ops in onnxruntime) to see TILE eDSL in action under onnxruntime. PlaidML also provides an op library which has been built on top of TILE eDSL to facilitate developers, this library provides access to eDSL implementations of commonly used ML operations with an easy to use fluent API. The table below lists supported layers and corresponding eDSL op library functions employed. The table also lists the Intel hardware support for each of the layers. CPU refers to Intel<sup>®</sup>
Atom, Core, and Xeon processors. GPU refers to the Intel Integrated Graphics. VPU refers to USB based Intel<sup>®</sup> Movidius<sup>TM</sup>
VPUs as well as Intel<sup>®</sup> Vision accelerator Design with Intel Movidius <sup>TM</sup> MyriadX VPU. FPGA refers to Intel<sup>®</sup> Vision Accelerator Design with an Intel<sup>®</sup> Arria<sup>®</sup> 10 FPGA.

| **ONNX Layers** | **eDSL Layers** | **CPU** | **GPU** | **VPU** | **FPGA** |
| --- | --- | --- | --- | --- | --- |
| Abs | abs | Yes | No | No | No
| Acos | acos | Yes | No | No | No
| Acosh | acosh | Yes | No | No | No
| Add | + | Yes | No | No | No
| And | - | Yes | No | No | No
| ArgMax | argmax | Yes | No | No | No
| ArgMin | - | Yes | No | No | No
| Asin | asin | Yes | No | No | No
| Asinh | - | No | No | No | No
| Atan | atan | Yes | No | No | No
| Atanh | - | No | No | No | No
| AveragePool | - | No | No | No | No
| BatchNormalization | - | No | No | No | No
| BitShift | - | No | No | No | No
| Cast | - | No | No | No | No
| Ceil | - | No | No | No | No
| Clip | - | No | No | No | No
| Compress | - | No | No | No | No
| Concat | - | No | No | No | No
| ConcatFromSquence | - | No | No | No | No
| Constant | - | No | No | No | No
| ConstantOfShape | - | No | No | No | No
| Conv | convolution | Yes | No | No | No
| ConvInteger | - | No | No | No | No
| ConvTranspose | - | No | No | No | No
| Cos | - | No | No | No | No
| Cosh | - | No | No | No | No
| Cumsum | - | No | No | No | No
| DepthToSpace | - | No | No | No | No
| DequantizeLinear | - | No | No | No | No
| Det | - | No | No | No | No
| Div | - | No | No | No | No
| Dropout | - | No | No | No | No
| Einsum | - | No | No | No | No
| Elu | - | No | No | No | No
| Equal | - | No | No | No | No
| Erf | - | No | No | No | No
| Exp | - | No | No | No | No
| Expand | - | No | No | No | No
| EyeLike | - | No | No | No | No
| Flatten | - | No | No | No | No
| Floor | - | No | No | No | No
| GRU | - | No | No | No | No
| Gather | - | No | No | No | No
| GatherElements | - | No | No | No | No
| GatherND | - | No | No | No | No
| Gemm | - | No | No | No | No
| GlobalAveragePool | - | No | No | No | No
| GlobalLpPool | - | No | No | No | No
| GlobalMaxPool | - | No | No | No | No
| Greater | - | No | No | No | No
| HardSigmoid | - | No | No | No | No
| HardMax | - | No | No | No | No
| Identity | - | No | No | No | No
| If | - | No | No | No | No
| InstantNormalization | - | No | No | No | No
| IsInf | - | No | No | No | No
| IsNaN | - | No | No | No | No
| LRN | - | No | No | No | No
| LeakyRelu | - | No | No | No | No
| Less | - | No | No | No | No
| Log | - | No | No | No | No
| LogSoftmax | - | No | No | No | No
| Loop | - | No | No | No | No
| LpNormalization | - | No | No | No | No
| LpPool | - | No | No | No | No
| MatMul | - | No | No | No | No
| MatMulInteger | - | No | No | No | No
| Max | - | No | No | No | No
| MaxPool | - | No | No | No | No
| MaxRoiPool | - | No | No | No | No
| MaxUnpool | - | No | No | No | No
| Mean | - | No | No | No | No
| Min | - | No | No | No | No
| Mod | - | No | No | No | No
| Mul | - | No | No | No | No
| Multinomial | - | No | No | No | No
| Neg | - | No | No | No | No
| NonMaxSuppression | - | No | No | No | No
| NonZero | - | No | No | No | No
| Not | - | No | No | No | No
| OneHot | - | No | No | No | No
| Or | - | No | No | No | No
| PRelu | - | No | No | No | No
| Pad | - | No | No | No | No
| Pow | - | No | No | No | No
| QLinearConv | - | No | No | No | No
| QLinearMatmul | - | No | No | No | No
| QuantizeLinear | - | No | No | No | No
| RNN | - | No | No | No | No
| RandomNormal | - | No | No | No | No
| RandomNormalLike | - | No | No | No | No
| RandomUniform | - | No | No | No | No
| RandomUniformLike | - | No | No | No | No
| Reciprocal | - | No | No | No | No
| ReduceL1 | - | No | No | No | No
| ReduceL2 | - | No | No | No | No
| ReduceLogSum | - | No | No | No | No
| ReduceLogSumExp | - | No | No | No | No
| ReduceMax | - | No | No | No | No
| ReduceMean | - | No | No | No | No
| ReduceMin | - | No | No | No | No
| ReduceProd | - | No | No | No | No
| ReduceSum | - | No | No | No | No
| ReduceSumSquare | - | No | No | No | No
| Relu | - | No | No | No | No
| Reshape | - | No | No | No | No
| Resize | - | No | No | No | No
| ReverseSequence | - | No | No | No | No
| RoiAlign | - | No | No | No | No
| Round | - | No | No | No | No
| Scan | - | No | No | No | No
| Scatter | - | No | No | No | No
| ScatterElements | - | No | No | No | No
| ScatterND | - | No | No | No | No
| Selu | - | No | No | No | No
| SequenceAt | - | No | No | No | No
| SequenceConstruct | - | No | No | No | No
| SequenceEmpty | - | No | No | No | No
| SequenceErase | - | No | No | No | No
| SequenceInsert | - | No | No | No | No
| SequenceLength | - | No | No | No | No
| Shape | - | No | No | No | No
| Shrink | - | No | No | No | No
| Sigmoid | - | No | No | No | No
| Sign | - | No | No | No | No
| Sin | - | No | No | No | No
| Sinh | - | No | No | No | No
| Size | - | No | No | No | No
| Slice | - | No | No | No | No
| Softmax | - | No | No | No | No
| SoftPlus | - | No | No | No | No
| SoftSign | - | No | No | No | No
| SpaceToDepth | - | No | No | No | No
| Split | - | No | No | No | No
| SplitToSequence | - | No | No | No | No
| Sqrt | - | No | No | No | No
| Squeeze | - | No | No | No | No
| StringNormalizer | - | No | No | No | No
| Sub | - | No | No | No | No
| Sum | - | No | No | No | No
| Tan | - | No | No | No | No
| Tanh | - | No | No | No | No
| TfdfVectorizer | - | No | No | No | No
| ThresholdRelu | - | No | No | No | No
| Tile | - | No | No | No | No
| TopK | - | No | No | No | No
| Transpose | - | No | No | No | No
| Unique | - | No | No | No | No
| UnSqueeze | - | No | No | No | No
| UpSample | - | No | No | No | No
| Where | - | No | No | No | No
| Xor | - | No | No | No | No

## Topology Support

TODO: add topology support here 

## How to run the ONNX node/model tests 

TODO: add instructions on running onnx_

## Sample Inference Resnet50

TODO:
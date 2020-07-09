# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# compile ASM with /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
# compile CXX with /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
ASM_FLAGS = -O2 -g -DNDEBUG -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.15.sdk -fPIC  

ASM_DEFINES = -DEIGEN_MPL2_ONLY -DNSYNC_ATOMIC_CPP11 -DPLATFORM_POSIX -DUSE_EIGEN_FOR_BLAS -DUSE_PLAIDML=1

ASM_INCLUDES = -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/include/onnxruntime -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/include/onnxruntime/core/session -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/inc -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/amd64 -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/cmake/external/eigen -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/cmake/external/nsync/public -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/cmake -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime 

CXX_FLAGS =  -fstack-protector-strong -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -Wno-deprecated -Wall -Wextra -ffunction-sections -fdata-sections -Werror -Wno-parentheses -O2 -g -DNDEBUG -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.15.sdk -fPIC   -std=gnu++14

CXX_DEFINES = -DEIGEN_MPL2_ONLY -DNSYNC_ATOMIC_CPP11 -DPLATFORM_POSIX -DUSE_EIGEN_FOR_BLAS -DUSE_PLAIDML=1

CXX_INCLUDES = -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/include/onnxruntime -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/include/onnxruntime/core/session -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/inc -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/amd64 -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/cmake/external/eigen -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/cmake/external/nsync/public -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/cmake -I/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime 

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/DgemmKernelSse2.S.o_FLAGS = -msse2

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SgemmKernelSse2.S.o_FLAGS = -msse2

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SgemmTransposePackB16x4Sse2.S.o_FLAGS = -msse2

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SconvKernelSse2.S.o_FLAGS = -msse2

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SpoolKernelSse2.S.o_FLAGS = -msse2

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/DgemmKernelAvx.S.o_FLAGS = -mavx

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SgemmKernelAvx.S.o_FLAGS = -mavx

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SgemmKernelM1Avx.S.o_FLAGS = -mavx

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SgemmKernelM1TransposeBAvx.S.o_FLAGS = -mavx

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SgemmTransposePackB16x4Avx.S.o_FLAGS = -mavx

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SconvKernelAvx.S.o_FLAGS = -mavx

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SpoolKernelAvx.S.o_FLAGS = -mavx

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/QgemmU8S8KernelAvx2.S.o_FLAGS = -mavx2 -mfma

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/QgemvU8S8KernelAvx2.S.o_FLAGS = -mavx2 -mfma

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/QgemmU8U8KernelAvx2.S.o_FLAGS = -mavx2 -mfma

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/DgemmKernelFma3.S.o_FLAGS = -mavx2 -mfma

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SgemmKernelFma3.S.o_FLAGS = -mavx2 -mfma

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SconvKernelFma3.S.o_FLAGS = -mavx2 -mfma

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/LogisticKernelFma3.S.o_FLAGS = -mavx2 -mfma

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/TanhKernelFma3.S.o_FLAGS = -mavx2 -mfma

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/ErfKernelFma3.S.o_FLAGS = -mavx2 -mfma

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/DgemmKernelAvx512F.S.o_FLAGS = -mavx512f

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SgemmKernelAvx512F.S.o_FLAGS = -mavx512f

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SconvKernelAvx512F.S.o_FLAGS = -mavx512f

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/SpoolKernelAvx512F.S.o_FLAGS = -mavx512f

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/QgemmU8S8KernelAvx512Core.S.o_FLAGS = -mavx512bw -mavx512dq -mavx512vl

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/QgemvU8S8KernelAvx512Core.S.o_FLAGS = -mavx512bw -mavx512dq -mavx512vl

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/QgemmU8S8KernelAvx512Vnni.S.o_FLAGS = -mavx512bw -mavx512dq -mavx512vl

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/QgemvU8S8KernelAvx512Vnni.S.o_FLAGS = -mavx512bw -mavx512dq -mavx512vl

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/QgemmU8U8KernelAvx512Core.S.o_FLAGS = -mavx512bw -mavx512dq -mavx512vl

# Custom flags: CMakeFiles/onnxruntime_mlas.dir/Users/snazir/Documents/PlaidML/plaid_onnx/onnxruntime/onnxruntime/core/mlas/lib/x86_64/QgemmU8U8KernelAvx512Vnni.S.o_FLAGS = -mavx512bw -mavx512dq -mavx512vl


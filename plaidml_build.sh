#!/bin/bash

# This is a temporary file created for creating a Gitlab pipeline
# This file will be deleted once this branch is finalized 

# This file contains commands that build onnxruntime with plaidml execution 
# provider support and runs internal node and inference tests 

# download plaidml-v1 from source and build it 
# TODO: PlaidML - this should be done in .gitmodules and the build instructions 
# added to onnxruntime OR post plaidml-v1 release the user will be instructed to
# install plaidml-v1 through pip    

#timestamp function 
START=$(date +%s)

#DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"` #add %3N as we want millisecond too
print timestamp
git clone --recursive --branch plaidml-v1 https://github.com/plaidml/plaidml.git ./build/plaidml
cd build/plaidml/
./configure
conda activate .cenv/
bazelisk build plaidml:shlib
conda deactivate

export TODO_TEMP_PLAIDML_DIR=$PWD
export TODO_TEMP_PLAIDML_LIB_DIR=$PWD/bazel-bin/plaidml/

cd ../../

./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_plaidml

#getting the time 
END=$(date +%s)
DIFF=$(( $END - $START ))
echo "-----------------------------------"
echo "This build took $DIFF seconds"
echo "-----------------------------------"
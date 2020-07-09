// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(ContribOpTest, ExpandDims_0) {
  OpTester test("ExpandDims", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("X", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int32_t>("axis", {}, {-1});
  test.AddOutput<float>("Y", {2, 3, 1}, std::vector<float>(6, 1.0f));
  std::unordered_set<std::string> excluded_providers;
  //Plaidml removed from tests for now 
   excluded_providers.insert(kPlaidMLExecutionProvider);
   test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
}

TEST(ContribOpTest, ExpandDims_1) {
  OpTester test("ExpandDims", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("X", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int32_t>("axis", {}, {1});
  test.AddOutput<float>("Y", {2, 1, 3}, std::vector<float>(6, 1.0f));
  //TODO: Plaidml-TODO need to add some ops to get this to work tanh is missing 
  std::unordered_set<std::string> excluded_providers;
  //Plaidml removed from tests for now 
   excluded_providers.insert(kPlaidMLExecutionProvider);
   test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
}

}  // namespace test
}  // namespace onnxruntime

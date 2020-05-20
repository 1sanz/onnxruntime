// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include "core/providers/plaidml/plaidml_execution_provider.h"
#include "test/providers/provider_test_utils.h"
#include "default_providers.h"
#include "gtest/gtest.h"
#include "core/session/inference_session.h"
#include "test/framework/test_utils.h"

namespace onnxruntime {
namespace test {

// TODO: This is based on the nGraph tests, investigate if we want something different
void RunTest(
    const std::string& model_path,
    const NameMLValMap& feeds,
    const std::vector<std::string>& output_names,
    const std::vector<std::vector<int64_t>>& expected_shapes,
    const std::vector<std::vector<float>>& expected_values,
    const Environment& env) {
  SessionOptions so;
  InferenceSession session_object(so, env);

  auto status = session_object.RegisterExecutionProvider(DefaultPlaidMLExecutionProvider());
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = session_object.Load(model_path);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options{};
  run_options.run_tag = "PlaidML EP test tag";
  run_options.run_log_verbosity_level = 1;

  std::vector<OrtValue> fetches;
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Run failed with status: " << status.ErrorMessage();
    return;
  }

  for (size_t idx = 0; idx < expected_values.size(); ++idx) {
    auto& got_tensor = fetches[idx].Get<Tensor>();
    auto* got = got_tensor.Data<float>();
    auto& expected = expected_values[idx];
    TensorShape expected_shape(expected_shapes[idx]);
    EXPECT_EQ(got_tensor.Shape(), expected_shape);
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_EQ(got[i], expected[i]);
    }
  }
}

// TODO: This is separated out as in nGraph, but I think we may want to make it integrated
void add_feeds(NameMLValMap& feeds, std::string name, std::vector<int64_t> dims, std::vector<float> value) {
  OrtValue ml_value;

  auto allocator = TestPlaidMLExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
  ASSERT_TRUE(allocator) << "Attempting to get allocator for input tensors yielded null pointer";
  CreateMLValue<float>(allocator, dims, value, &ml_value);

  feeds.insert(std::make_pair(name, ml_value));
}

TEST(PlaidMLExecutionProviderTest, Basic_Test) {
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {4.0f, 8.0f, 12.0f, 16.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  // TODO: We're borrowing nGraph test data just to get something basic going, but switch to our own network & inputs eventually
  RunTest("testdata/ngraph/Basic_Test.onnx", feeds, {"Z"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, AbsTest)
{
  NameMLValMap feeds;
  //TODO: remove hardcoded values add expected value computation 
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});

  std::vector<std::vector<float>> expected_values = {
      {1.0f, 2.0f, 3.0f, 4.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/abs.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, AddTest)
{
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {3.0f, 4.0f, 5.0f, 6.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/add.onnx", feeds, {"Z"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, CeilTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {-4.5f, -2.2f, 3.7f, 4.1f});

  std::vector<std::vector<float>> expected_values = {
      {-4.0f, -2.0f, 4.0f, 5.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/ceil.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, ClipTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {-4.0f, -2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "min_val", {1}, {-1.0f});
  add_feeds(feeds, "max_val", {1}, {1.0f});
  std::vector<std::vector<float>> expected_values = {
      {-1.0f, -1.0f, 1.0f, 1.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/clip.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

// TODO: fix -> use keras bridge conv style
// TEST(PlaidMLExecutionProviderTest, ConvTest)
// {
//   NameMLValMap feeds;

//   add_feeds(feeds, "I", {1,1,5,5}, {0., 1., 2., 3., 4.,5.,
//                          6., 7., 8., 9.,10., 
//                          11., 12., 13., 14.,15., 
//                          16., 17., 18., 19.,20., 
//                          21., 22., 23., 24.});

//     add_feeds(feeds, "K", {1,1,5,5}, {1., 1., 1.,
//                                       1., 1., 1.,
//                                       1., 1., 1.});


//   std::vector<std::vector<float>> expected_values = {
//       {12., 21., 27., 33., 24.,
//        33., 54., 63., 72., 51.,
//        63., 99., 108., 117., 81.,
//        93., 144., 153., 162., 111.,
//        72., 111., 117., 123., 84.}};

//   std::vector<std::vector<int64_t>> expected_shapes = {
//       {4}};

//   RunTest("testdata/plaidml/conv.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
// }

// TEST(PlaidMLExecutionProviderTest, TanTest)
// {
//   NameMLValMap feeds;

//   add_feeds(feeds, "A", {4}, {0.0f, 90.0f, 180.0f, 270.0f});

//   std::vector<std::vector<float>> expected_values = {
//       {1.0f, 0.0f, 0.0f, 5.0f}};

//   std::vector<std::vector<int64_t>> expected_shapes = {
//       {4}};

//   RunTest("testdata/plaidml/tan.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
// }

TEST(PlaidMLExecutionProviderTest, CosTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {-1.0f, 0.0f, 1.0f, 3.141592653589793f});

  std::vector<std::vector<float>> expected_values = {
      {0.5403023, 1.0, 0.5403023,-1.0}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/cos.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

// TEST(PlaidMLExecutionProviderTest, CoshTest)
// {
//   NameMLValMap feeds;

//   add_feeds(feeds, "A", {4}, {-4.5f, -2.2f, 3.7f, 4.1f});

//   std::vector<std::vector<float>> expected_values = {
//       {-4.0f, -2.0f, 4.0f, 5.0f}};

//   std::vector<std::vector<int64_t>> expected_shapes = {
//       {4}};

//   RunTest("testdata/plaidml/cosh.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
// }

// TODO:  invalid model. Error in Node: : No Op registered for Cumsum with domain_version of 11
// TEST(PlaidMLExecutionProviderTest, CumsumTest)
// {
//   NameMLValMap feeds;

//   add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});

//   std::vector<std::vector<float>> expected_values = {
//       {2.0f, 4.0f, 6.0f, 8.0f}};

//   std::vector<std::vector<int64_t>> expected_shapes = {
//       {4}};

//   RunTest("testdata/plaidml/cumsum.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
// }

TEST(PlaidMLExecutionProviderTest, DivTest)
{
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {4.0f, 8.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 3.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {2.0f, 4.0f, 1.0f, 2.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/div.onnx", feeds, {"Z"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, ExpTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {-1.5255321 ,  0.49402064, 0.0, 2.3025851});

  std::vector<std::vector<float>> expected_values = {
      {0.21750529,  1.6388924 ,  1.0,  10.0}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/exp.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

// TODO: need to handle bool tensor type -> add a template 
// TEST(PlaidMLExecutionProviderTest, LessTest)
// {
//   NameMLValMap feeds;
//   add_feeds(feeds, "A", {4}, {4.0f, 8.0f, 3.0f, 4.0f});
//   add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 3.0f, 2.0f});

//   std::vector<std::vector<float>> expected_values = {
//       {1.0, 1.0, 1.0, 1.0}};

//   std::vector<std::vector<int64_t>> expected_shapes = {
//       {4}};

//   RunTest("testdata/plaidml/less.onnx", feeds, {"Z"}, expected_shapes, expected_values, GetEnvironment());
// }

TEST(PlaidMLExecutionProviderTest, LogTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {0.21750529,  1.6388924 ,  1.0,  10.0});

  std::vector<std::vector<float>> expected_values = {
      {-1.5255321 ,  0.49402064, 0.0, 2.3025851}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/log.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, MulTest)
{
  NameMLValMap feeds;
  // TODO: fix dimension mismatch issue
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {2.0f, 4.0f, 6.0f, 8.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/mul.onnx", feeds, {"Z"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, ReluTest)
{
  NameMLValMap feeds;
  //TODO: remove hardcoded values add expected value computation 
  add_feeds(feeds, "X", {3,4,5}, {0.9105323 , -0.17202467,  0.19540711,  0.5057067 , 0.05839625,
                                  0.4989552 , -0.24061275,  0.31148124, -0.25042275, -1.1431444, 
                                  0.6542162 , -1.7285362 ,  0.88606644,  1.6676011 ,-1.3186206, 
                                  -2.2778082 , -0.77277696, -0.8941763 , -0.71552706, -1.1515434,
                                  0.12072674, -1.3711382 ,  0.37264553,  1.2529551 ,-0.33971852,
                                  -0.7263648 ,  0.9757802 , -0.51076573, -1.2931641 ,0.06053324,
                                  -0.05999891,  0.6783777 , -0.88862735, -0.12444367, -0.71822774,
                                  -0.17955114, -0.18401518,  1.7149601 ,  0.99298984, -0.560877,
                                  -0.04585629, -0.48615912,  1.1301773 ,  0.93092895, 0.42474958,
                                  -1.560841  , -0.13032605, -0.12858576,  0.6732823 ,0.27759868,
                                  0.05039625,  0.1503488 , -0.07602423,  0.24816841, 0.7897034,
                                  -0.5707704 ,  2.068636  ,  1.2006909 , -0.7577646 ,-1.5209451});

  std::vector<std::vector<float>> expected_values = {
      {0.9105323 , 0.        , 0.19540711, 0.5057067 , 0.05839625,
      0.4989552 , 0.        , 0.31148124, 0.        , 0.,
      0.6542162 , 0.        , 0.88606644, 1.6676011 , 0.,
      0.        , 0.        , 0.        , 0.        , 0.,
      0.12072674, 0.        , 0.37264553, 1.2529551 , 0.,
      0.        , 0.9757802 , 0.        , 0.        , 0.06053324,
      0.        , 0.6783777 , 0.        , 0.        , 0.,
      0.        , 0.        , 1.7149601 , 0.99298984, 0.,
      0.        , 0.        , 1.1301773 , 0.93092895, 0.42474958,
      0.        , 0.        , 0.        , 0.6732823 , 0.27759868,
      0.05039625, 0.1503488 , 0.        , 0.24816841, 0.7897034 ,
      0.        , 2.068636  , 1.2006909 , 0.        , 0.}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {3,4,5}};

  RunTest("testdata/plaidml/relu.onnx", feeds, {"Y"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, SigmoidTest)
{
  NameMLValMap feeds;
  //TODO: remove hardcoded values add expected value computation 
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});

  std::vector<std::vector<float>> expected_values = {
      {0.7310586 , 0.880797  , 0.95257413, 0.98201376 }};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/sigmoid.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

// TEST(PlaidMLExecutionProviderTest, SinTest)
// {
//   NameMLValMap feeds;

//   add_feeds(feeds, "A", {4}, {-4.5f, -2.2f, 3.7f, 4.1f});

//   std::vector<std::vector<float>> expected_values = {
//       {-4.0f, -2.0f, 4.0f, 5.0f}};

//   std::vector<std::vector<int64_t>> expected_shapes = {
//       {4}};

//   RunTest("testdata/plaidml/sin.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
// }

// TODO: fix in plaidml required  (conversion to the LLVM IR dialect failed)
// TEST(PlaidMLExecutionProviderTest, SinhTest)
// {
//   NameMLValMap feeds;

//   add_feeds(feeds, "A", {4}, {-4.5f, -2.2f, 3.7f, 4.1f});

//   std::vector<std::vector<float>> expected_values = {
//       {-4.0f, -2.0f, 4.0f, 5.0f}};

//   std::vector<std::vector<int64_t>> expected_shapes = {
//       {4}};

//   RunTest("testdata/plaidml/sinh.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
// }

// TODO: fix ShapeInferenceError
// TEST(PlaidMLExecutionProviderTest, SoftmaxTest)
// {
//   NameMLValMap feeds;

//   add_feeds(feeds, "A", {4}, {2.0f, 2.0f, 3.0f, 4.0f});

//   std::vector<std::vector<float>> expected_values = {
//       {0.7310586 , 0.880797  , 0.95257413, 0.98201376 }};

//   std::vector<std::vector<int64_t>> expected_shapes = {
//       {4}};

//    RunTest("testdata/plaidml/softmax.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
// }

TEST(PlaidMLExecutionProviderTest, SqrtTest)
{
  NameMLValMap feeds;

  add_feeds(feeds, "A", {4}, {4.0f, 9.0f, 4.0f, 16.0f});

  std::vector<std::vector<float>> expected_values = {
      {2.0, 3.0, 2.0,4.0}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/sqrt.onnx", feeds, {"B"}, expected_shapes, expected_values, GetEnvironment());
}

TEST(PlaidMLExecutionProviderTest, SubTest)
{
  NameMLValMap feeds;
  add_feeds(feeds, "A", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  add_feeds(feeds, "B", {4}, {2.0f, 2.0f, 2.0f, 2.0f});

  std::vector<std::vector<float>> expected_values = {
      {-1.0f, 0.0f, 1.0f, 2.0f}};

  std::vector<std::vector<int64_t>> expected_shapes = {
      {4}};

  RunTest("testdata/plaidml/sub.onnx", feeds, {"Z"}, expected_shapes, expected_values, GetEnvironment());
}
}  // namespace test
}  // namespace onnxruntime

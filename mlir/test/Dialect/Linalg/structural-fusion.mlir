// RUN: mlir-opt %s -split-input-file -linalg-unfuse '-linalg-structural-fusion=strategy=seed(conv),producers(bcast),consumers(act),consumers(pad),producers(pad),consumers(pool)' \
// RUN: | FileCheck %s

// Verify that the expected operations are put into the fused operation
// CHECK-LABEL: func @forward(
// CHECK: %[[ARG0:[a-z0-9_]+]]: tensor<1x3x416x416xf32>)
// CHECK: %[[CST0:[a-z0-9_]+]] = arith.constant dense<1.000000e+00> : tensor<16xf32>
// CHECK: %[[CST1:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST2:[a-z0-9_]+]] = arith.constant dense<1.000000e+00> : tensor<16x3x3x3xf32>
// CHECK: %[[INIT_TENSOR0:[a-z0-9_]+]] = linalg.init_tensor [2, 2] : tensor<2x2xf32>
// CHECK: %[[INIT_TENSOR1:[a-z0-9_]+]] = linalg.init_tensor [1, 16, 416, 416] : tensor<1x16x416x416xf32>
// CHECK: %[[INIT_TENSOR2:[a-z0-9_]+]] = linalg.init_tensor [1, 16, 208, 208] : tensor<1x16x208x208xf32>
// CHECK: %[[FUSED:[a-z0-9_]+]] = linalg.fused(%[[ARG1:[a-z0-9_]+]] = %[[CST2]] {{.*}}, %[[ARG2:[a-z0-9_]+]] = %[[CST0]] {{.*}}, %[[ARG3:[a-z0-9_]+]] = %[[INIT_TENSOR1]] {{.*}}, %[[ARG4:[a-z0-9_]+]] = %[[CST1]] {{.*}}, %[[ARG5:[a-z0-9_]+]] = %[[ARG0]] {{.*}}, %[[ARG6:[a-z0-9_]+]] = %[[INIT_TENSOR0]] {{.*}}, %[[ARG7:[a-z0-9_]+]] = %[[INIT_TENSOR2]]
// CHECK: %[[PAD:[a-z0-9_]+]] = tensor.pad %[[ARG5]] low[0, 0, 1, 1] high[0, 0, 1, 1]
// CHECK:   tensor.yield %[[ARG4]]
// CHECK: %[[BROAD:[a-z0-9_]+]] = linalg.broadcast_bias_2d_fchw ins(%[[ARG2]]{{.*}}) outs(%[[ARG3]]{{.*}})
// CHECK: %[[CONV:[a-z0-9_]+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[PAD]], %[[ARG1]]{{.*}}) outs(%[[BROAD]]{{.*}})
// CHECK: %[[LRELU:[a-z0-9_]+]] = linalg.lrelu_2d_nchw ins(%[[CONV]], %[[ARG4]]{{.*}}) outs(%[[CONV]]{{.*}})
// CHECK: %[[MAXPOOL:[a-z0-9_]+]] = linalg.pooling_nchw_max {{.*}} ins(%[[LRELU]], %[[ARG6]]{{.*}}) outs(%[[ARG7]]{{.*}})
// CHECK: linalg.yield %[[MAXPOOL]]
// CHECK: return %[[FUSED]]

#map0 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "TinyYoloV2"} {
  func @forward(%arg0: tensor<1x3x416x416xf32>) -> tensor<1x16x208x208xf32> {
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<16xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<16x3x3x3xf32>
    %0 = linalg.init_tensor [2, 2] : tensor<2x2xf32>
    %1 = linalg.init_tensor [1, 16, 416, 416] : tensor<1x16x416x416xf32>
    %42 = linalg.init_tensor [1, 16, 208, 208] : tensor<1x16x208x208xf32>
    %2 = linalg.broadcast_bias_2d_fchw ins(%cst_0 : tensor<16xf32>) outs(%1 : tensor<1x16x416x416xf32>) -> tensor<1x16x416x416xf32>
    %3 = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
        ^bb0(%arg8: index, %arg9: index, %arg10: index, %arg11: index):
        tensor.yield %cst_1 : f32
    } : tensor<1x3x416x416xf32> to tensor<1x3x418x418xf32>
    %4 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%3, %cst_2 : tensor<1x3x418x418xf32>, tensor<16x3x3x3xf32>) outs(%2 : tensor<1x16x416x416xf32>) -> tensor<1x16x416x416xf32>
    %5 = linalg.lrelu_2d_nchw ins(%4, %cst_1 : tensor<1x16x416x416xf32>, f32) outs(%4 : tensor<1x16x416x416xf32>) -> tensor<1x16x416x416xf32>
    %6 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%5, %0 : tensor<1x16x416x416xf32>, tensor<2x2xf32>) outs(%42 : tensor<1x16x208x208xf32>) -> tensor<1x16x208x208xf32>
    return %6 : tensor<1x16x208x208xf32>
  }
}


// RUN: mlir-opt %s -split-input-file -linalg-unfuse -linalg-structural-fusion \
// RUN: | FileCheck %s

// CHECK-LABEL: func @forward(
// CHECK-SAME:  %[[ARG0:[a-z0-9_]+]]: tensor<1x3x416x416xf32>)
func.func @forward(%arg0: tensor<1x3x416x416xf32>) -> tensor<1x16x208x208xf32> {
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
  // CHECK: %[[CST:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[FUSED:[a-z0-9_]+]] = linalg.fused(%[[ARG1:[a-z0-9_]+]] = %[[ARG0]] {{.*}}, %[[ARG2:[a-z0-9_]+]] = %[[CST]] {{.*}})
  // CHECK: %[[PAD:[a-z0-9_]+]] = tensor.pad %[[ARG1]] low[0, 0, 1, 1] high[0, 0, 1, 1]
  // CHECK:   tensor.yield %[[ARG2]]
  // CHECK: %[[INIT_TENSOR0:[a-z0-9_]+]] = linalg.init_tensor [1, 16, 416, 416] : tensor<1x16x416x416xf32>
  // CHECK: %[[CST0:[a-z0-9_]+]] = arith.constant dense<1.000000e+00> : tensor<16xf32>
  // CHECK: %[[CST1:[a-z0-9_]+]] = arith.constant dense<1.000000e+00> : tensor<16x3x3x3xf32>
  // CHECK: %[[CST2:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[INIT_TENSOR1:[a-z0-9_]+]] = linalg.init_tensor [1, 16, 208, 208] : tensor<1x16x208x208xf32>
  // CHECK: %[[INIT_TENSOR2:[a-z0-9_]+]] = linalg.init_tensor [2, 2] : tensor<2x2xf32>
  // CHECK: %[[BROAD:[a-z0-9_]+]] = linalg.broadcast_bias_2d_fchw ins(%[[CST0]]{{.*}}) outs(%[[INIT_TENSOR0]]{{.*}})
  // CHECK: %[[CONV:[a-z0-9_]+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[PAD]], %[[CST1]]{{.*}}) outs(%[[BROAD]]{{.*}})
  // CHECK: %[[LRELU:[a-z0-9_]+]] = linalg.lrelu_2d_nchw ins(%[[CONV]], %[[CST2]]{{.*}}) outs(%[[CONV]]{{.*}})
  // CHECK: %[[MAXPOOL:[a-z0-9_]+]] = linalg.pooling_nchw_max {{.*}} ins(%[[LRELU]], %[[INIT_TENSOR2]]{{.*}}) outs(%[[INIT_TENSOR1]]{{.*}})
  // CHECK: linalg.yield %[[MAXPOOL]]
  // CHECK: return %[[FUSED]]
  return %6 : tensor<1x16x208x208xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x10x10xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x8x8xf32>
func.func @unfuse_conv_2d_tensor_add(%ifm : tensor<1x1024x10x10xf32>, %summand : tensor<1x1024x8x8xf32>) -> tensor<1x1024x8x8xf32> {
    %zero = arith.constant 0.0 : f32
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %init = tensor.splat %zero : tensor<1x1024x8x8xf32>
    %result = linalg.conv_2d_tensor_add
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %summand, %weights, %bias : tensor<1x1024x10x10xf32>, tensor<1x1024x8x8xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>)
        outs(%init : tensor<1x1024x8x8xf32>)
        -> tensor<1x1024x8x8xf32>
    // CHECK: %[[FUSED:[a-z0-9_]+]] = linalg.fused(%[[ARG2:[a-z0-9_]+]] = %[[ifm]] {{.*}})
    // CHECK: %[[INIT_TENSOR:[a-z0-9_]+]] = linalg.init_tensor [1, 1024, 8, 8] : tensor<1x1024x8x8xf32>
    // CHECK: %[[CST:[a-z0-9_]+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK: %[[CST0:[a-z0-9_]+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK: %[[BROAD:[a-z0-9_]+]] = linalg.broadcast_bias_2d_fchw ins(%[[CST]]{{.*}}) outs(%[[INIT_TENSOR]]{{.*}})
    // CHECK: %[[CONV:[a-z0-9_]+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[ARG2]], %[[CST0]]{{.*}}) outs(%[[BROAD]]{{.*}})
    // CHECK: linalg.yield %[[CONV]]
    // CHECK: %[[ADDF:[a-z0-9_]+]] = arith.addf %[[FUSED]], %[[summand]]
    // CHECK: return %[[ADDF]]
    return %result : tensor<1x1024x8x8xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_relu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x17x17xf32>
func.func @unfuse_conv_2d_relu(%ifm : tensor<1x1024x17x17xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %init = tensor.splat %zero : tensor<1x1024x7x7xf32>
    %result = linalg.conv_2d_relu
        {dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
        ins(%ifm, %weights, %bias : tensor<1x1024x17x17xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>)
        outs(%init : tensor<1x1024x7x7xf32>)
        -> tensor<1x1024x7x7xf32>
    // CHECK: %[[FUSED:[a-z0-9_]+]] = linalg.fused(%[[ARG1:[a-z0-9_]+]] = %[[ifm]] {{.*}})
    // CHECK: %[[INIT_TENSOR:[a-z0-9_]+]] = linalg.init_tensor [1, 1024, 7, 7] : tensor<1x1024x7x7xf32>
    // CHECK: %[[CST:[a-z0-9_]+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK: %[[CST0:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[CST1:[a-z0-9_]+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK: %[[BROAD:[a-z0-9_]+]] = linalg.broadcast_bias_2d_fchw ins(%[[CST]]{{.*}}) outs(%[[INIT_TENSOR]]{{.*}})
    // CHECK: %[[CONV:[a-z0-9_]+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[ARG1]], %[[CST1]]{{.*}}) outs(%[[BROAD]]{{.*}})
    // CHECK: %[[RELU:[a-z0-9_]+]] = linalg.relu_2d_nchw ins(%[[CONV]]{{.*}}) outs(%[[CONV]]{{.*}})
    // CHECK: linalg.yield %[[RELU]]
    // CHECK: return %[[FUSED]]
    return %result : tensor<1x1024x7x7xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add_relu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x17x17xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x7x7xf32>
func.func @unfuse_conv_2d_tensor_add_relu(%ifm : tensor<1x1024x17x17xf32>, %summand : tensor<1x1024x7x7xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %init = tensor.splat %zero : tensor<1x1024x7x7xf32>
    %result = linalg.conv_2d_tensor_add_relu
        {dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
        ins(%ifm, %summand, %weights, %bias : tensor<1x1024x17x17xf32>, tensor<1x1024x7x7xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>)
        outs(%init : tensor<1x1024x7x7xf32>)
        -> tensor<1x1024x7x7xf32>
    // CHECK: %[[CST:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[FUSED:[a-z0-9_]+]] = linalg.fused(%[[ARG2:[a-z0-9_]+]] = %[[ifm]] {{.*}})
    // CHECK: %[[INIT_TENSOR:[a-z0-9_]+]] = linalg.init_tensor [1, 1024, 7, 7] : tensor<1x1024x7x7xf32>
    // CHECK: %[[CST0:[a-z0-9_]+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK: %[[CST1:[a-z0-9_]+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK: %[[BROAD:[a-z0-9_]+]] = linalg.broadcast_bias_2d_fchw ins(%[[CST0]]{{.*}}) outs(%[[INIT_TENSOR]]{{.*}})
    // CHECK: %[[CONV:[a-z0-9_]+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[ARG2]], %[[CST1]]{{.*}}) outs(%[[BROAD]]{{.*}})
    // CHECK: linalg.yield %[[CONV]]
    // CHECK: %[[ADDF:[a-z0-9_]+]] = arith.addf %[[FUSED]], %[[summand]]
    // CHECK: %[[RELU:[a-z0-9_]+]] = linalg.relu_2d_nchw ins(%[[ADDF]]{{.*}}) outs(%[[ADDF]]{{.*}})
    // CHECK: return %[[RELU]]
    return %result : tensor<1x1024x7x7xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_lrelu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
func.func @unfuse_conv_2d_lrelu(%ifm : tensor<1x1024x15x15xf32>) -> tensor<1x1024x13x13xf32> {
    %zero = arith.constant 0.0 : f32
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %alpha = arith.constant 2.000000e-02 : f32
    %init = tensor.splat %zero : tensor<1x1024x13x13xf32>
    %result = linalg.conv_2d_lrelu
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %weights, %bias, %alpha : tensor<1x1024x15x15xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, f32)
        outs(%init : tensor<1x1024x13x13xf32>)
        -> tensor<1x1024x13x13xf32>
    // CHECK: %[[FUSED:[a-z0-9_]+]] = linalg.fused(%[[ARG1:[a-z0-9_]+]] = %[[ifm]] {{.*}})
    // CHECK: %[[INIT_TENSOR:[a-z0-9_]+]] = linalg.init_tensor [1, 1024, 13, 13] : tensor<1x1024x13x13xf32>
    // CHECK: %[[CST:[a-z0-9_]+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK: %[[CST0:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[CST1:[a-z0-9_]+]] = arith.constant 2.000000e-02 : f32
    // CHECK: %[[CST2:[a-z0-9_]+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK: %[[BROAD:[a-z0-9_]+]] = linalg.broadcast_bias_2d_fchw ins(%[[CST]]{{.*}}) outs(%[[INIT_TENSOR]]{{.*}})
    // CHECK: %[[CONV:[a-z0-9_]+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[ARG1]], %[[CST2]]{{.*}}) outs(%[[BROAD]]{{.*}})
    // CHECK: %[[LRELU:[a-z0-9_]+]] = linalg.lrelu_2d_nchw ins(%[[CONV]]{{.*}}, %[[CST1]]{{.*}}) outs(%[[CONV]]{{.*}})
    // CHECK: linalg.yield %[[LRELU]]
    // CHECK: return %[[FUSED]]
    return %result : tensor<1x1024x13x13xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add_lrelu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x13x13xf32>
func.func @unfuse_conv_2d_tensor_add_lrelu(%ifm : tensor<1x1024x15x15xf32>, %summand : tensor<1x1024x13x13xf32>) -> tensor<1x1024x13x13xf32> {
    %zero = arith.constant 0.0 : f32
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %alpha = arith.constant 2.000000e-02 : f32
    %init = tensor.splat %zero : tensor<1x1024x13x13xf32>
    %result = linalg.conv_2d_tensor_add_lrelu
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %summand, %weights, %bias, %alpha : tensor<1x1024x15x15xf32>, tensor<1x1024x13x13xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, f32)
        outs(%init : tensor<1x1024x13x13xf32>)
        -> tensor<1x1024x13x13xf32>
    // CHECK: %[[CST:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[CST0:[a-z0-9_]+]] = arith.constant 2.000000e-02 : f32
    // CHECK: %[[FUSED:[a-z0-9_]+]] = linalg.fused(%[[ARG2:[a-z0-9_]+]] = %[[ifm]] {{.*}})
    // CHECK: %[[INIT_TENSOR:[a-z0-9_]+]] = linalg.init_tensor [1, 1024, 13, 13] : tensor<1x1024x13x13xf32>
    // CHECK: %[[CST1:[a-z0-9_]+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK: %[[CST2:[a-z0-9_]+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK: %[[BROAD:[a-z0-9_]+]] = linalg.broadcast_bias_2d_fchw ins(%[[CST1]]{{.*}}) outs(%[[INIT_TENSOR]]{{.*}})
    // CHECK: %[[CONV:[a-z0-9_]+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[ARG2]], %[[CST2]]{{.*}}) outs(%[[BROAD]]{{.*}})
    // CHECK: linalg.yield %[[CONV]]
    // CHECK: %[[ADDF:[a-z0-9_]+]] = arith.addf %[[FUSED]], %[[summand]]
    // CHECK: %[[LRELU:[a-z0-9_]+]] = linalg.lrelu_2d_nchw ins(%[[ADDF]]{{.*}}, %[[CST0]]{{.*}}) outs(%[[ADDF]]{{.*}})
    // CHECK: return %[[LRELU]]
    return %result : tensor<1x1024x13x13xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_lrelu_maxpool(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
func.func @unfuse_conv_2d_lrelu_maxpool(%ifm : tensor<1x1024x15x15xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %alpha = arith.constant 2.000000e-02 : f32

    %init = tensor.splat %zero : tensor<1x1024x7x7xf32>
    %result = linalg.conv_2d_lrelu_maxpool
        {
            dilations = dense<1> : tensor<2xi64>,
            strides = dense<1> : tensor<2xi64>,
            mp_kernel_size = dense<2> : tensor<2xi64>,
            mp_strides = dense<2> : tensor<2xi64>,
            mp_dilations = dense<1> : tensor<2xi64>,
            mp_padding = dense<[0, 1, 0, 1]> : tensor<4xi64>
        }
        ins(%ifm, %weights, %bias, %alpha : tensor<1x1024x15x15xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, f32)
        outs(%init : tensor<1x1024x7x7xf32>)
        -> tensor<1x1024x7x7xf32>
    // CHECK: %[[FUSED:[a-z0-9_]+]] = linalg.fused(%[[ARG1:[a-z0-9_]+]] = %[[ifm]] {{.*}})
    // CHECK: %[[INIT_TENSOR:[a-z0-9_]+]] = linalg.init_tensor [1, 1024, 13, 13] : tensor<1x1024x13x13xf32>
    // CHECK: %[[CST:[a-z0-9_]+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK: %[[CST0:[a-z0-9_]+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK: %[[CST1:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[CST2:[a-z0-9_]+]] = arith.constant 2.000000e-02 : f32
    // CHECK: %[[CST3:[a-z0-9_]+]] = arith.constant 0xFF800000 : f32
    // CHECK: %[[CST4:[a-z0-9_]+]] = arith.constant dense<0.000000e+00> : tensor<1x1024x7x7xf32>
    // CHECK: %[[INIT_TENSOR1:[a-z0-9_]+]] = linalg.init_tensor [2, 2] : tensor<2x2xf32>
    // CHECK: %[[BROAD:[a-z0-9_]+]] = linalg.broadcast_bias_2d_fchw ins(%[[CST]]{{.*}}) outs(%[[INIT_TENSOR]]{{.*}})
    // CHECK: %[[CONV:[a-z0-9_]+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[ARG1]], %[[CST0]]{{.*}}) outs(%[[BROAD]]{{.*}})
    // CHECK: %[[LRELU:[a-z0-9_]+]] = linalg.lrelu_2d_nchw ins(%[[CONV]]{{.*}}, %[[CST2]]{{.*}}) outs(%[[CONV]]{{.*}})
    // CHECK: %[[PAD:[a-z0-9_]+]] = tensor.pad %[[LRELU]] low[0, 0, 0, 0] high[0, 0, 1, 1]
    // CHECK:   tensor.yield %[[CST3]]
    // CHECK: %[[POOL:[a-z0-9_]+]] = linalg.pooling_nchw_max {{.*}} ins(%[[PAD]], %[[INIT_TENSOR1]]{{.*}}) outs(%[[CST4]]{{.*}})
    // CHECK: linalg.yield %[[POOL]]
    // CHECK: return %[[FUSED]]
    return %result : tensor<1x1024x7x7xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_relu_maxpool(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
func.func @unfuse_conv_2d_relu_maxpool(%ifm : tensor<1x1024x15x15xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>

    %init = tensor.splat %zero : tensor<1x1024x7x7xf32>
    %result = linalg.conv_2d_relu_maxpool
        {
            dilations = dense<1> : tensor<2xi64>,
            strides = dense<1> : tensor<2xi64>,
            mp_kernel_size = dense<2> : tensor<2xi64>,
            mp_strides = dense<2> : tensor<2xi64>,
            mp_dilations = dense<1> : tensor<2xi64>,
            mp_padding = dense<[0, 1, 0, 1]> : tensor<4xi64>
        }
        ins(%ifm, %weights, %bias : tensor<1x1024x15x15xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>)
        outs(%init : tensor<1x1024x7x7xf32>)
        -> tensor<1x1024x7x7xf32>
    // CHECK: %[[FUSED:[a-z0-9_]+]] = linalg.fused(%[[ARG1:[a-z0-9_]+]] = %[[ifm]] {{.*}})
    // CHECK: %[[INIT_TENSOR:[a-z0-9_]+]] = linalg.init_tensor [1, 1024, 13, 13] : tensor<1x1024x13x13xf32>
    // CHECK: %[[CST:[a-z0-9_]+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK: %[[CST0:[a-z0-9_]+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK: %[[CST1:[a-z0-9_]+]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[CST2:[a-z0-9_]+]] = arith.constant 0xFF800000 : f32
    // CHECK: %[[CST3:[a-z0-9_]+]] = arith.constant dense<0.000000e+00> : tensor<1x1024x7x7xf32>
    // CHECK: %[[INIT_TENSOR1:[a-z0-9_]+]] = linalg.init_tensor [2, 2] : tensor<2x2xf32>
    // CHECK: %[[BROAD:[a-z0-9_]+]] = linalg.broadcast_bias_2d_fchw ins(%[[CST]]{{.*}}) outs(%[[INIT_TENSOR]]{{.*}})
    // CHECK: %[[CONV:[a-z0-9_]+]] = linalg.conv_2d_nchw_fchw {{.*}} ins(%[[ARG1]], %[[CST0]]{{.*}}) outs(%[[BROAD]]{{.*}})
    // CHECK: %[[RELU:[a-z0-9_]+]] = linalg.relu_2d_nchw ins(%[[CONV]]{{.*}}) outs(%[[CONV]]{{.*}})
    // CHECK: %[[PAD:[a-z0-9_]+]] = tensor.pad %[[RELU]] low[0, 0, 0, 0] high[0, 0, 1, 1]
    // CHECK:   tensor.yield %[[CST2]]
    // CHECK: %[[POOL:[a-z0-9_]+]] = linalg.pooling_nchw_max {{.*}} ins(%[[PAD]], %[[INIT_TENSOR1]]{{.*}}) outs(%[[CST3]]{{.*}})
    // CHECK: linalg.yield %[[POOL]]
    // CHECK: return %[[FUSED]]
    return %result : tensor<1x1024x7x7xf32>
}

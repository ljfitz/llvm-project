// RUN: mlir-opt %s -split-input-file -linalg-unfuse | FileCheck %s

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x10x10xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x8x8xf32>
func.func @unfuse_conv_2d_tensor_add(%ifm : tensor<1x1024x10x10xf32>, %summand : tensor<1x1024x8x8xf32>) -> tensor<1x1024x8x8xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>

    %init = tensor.splat %zero : tensor<1x1024x8x8xf32>
    %result = linalg.conv_2d_tensor_add
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %summand, %weights, %bias : tensor<1x1024x10x10xf32>, tensor<1x1024x8x8xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>)
        outs(%init : tensor<1x1024x8x8xf32>)
        -> tensor<1x1024x8x8xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK=SAME: outs(%[[biased]] :
    // CHECK: %[[out:.+]] = arith.addf %[[conv]], %[[summand]]

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x8x8xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_relu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x17x17xf32>
func.func @unfuse_conv_2d_relu(%ifm : tensor<1x1024x17x17xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>

    %init = tensor.splat %zero : tensor<1x1024x7x7xf32>
    %result = linalg.conv_2d_relu
        {dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
        ins(%ifm, %weights, %bias : tensor<1x1024x17x17xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>)
        outs(%init : tensor<1x1024x7x7xf32>)
        -> tensor<1x1024x7x7xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw

    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :

    // CHECK: %[[out:.+]] = linalg.relu_2d_nchw
    // CHECK-SAME: ins(%[[conv]] :
    // CHECK-SAME: outs(%[[conv]] :

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x7x7xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add_relu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x17x17xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x7x7xf32>
func.func @unfuse_conv_2d_tensor_add_relu(%ifm : tensor<1x1024x17x17xf32>, %summand : tensor<1x1024x7x7xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>

    %init = tensor.splat %zero : tensor<1x1024x7x7xf32>
    %result = linalg.conv_2d_tensor_add_relu
        {dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
        ins(%ifm, %summand, %weights, %bias : tensor<1x1024x17x17xf32>, tensor<1x1024x7x7xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>)
        outs(%init : tensor<1x1024x7x7xf32>)
        -> tensor<1x1024x7x7xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw

    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :

    // CHECK: %[[add:.+]] = arith.addf %[[conv]], %[[summand]]

    // CHECK: %[[out:.+]] = linalg.relu_2d_nchw
    // CHECK-SAME: ins(%[[add]] :
    // CHECK-SAME: outs(%[[add]] :

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x7x7xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_lrelu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
func.func @unfuse_conv_2d_lrelu(%ifm : tensor<1x1024x15x15xf32>) -> tensor<1x1024x13x13xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[alpha:.+]] = arith.constant 2.000000e-02 : f32
    %alpha = arith.constant 2.000000e-02 : f32

    %init = tensor.splat %zero : tensor<1x1024x13x13xf32>
    %result = linalg.conv_2d_lrelu
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %weights, %bias, %alpha : tensor<1x1024x15x15xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, f32)
        outs(%init : tensor<1x1024x13x13xf32>)
        -> tensor<1x1024x13x13xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK-SAME: ins(%[[bias]] :

    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :

    // CHECK: %[[out:.+]] = linalg.lrelu_2d_nchw
    // CHECK-SAME: ins(%[[conv]], %[[alpha]] :
    // CHECK-SAME: outs(%[[conv]] :

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x13x13xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_tensor_add_lrelu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
// CHECK-SAME: %[[summand:.+]]: tensor<1x1024x13x13xf32>
func.func @unfuse_conv_2d_tensor_add_lrelu(%ifm : tensor<1x1024x15x15xf32>, %summand : tensor<1x1024x13x13xf32>) -> tensor<1x1024x13x13xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[alpha:.+]] = arith.constant 2.000000e-02 : f32
    %alpha = arith.constant 2.000000e-02 : f32

    %init = tensor.splat %zero : tensor<1x1024x13x13xf32>
    %result = linalg.conv_2d_tensor_add_lrelu
        {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%ifm, %summand, %weights, %bias, %alpha : tensor<1x1024x15x15xf32>, tensor<1x1024x13x13xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, f32)
        outs(%init : tensor<1x1024x13x13xf32>)
        -> tensor<1x1024x13x13xf32>

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK-SAME: ins(%[[bias]] :

    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :

    // CHECK: %[[add:.+]] = arith.addf %[[conv]], %[[summand]]

    // CHECK: %[[out:.+]] = linalg.lrelu_2d_nchw
    // CHECK-SAME: ins(%[[add]], %[[alpha]] :
    // CHECK-SAME: outs(%[[add]] :

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x13x13xf32>
}

// -----

// CHECK-LABEL: func @unfuse_conv_2d_lrelu_maxpool(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
func.func @unfuse_conv_2d_lrelu_maxpool(%ifm : tensor<1x1024x15x15xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[alpha:.+]] = arith.constant 2.000000e-02 : f32
    %alpha = arith.constant 2.000000e-02 : f32

    // CHECK-DAG: %[[pad_value:.+]] = arith.constant 0xFF800000 : f32

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

    // CHECK: %[[biased:.+]] = linalg.broadcast_bias_2d_fchw
    // CHECK-SAME: ins(%[[bias]] :

    // CHECK: %[[conv:.+]] = linalg.conv_2d_nchw_fchw
    // CHECK-SAME: ins(%[[ifm]], %[[weights]] :
    // CHECK-SAME: outs(%[[biased]] :

    // CHECK: %[[lrelu:.+]] = linalg.lrelu_2d_nchw
    // CHECK-SAME: ins(%[[conv]], %[[alpha]] :
    // CHECK-SAME: outs(%[[conv]] :

    // CHECK: %[[padded:.+]] = tensor.pad %[[lrelu]] low[0, 0, 0, 0] high[0, 0, 1, 1]
    // CHECK: tensor.yield %[[pad_value]] : f32

    // CHECK: %[[pool:.+]] = linalg.init_tensor [2, 2]

    // CHECK: %[[out:.+]] = linalg.pooling_nchw_max
    // CHECK: ins(%[[padded]], %[[pool]] :

    // CHECK: return %[[out]]
    return %result : tensor<1x1024x7x7xf32>
}

// -----

// CHECK-DAG: #[[sumIn:.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK-DAG: #[[sumOut:.+]] = affine_map<(d0, d1, d2) -> (d0, 0, d1)>
// CHECK-DAG: #[[divIn:.+]] = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
// CHECK-DAG: #[[divOut:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK: func @unfuse_softmax(
// CHECK-SAME: %[[ifm:.+]]: tensor<4x3x2xf32>
func.func @unfuse_softmax(%ifm : tensor<4x3x2xf32>) -> tensor<4x3x2xf32> {
    // BUG: linalg.softmax does not have a proper assembly format with $dim!
    %result = "linalg.softmax" (%ifm) {dim = -2 : i64} : (tensor<4x3x2xf32>) -> (tensor<4x3x2xf32>)

    // CHECK: %[[exp:.+]] = math.exp %[[ifm]]

    // CHECK: %[[sum:.+]] = linalg.generic
    // CHECK: indexing_maps = [#[[sumIn]], #[[sumOut]]]
    // CHECK: iterator_types = ["parallel", "parallel", "reduction"]
    // CHECK: ins(%[[exp]] :

    // CHECK: %[[out:.+]] = linalg.generic
    // CHECK: indexing_maps = [#[[divIn]], #[[divOut]]]
    // CHECK: iterator_types = ["parallel", "parallel", "parallel"]
    // CHECK: ins(%[[sum]]
    // CHECK: outs(%[[exp]]

    // CHECK: return %[[out]]
    return %result : tensor<4x3x2xf32>
}

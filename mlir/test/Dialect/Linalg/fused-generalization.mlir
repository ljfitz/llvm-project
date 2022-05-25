// RUN: mlir-opt %s -split-input-file -linalg-unfuse | FileCheck %s

// CHECK-LABEL: func @generalize_conv_2d_relu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x17x17xf32>
func @generalize_conv_2d_relu(%ifm : tensor<1x1024x17x17xf32>) -> tensor<1x1024x7x7xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[alpha:.+]] = arith.constant 2.000000e-02 : f32
    %alpha = arith.constant 2.000000e-02 : f32
    %init = tensor.splat %zero : tensor<1x1024x7x7xf32>

    %result = linalg.conv_2d_lrelu 
        {dilation = dense<2> : tensor<2xi64>, stride = dense<2> : tensor<2xi64>} 
        ins(%ifm, %weights, %bias, %alpha : tensor<1x1024x17x17xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, f32) 
        outs(%init : tensor<1x1024x7x7xf32>) 
        -> tensor<1x1024x7x7xf32>
    
    // CHECK: %[[biased:.+]] = linalg.apply_bias_2d_fchw
    // CHECK: ins(%[[init:.+]], %[[bias]] :
    // CHECK: outs(%[[init]] :

    // CHECK: %[[conv:.*]] = linalg.conv_2d_nchw_fchw
    // CHECK: ins(%[[ifm]], %[[weights]] :
    // CHECK: outs(%[[biased]] :

    // CHECK: %[[out:.*]] = linalg.lrelu_2d_nchw
    // CHECK: ins(%[[conv]], %[[alpha]] :
    // CHECK: outs(%[[conv]] :
    
    // CHECK: return %[[out]]
    return %result : tensor<1x1024x7x7xf32>
}

// -----

// CHECK-LABEL: func @generalize_conv_2d_lrelu(
// CHECK-SAME: %[[ifm:.+]]: tensor<1x1024x15x15xf32>
func @generalize_conv_2d_lrelu(%ifm : tensor<1x1024x15x15xf32>) -> tensor<1x1024x13x13xf32> {
    %zero = arith.constant 0.0 : f32
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[alpha:.+]] = arith.constant 2.000000e-02 : f32
    %alpha = arith.constant 2.000000e-02 : f32
    %init = tensor.splat %zero : tensor<1x1024x13x13xf32>

    %result = linalg.conv_2d_lrelu 
        {dilation = dense<1> : tensor<2xi64>, stride = dense<1> : tensor<2xi64>} 
        ins(%ifm, %weights, %bias, %alpha : tensor<1x1024x15x15xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, f32) 
        outs(%init : tensor<1x1024x13x13xf32>) 
        -> tensor<1x1024x13x13xf32>
    
    // CHECK: %[[biased:.+]] = linalg.apply_bias_2d_fchw
    // CHECK: ins(%[[init:.+]], %[[bias]] :
    // CHECK: outs(%[[init]] :

    // CHECK: %[[conv:.*]] = linalg.conv_2d_nchw_fchw
    // CHECK: ins(%[[ifm]], %[[weights]] :
    // CHECK: outs(%[[biased]] :

    // CHECK: %[[out:.*]] = linalg.lrelu_2d_nchw
    // CHECK: ins(%[[conv]], %[[alpha]] :
    // CHECK: outs(%[[conv]] :
    
    // CHECK: return %[[out]]
    return %result : tensor<1x1024x13x13xf32>
}

// -----

// TODO: @generalize_conv_2d_lrelu_maxpool

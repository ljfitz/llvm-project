// RUN: mlir-opt %s -split-input-file -linalg-generalize-fused-ops | FileCheck %s

// TODO: @generalize_conv_2d_relu

// -----

// CHECK-LABEL: func @generalize_conv_2d_lrelu(
// CHECK-SAME: %[[arg0:.+]]: tensor<1x1024x15x15xf32>
func @generalize_conv_2d_lrelu(%ifm : tensor<1x1024x15x15xf32>) -> tensor<1x1024x13x13xf32> {
    // CHECK-DAG: %[[weights:.+]] = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    %weights = arith.constant dense<5.000000e-01> : tensor<1024x1024x3x3xf32>
    // CHECK-DAG: %[[bias:.+]] = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    %bias = arith.constant dense<3.000000e-01> : tensor<1024xf32>
    // CHECK-DAG: %[[leak:.+]] = arith.constant 2.000000e-02 : f32
    %leak = arith.constant 2.000000e-02 : f32

    %init = linalg.init_tensor [1, 1024, 13, 13] : tensor<1x1024x13x13xf32>
    %result = linalg.conv_2d_lrelu 
        {dilation = dense<1> : tensor<2xi64>, stride = dense<1> : tensor<2xi64>} 
        ins(%ifm, %weights, %bias, %leak : tensor<1x1024x15x15xf32>, tensor<1024x1024x3x3xf32>, tensor<1024xf32>, f32) 
        outs(%init : tensor<1x1024x13x13xf32>) 
        -> tensor<1x1024x13x13xf32>
    
    return %result : tensor<1x1024x13x13xf32>
}

// -----

// TODO: @generalize_conv_2d_lrelu_maxpool

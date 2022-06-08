// RUN: mlir-opt -split-input-file %s

func.func @fuse_on_tensor(%arg0: tensor<1x3x108x108xf32>) -> tensor<1x16x106x106xf32> {
    %result = linalg.fused (%ifm = %arg0 : tensor<1x3x108x108xf32>) {
        %weights = arith.constant dense<5.000000e-01> : tensor<16x3x3x3xf32>
        %init = arith.constant dense<0.000000e00> : tensor<1x16x106x106xf32>
        %conv = linalg.conv_2d_nchw_fchw
            {dilation = dense<1> : tensor<2xi64>, stride = dense<1> : tensor<2xi64>}
            ins(%ifm, %weights : tensor<1x3x108x108xf32>, tensor<16x3x3x3xf32>)
            outs(%init : tensor<1x16x106x106xf32>)
            -> tensor<1x16x106x106xf32>
        %act = linalg.relu_2d_nchw
            ins(%conv : tensor<1x16x106x106xf32>)
            outs(%conv : tensor<1x16x106x106xf32>)
            -> tensor<1x16x106x106xf32>
        linalg.yield %act : tensor<1x16x106x106xf32>
    } -> tensor<1x16x106x106xf32>
    return %result : tensor<1x16x106x106xf32>
}

// -----

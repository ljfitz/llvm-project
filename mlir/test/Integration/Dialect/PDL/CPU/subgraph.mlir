// RUN: mlir-opt %s  -allow-unregistered-dialect -test-pdl-bytecode-pass -split-input-file | FileCheck %s

module @patterns {

  pdl.pattern : benefit(1) {

    // ( ( transpose -> maxpool ) -> transpose )
    // %7 = "tosa.transpose"(%5, %6) : (tensor<1x3x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x3xf32>
    // %8 = "tosa.max_pool2d"(%7) {kernel = [2, 2], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x128x128x3xf32>) -> tensor<1x64x64x3xf32>
    // %10 = "tosa.transpose"(%8, %9) : (tensor<1x64x64x3xf32>, tensor<4xi32>) -> tensor<1x3x64x64xf32>

    %ARG0_TYPE = pdl.type
    %ARG1_TYPE = pdl.type
    %ARG2_TYPE = pdl.type
    %TRANSPOSE_TYPE = pdl.type
    %MAXPOOL_TYPE = pdl.type
    %TRANSPOSE_TWO_TYPE = pdl.type

    %ARG0 = pdl.operand : %ARG0_TYPE
    %ARG1 = pdl.operand : %ARG1_TYPE
    %ARG2 = pdl.operand : %ARG2_TYPE

    %TRANSPOSE_OP = pdl.operation "tosa.transpose" (%ARG0, %ARG1 : !pdl.value, !pdl.value) -> (%TRANSPOSE_TYPE : !pdl.type)
    %TRANSPOSE_VALUE = pdl.result 0 of %TRANSPOSE_OP

    %MAXPOOL_OP = pdl.operation "tosa.max_pool2d" (%TRANSPOSE_VALUE : !pdl.value) -> (%MAXPOOL_TYPE : !pdl.type)
    %MAXPOOL_VALUE = pdl.result 0 of %MAXPOOL_OP

    %TRANSPOSE_TWO_OP = pdl.operation "tosa.transpose" (%MAXPOOL_VALUE, %ARG2 : !pdl.value, !pdl.value) -> (%TRANSPOSE_TWO_TYPE : !pdl.type)
    %TRANSPOSE_TWO_VALUE = pdl.result 0 of %TRANSPOSE_TWO_OP

    // %alpha_attr = pdl.attribute
    // %op1 = pdl.operation "tosa.const" {"value" = %alpha_attr2} -> (%alpha_const_type2 : !pdl.type)
    // %val1 = pdl.result 0 of %op1

    // Rewrite rule
    pdl.rewrite {

      %SUBGRAPH_OP = pdl.apply_native_rewrite "createSubgraphOp" (%TRANSPOSE_OP : !pdl.operation) : !pdl.operation
      // %SUBGRAPH_TWO_OP = pdl.apply_native_rewrite "createSubgraphOp" (%SUBGRAPH_OP : !pdl.operation) : !pdl.operation

      %SUBGRAPH_THREE_OP = pdl.apply_native_rewrite "createSubgraphOp" (%TRANSPOSE_TWO_OP : !pdl.operation) : !pdl.operation

      // %SUBGRAPH_TWO_OP = pdl.apply_native_rewrite "createSubgraphOp" (%MAXPOOL_OP : !pdl.operation) : !pdl.operation
      pdl.apply_native_rewrite "addOpToSubgraphOp" (%SUBGRAPH_OP, %MAXPOOL_OP : !pdl.operation, !pdl.operation)
    }
  }
}


module @ir attributes { test.mlp_split } {
  func.func @forward(%arg0: tensor<1x3x128x128xf32>) -> tensor<1x3x64x64xf32> {
    %1 = "tosa.const"() {value = dense<1.000000e-01> : tensor<f32>} : () -> tensor<f32>
    %2 = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %3 = "tosa.greater_equal"(%arg0, %2) : (tensor<1x3x128x128xf32>, tensor<f32>) -> tensor<1x3x128x128xi1>
    %4 = "tosa.mul"(%arg0, %1) {shift = 0 : i32} : (tensor<1x3x128x128xf32>, tensor<f32>) -> tensor<1x3x128x128xf32>
    %5 = "tosa.select"(%3, %arg0, %4) : (tensor<1x3x128x128xi1>, tensor<1x3x128x128xf32>, tensor<1x3x128x128xf32>) -> tensor<1x3x128x128xf32>
    %6 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
    %7 = "tosa.transpose"(%5, %6) : (tensor<1x3x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x3xf32>
    %8 = "tosa.max_pool2d"(%7) {kernel = [2, 2], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x128x128x3xf32>) -> tensor<1x64x64x3xf32>
    %9 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %10 = "tosa.transpose"(%8, %9) : (tensor<1x64x64x3xf32>, tensor<4xi32>) -> tensor<1x3x64x64xf32>
    return %10 : tensor<1x3x64x64xf32>
  }
}

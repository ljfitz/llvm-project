// RUN: mlir-opt %s  -allow-unregistered-dialect -test-pdl-bytecode-pass -split-input-file | FileCheck %s

// -----

module @patterns {

  pdl.pattern : benefit(1) {

    // ( ( transpose -> maxpool ) -> transpose )
    // %7 = "tosa.transpose"(%5, %6) : (tensor<1x3x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x3xf32>
    // %8 = "tosa.max_pool2d"(%7) {kernel = [2, 2], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x128x128x3xf32>) -> tensor<1x64x64x3xf32>
    // %10 = "tosa.transpose"(%8, %9) : (tensor<1x64x64x3xf32>, tensor<4xi32>) -> tensor<1x3x64x64xf32>

    %ARG0_TYPE = pdl.type
    %ARG0 = pdl.operand : %ARG0_TYPE
    %ARG1_TYPE = pdl.type
    %ARG1 = pdl.operand : %ARG1_TYPE
    %TRANSPOSE_TYPE = pdl.type
    %TRANSPOSE_OP = pdl.operation "tosa.transpose" (%ARG0, %ARG1 : !pdl.value, !pdl.value) -> (%TRANSPOSE_TYPE : !pdl.type)
    %TRANSPOSE_VALUE = pdl.result 0 of %TRANSPOSE_OP

    %MAXPOOL_TYPE = pdl.type
    %MAXPOOL_OP = pdl.operation "tosa.max_pool2d" (%TRANSPOSE_VALUE : !pdl.value) -> (%MAXPOOL_TYPE : !pdl.type)
    %MAXPOOL_VALUE = pdl.result 0 of %MAXPOOL_OP

    %CONST_TYPE = pdl.type
    %CONST_ATTR = pdl.attribute
    %CONST_OP = pdl.operation "tosa.const" {"value" = %CONST_ATTR} -> (%CONST_TYPE : !pdl.type)
    %CONST_VALUE = pdl.result 0 of %CONST_OP

    %TRANSPOSE_TWO_TYPE = pdl.type
    %TRANSPOSE_TWO_OP = pdl.operation "tosa.transpose" (%MAXPOOL_VALUE, %CONST_VALUE : !pdl.value, !pdl.value) -> (%TRANSPOSE_TWO_TYPE : !pdl.type)
    %TRANSPOSE_TWO_VALUE = pdl.result 0 of %TRANSPOSE_TWO_OP

    // %alpha_attr = pdl.attribute
    // %op1 = pdl.operation "tosa.const" {"value" = %alpha_attr2} -> (%alpha_const_type2 : !pdl.type)
    // %val1 = pdl.result 0 of %op1

    // Rewrite rules
    pdl.rewrite {

      // Patterns are applied independently from each other.
      // Ensure that we do not match inside subgraph ops by modifying `applyPatternsAndFoldGreedily`.

      %SUBGRAPH_OP = pdl.apply_native_rewrite "createSubgraphOp" (%TRANSPOSE_OP : !pdl.operation) : !pdl.operation
      %SUBGRAPH_TWO_OP = pdl.apply_native_rewrite "addOpToSubgraphOp" (%SUBGRAPH_OP, %MAXPOOL_OP : !pdl.operation, !pdl.operation) : !pdl.operation
      %SUBGRAPH_THREE_OP = pdl.apply_native_rewrite "createSubgraphOp" (%SUBGRAPH_TWO_OP : !pdl.operation) : !pdl.operation
      %SUBGRAPH_FOUR_OP = pdl.apply_native_rewrite "addOpToSubgraphOp" (%SUBGRAPH_THREE_OP, %TRANSPOSE_TWO_OP : !pdl.operation, !pdl.operation) : !pdl.operation
    }
  }
}

// CHECK-LABEL:   module @ir attributes {test.mlp_split} {
// CHECK:           func.func @forward(%[[VAL_0:.*]]: tensor<1x3x128x128xf32>) -> tensor<1x3x64x64xf32> {
// CHECK:             %[[VAL_1:.*]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:             %[[VAL_2:.*]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
// CHECK:             %[[VAL_3:.*]] = "tosa.const"() {value = dense<1.000000e-01> : tensor<f32>} : () -> tensor<f32>
// CHECK:             %[[VAL_4:.*]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK:             %[[VAL_5:.*]] = "tosa.greater_equal"(%[[VAL_0]], %[[VAL_4]]) : (tensor<1x3x128x128xf32>, tensor<f32>) -> tensor<1x3x128x128xi1>
// CHECK:             %[[VAL_6:.*]] = "tosa.mul"(%[[VAL_0]], %[[VAL_3]]) {shift = 0 : i32} : (tensor<1x3x128x128xf32>, tensor<f32>) -> tensor<1x3x128x128xf32>
// CHECK:             %[[VAL_7:.*]] = "tosa.select"(%[[VAL_5]], %[[VAL_0]], %[[VAL_6]]) : (tensor<1x3x128x128xi1>, tensor<1x3x128x128xf32>, tensor<1x3x128x128xf32>) -> tensor<1x3x128x128xf32>
// CHECK:             %[[VAL_8:.*]] = linalg.subgraph(%[[VAL_9:.*]] = %[[VAL_7]] : tensor<1x3x128x128xf32>, %[[VAL_10:.*]] = %[[VAL_2]] : tensor<4xi32>, %[[VAL_11:.*]] = %[[VAL_1]] : tensor<4xi32>) {
// CHECK:               %[[VAL_12:.*]] = linalg.subgraph(%[[VAL_13:.*]] = %[[VAL_9]] : tensor<1x3x128x128xf32>, %[[VAL_14:.*]] = %[[VAL_10]] : tensor<4xi32>) {
// CHECK:                 %[[VAL_15:.*]] = "tosa.transpose"(%[[VAL_13]], %[[VAL_14]]) : (tensor<1x3x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x3xf32>
// CHECK:                 %[[VAL_16:.*]] = "tosa.max_pool2d"(%[[VAL_15]]) {kernel = [2, 2], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x128x128x3xf32>) -> tensor<1x64x64x3xf32>
// CHECK:                 linalg.yield %[[VAL_16]] : tensor<1x64x64x3xf32>
// CHECK:               } -> tensor<1x64x64x3xf32>
// CHECK:               %[[VAL_17:.*]] = "tosa.transpose"(%[[VAL_18:.*]], %[[VAL_11]]) : (tensor<1x64x64x3xf32>, tensor<4xi32>) -> tensor<1x3x64x64xf32>
// CHECK:               linalg.yield %[[VAL_17]] : tensor<1x3x64x64xf32>
// CHECK:             } -> tensor<1x3x64x64xf32>
// CHECK:             return %[[VAL_19:.*]] : tensor<1x3x64x64xf32>
// CHECK:           }
// CHECK:         }
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

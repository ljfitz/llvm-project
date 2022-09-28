// RUN: mlir-opt %s  -allow-unregistered-dialect -test-pdl-bytecode-pass -split-input-file | FileCheck %s

module @patterns {

  pdl.pattern : benefit(1) {
    // Define input and types
    %in_type = pdl.type
    %other_type = pdl.type
    // %subgraph_type = pdl.type
    %alpha_const_type = pdl.type
    %alpha_const_type2 = pdl.type

    // %out_type = pdl.type
    %alpha_attr = pdl.attribute
    %alpha_attr2 = pdl.attribute

    %in_operand = pdl.operand : %in_type
    // %other_operand = pdl.operand : %other_type

    // Define ops for sugraph fusion
    %CONST_0 = pdl.operation "tosa.const" {"value" = %alpha_attr} -> (%alpha_const_type : !pdl.type)
    %CONST_0_RESULT = pdl.result 0 of %CONST_0




    // %7 = "tosa.transpose"(%5, %6) : (tensor<1x3x128x128xf32>, tensor<4xi32>) -> tensor<1x128x128x3xf32>
    // %8 = "tosa.max_pool2d"(%7) {kernel = [2, 2], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x128x128x3xf32>) -> tensor<1x64x64x3xf32>




    %op1 = pdl.operation "tosa.const" {"value" = %alpha_attr2} -> (%alpha_const_type2 : !pdl.type)
    %val1 = pdl.result 0 of %op1

    %op2 = pdl.operation "tosa.mul" (%in_operand, %val1 : !pdl.value, !pdl.value) -> (%in_type : !pdl.type)
    %val2 = pdl.result 0 of %op2


    %op3 = pdl.operation "tosa.greater_equal" (%in_operand, %CONST_0_RESULT : !pdl.value, !pdl.value) -> (%other_type : !pdl.type)
    %val3 = pdl.result 0 of %op3

    // pdl.replace %root with (%inputOperand)
    // pdl.rewrite %op2 with "createSubgraph"


    // Rewrite rule
    pdl.rewrite {

      %subgraphOp = pdl.apply_native_rewrite "createSubgraphOp" (%op2 : !pdl.operation) : !pdl.operation
      %subgraphOpTwo = pdl.apply_native_rewrite "createSubgraphOp" (%op3 : !pdl.operation) : !pdl.operation


      // pdl.apply_native_rewrite "addOpToSubgraphOp" (%subgraphOp, %op3 : !pdl.operation, !pdl.operation)

      //1pdl.replace %op3 with 1%val3 : !pdl.value)



      // %op6 = pdl.operation "linalg.subgraph" (%in_operand : !pdl.value) -> (%in_type : !pdl.type)
      // // registerRewriteFunction

      // %val6 = pdl.result 0 of %op3
      // pdl.replace %op2 with (%val6 : !pdl.value) 
      
      // pdl.erase %op1
      // pdl.erase %op2

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

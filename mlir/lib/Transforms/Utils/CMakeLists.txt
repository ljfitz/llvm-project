add_mlir_library(MLIRTransformUtils
  CommutativityUtils.cpp
  ControlFlowSinkUtils.cpp
  DialectConversion.cpp
  FoldUtils.cpp
  GreedyPatternRewriteDriver.cpp
  InliningUtils.cpp
  LoopInvariantCodeMotionUtils.cpp
  RegionUtils.cpp
  SideEffectUtils.cpp
  TopologicalSortUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRLinalgPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRAnalysis
  MLIRLoopLikeInterface
  MLIRRewrite
  )

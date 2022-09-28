//===- TestPDLByteCode.cpp - Test rewriter bytecode functionality ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

/// Custom constraint invoked from PDL.
static LogicalResult customSingleEntityConstraint(PatternRewriter &rewriter,
                                                  Operation *rootOp) {
  return success(rootOp->getName().getStringRef() == "test.op");
}
static LogicalResult customMultiEntityConstraint(PatternRewriter &rewriter,
                                                 Operation *root,
                                                 Operation *rootCopy) {
  return customSingleEntityConstraint(rewriter, rootCopy);
}
static LogicalResult customMultiEntityVariadicConstraint(
    PatternRewriter &rewriter, ValueRange operandValues, TypeRange typeValues) {
  if (operandValues.size() != 2 || typeValues.size() != 2)
    return failure();
  return success();
}

// Custom creator invoked from PDL.
static Operation *customCreate(PatternRewriter &rewriter, Operation *op) {
  return rewriter.create(OperationState(op->getLoc(), "test.success"));
}
static auto customVariadicResultCreate(PatternRewriter &rewriter,
                                       Operation *root) {
  return std::make_pair(root->getOperands(), root->getOperands().getTypes());
}
static Type customCreateType(PatternRewriter &rewriter) {
  return rewriter.getF32Type();
}
static std::string customCreateStrAttr(PatternRewriter &rewriter) {
  return "test.str";
}

/// Subgraph operation implementation

/// Determines whether @p target can be wrapped in a `linalg.subgraph`
/// operation.
static bool canWrapInSubgraph(Operation *target) {
  if (!target)
    return false;

  // - target must have exactly 0 or 1 result
  switch (target->getNumResults()) {
  case 0:
    return true;

  case 1:
    // - the result must be any ranked tensor type
    return target->getResult(0).getType().isa<RankedTensorType>();

  default:
    return false;
  }
}

/// Collects @p op is used and traverses graph based on the @p root operation
static void collectUses(Operation *root, Operation *op,
                        SmallVectorImpl<Value> &uses) {
  const auto isLocal = [&](Value value) {
    auto source = value.getDefiningOp();
    if (auto barg = value.dyn_cast<BlockArgument>())
      source = barg.getOwner()->getParentOp();
    while (source) {
      if (source == root)
        return true;
      source = source->getParentOp();
    }
    return false;
  };
  const auto unite = [&](Value operand) {
    if (isLocal(operand))
      return;
    if (llvm::find(uses, operand) != uses.end())
      return;
    uses.push_back(operand);
  };

  llvm::for_each(op->getOperands(), unite);

  for (auto &region : op->getRegions()) {
    for (auto &op : region.getOps()) {
      collectUses(root, &op, uses);
    }
  }
}

/// Wraps @p target in a `linalg.subgraph` operation.
///
/// See canWrapInSubgraph() on what scenarios this transformation is allowed in.
///
/// @pre      `canWrapInSubgraph(target)`
///
/// Given IR of the form:
/// ```mlir
/// %r = target %o_0, ...
/// ```
///
/// This produces:
/// ```mlir
/// %r = linalg.subgraph (%c_0 = %o_0, ...) {
///   %w = target %c_0, ...
///   linalg.yield %w
/// }
/// ```
static Operation *wrapInSubgraph(PatternRewriter &rewriter, Operation *target) {

  assert(canWrapInSubgraph(target));

  SmallVector<Value> captures;
  collectUses(target, target, captures);

  // Create the Subgraph.
  OpBuilder builder(target);
  auto subgraph = builder.create<linalg::SubgraphOp>(
      target->getLoc(),
      /*resulType=*/target->getNumResults() ? target->getResult(0).getType()
                                            : Type{},
      /*captures=*/captures,
      [=](OpBuilder &builder, Location loc, BlockAndValueMapping &captures) {
        // Clone the target op into the Subgraph.
        auto clonedOp = builder.insert(target->clone(captures));
        // Yield the result of the cloned op (which may have none).
        builder.create<linalg::YieldOp>(target->getLoc(),
                                        clonedOp->getResults());
      });

  // Remove the target operation.
  target->replaceAllUsesWith(subgraph);
  target->erase();

  return subgraph.getOperation();
}

/// Determines whether @p consumer can be appended to @p target .
static bool canFuseConsumer(linalg::SubgraphOp target, Operation *consumer) {
  if (!consumer)
    return false;

  // - target may not be a user of consumer
  if (llvm::find(consumer->getUsers(), target) != consumer->getUsers().end())
    return false;

  // - consumer must be the only user of target (if any).
  if (llvm::any_of(target->getUsers(),
                   [&](auto user) { return user != consumer; }))
    return false;

  // - consumer must be wrappable
  return canWrapInSubgraph(consumer);
}

/// Computes the length of the longest path from \p op to the function entry
static size_t computeAncestorLength(Operation *op) {
  if (!op)
    return 0;

  // Cache the results to avoid re-computation
  static DenseMap<Operation *, size_t> lengthCache;

  // Check cache before starting the computation
  auto it = lengthCache.find(op);
  if (it != lengthCache.end()) {
    return it->second;
  }

  size_t currentLength = 0;
  for (unsigned idx = 0; idx < op->getNumOperands(); ++idx) {
    Operation *ancestor = op->getOperand(idx).getDefiningOp();
    currentLength =
        std::max(currentLength, 1 + computeAncestorLength(ancestor));
  }

  // Update cache with computed value
  lengthCache.insert({op, currentLength});
  return currentLength;
}

/// Fused @p consumer into @p target .
///
/// See canFuseConsumer() on what scenarios this transformation is allowed in.
///
/// @pre      `canFuseConsumer(target, consumer)`
///
/// Given IR of the form:
/// ```mlir
/// %z = linalg.subgraph (%x_0 = %y_0, ...) {
///   ...
/// }
/// %r = consumer %o_0, ..., %z, ...
/// ```
///
/// This produces:
/// ```mlir
/// %r = linalg.subgraph (%x_0 = %y_0, ..., %o_0, ...) {
///   ...
///   %r = consumer %o_0, ..., %z, ...
///   linalg.yield %r
/// }
/// ```
static linalg::SubgraphOp fuseConsumer(Operation *subgraphOp,
                                       Operation *consumer) {

  linalg::SubgraphOp target = static_cast<linalg::SubgraphOp>(subgraphOp);
  assert(canFuseConsumer(target, consumer));

  // Ensure that all operands of consumer are captured.
  auto newCaptures = llvm::to_vector(target.captures());
  collectUses(consumer, consumer, newCaptures);
  llvm::erase_if(newCaptures,
                 [&](Value capture) { return capture == target.result(); });

  // Create the new Subgraph.
  OpBuilder builder(consumer);
  auto result = builder.create<linalg::SubgraphOp>(
      target.getLoc(),
      /*resultType=*/consumer->getNumResults()
          ? consumer->getResult(0).getType()
          : Type{},
      newCaptures,
      [&](OpBuilder &builder, Location loc, BlockAndValueMapping &captures) {
        // Remap the capture arguments of the old Subgraph.
        for (unsigned idx = 0; idx < target.captures().size(); ++idx) {
          captures.map(target.getCaptureArgs()[idx],
                       captures.lookup(target.captures()[idx]));
        }

        // Clone the contents of the old body (without the terminator).
        for (auto &op : target.getBody().front().without_terminator()) {
          builder.insert(op.clone(captures));
        }

        // If the old fused op had a result, add it to the mapping.
        auto terminator = target.getBody().front().getTerminator();
        if (terminator->getNumOperands())
          captures.map(target.result(),
                       captures.lookup(terminator->getOperand(0)));

        // Clone the op to be appended to the new block.
        auto appended = builder.insert(consumer->clone(captures));
        // Yield the result of the cloned op (which may have none).
        builder.create<linalg::YieldOp>(target->getLoc(),
                                        appended->getResults());
      });

  // Remove the consumer and then the target operation.
  consumer->replaceAllUsesWith(result);
  consumer->erase();
  target.erase();
  return result;
}

static Operation *createSubgraphOp(PatternRewriter &rewriter, Operation *op) {
  op->dump();
  return wrapInSubgraph(rewriter, op);
}

static void addOpToSubgraphOp(PatternRewriter &rewriter, Operation *subgraphOp,
                              Operation *op) {
  // subgraphOp->dump();
  fuseConsumer(subgraphOp, op);
}

/// Custom rewriter invoked from PDL.
static void customRewriter(PatternRewriter &rewriter, Operation *root,
                           Value input) {
  rewriter.create(root->getLoc(), rewriter.getStringAttr("test.success"),
                  input);
  rewriter.eraseOp(root);
}

namespace {
struct TestPDLByteCodePass
    : public PassWrapper<TestPDLByteCodePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPDLByteCodePass)

  StringRef getArgument() const final { return "test-pdl-bytecode-pass"; }
  StringRef getDescription() const final {
    return "Test PDL ByteCode functionality";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    // Mark the pdl_interp dialect as a dependent. This is needed, because we
    // create ops from that dialect as a part of the PDL-to-PDLInterp lowering.
    registry.insert<pdl_interp::PDLInterpDialect>();
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() final {
    ModuleOp module = getOperation();

    // The test cases are encompassed via two modules, one containing the
    // patterns and one containing the operations to rewrite.
    ModuleOp patternModule = module.lookupSymbol<ModuleOp>(
        StringAttr::get(module->getContext(), "patterns"));
    ModuleOp irModule = module.lookupSymbol<ModuleOp>(
        StringAttr::get(module->getContext(), "ir"));
    if (!patternModule || !irModule)
      return;

    RewritePatternSet patternList(module->getContext());

    // Register ahead of time to test when functions are registered without a
    // pattern.
    patternList.getPDLPatterns().registerConstraintFunction(
        "multi_entity_constraint", customMultiEntityConstraint);
    patternList.getPDLPatterns().registerConstraintFunction(
        "single_entity_constraint", customSingleEntityConstraint);

    // Process the pattern module.
    patternModule.getOperation()->remove();
    PDLPatternModule pdlPattern(patternModule);

    // Note: This constraint was already registered, but we re-register here to
    // ensure that duplication registration is allowed (the duplicate mapping
    // will be ignored). This tests that we support separating the registration
    // of library functions from the construction of patterns, and also that we
    // allow multiple patterns to depend on the same library functions (without
    // asserting/crashing).
    pdlPattern.registerConstraintFunction("multi_entity_constraint",
                                          customMultiEntityConstraint);
    pdlPattern.registerConstraintFunction("multi_entity_var_constraint",
                                          customMultiEntityVariadicConstraint);
    pdlPattern.registerRewriteFunction("creator", customCreate);
    pdlPattern.registerRewriteFunction("var_creator",
                                       customVariadicResultCreate);
    pdlPattern.registerRewriteFunction("type_creator", customCreateType);
    pdlPattern.registerRewriteFunction("str_creator", customCreateStrAttr);
    pdlPattern.registerRewriteFunction("rewriter", customRewriter);

    pdlPattern.registerRewriteFunction("createSubgraphOp", createSubgraphOp);
    pdlPattern.registerRewriteFunction("addOpToSubgraphOp", addOpToSubgraphOp);

    patternList.add(std::move(pdlPattern));

    // Invoke the pattern driver with the provided patterns.
    (void)applyPatternsAndFoldGreedily(irModule.getBodyRegion(),
                                       std::move(patternList));
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestPDLByteCodePass() { PassRegistration<TestPDLByteCodePass>(); }
} // namespace test
} // namespace mlir

//===- StructuralFusion.cpp - linalg fused ops generation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg structural fusion pass. It creates
// linalg.fused ops based on external fusion recipes.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/SideEffectUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/WithColor.h"

#define DEBUG_TYPE "linalg-structural-fusion"

using namespace mlir;
using namespace mlir::linalg;

/// Determines whether @p op matches @p filter .
static bool matches(Operation *op, OperatorClass filter) {
  using llvm::operator&;
  return (linalg::classifyOperator(op) & filter) != OperatorClass::None;
}

/// Determines whether @p target can be wrapped in a `linalg.fused` operation.
static bool canWrapInFusedOp(Operation *target) {
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

static void collectUses(Operation *root, Operation *op, SmallVectorImpl<Value> &uses) {
  const auto isLocal = [&](Value value) {
    auto source = value.getDefiningOp();
    if (auto barg = value.dyn_cast<BlockArgument>())
      source = barg.getOwner()->getParentOp();
    while (source) {
      if (source == root) return true;
      source = source->getParentOp();
    }
    return false;
  };
  const auto unite = [&](Value operand) {
    if (isLocal(operand)) return;
    if (llvm::find(uses, operand) != uses.end()) return;
    uses.push_back(operand);
  };

  llvm::for_each(op->getOperands(), unite);

  for (auto &region : op->getRegions()) {
    for (auto &op : region.getOps()) {
      collectUses(root, &op, uses);
    }
  }
}

/// Wraps @p target in a `linalg.fused` operation.
///
/// See canWrapInFusedOp() on what scenarios this transformation is allowed in.
///
/// @pre      `canWrapInFusedOp(target)`
///
/// Given IR of the form:
/// ```mlir
/// %r = target %o_0, ...
/// ```
///
/// This produces:
/// ```mlir
/// %r = linalg.fused (%c_0 = %o_0, ...) {
///   %w = target %c_0, ...
///   linalg.yield %w
/// }
/// ```
static FusedOp wrapInFusedOp(Operation *target) {
  assert(canWrapInFusedOp(target));

  SmallVector<Value> captures;
  collectUses(target, target, captures);

  // Create the FusedOp.
  OpBuilder builder(target);
  auto fusedOp = builder.create<FusedOp>(
      target->getLoc(),
      /*resulType=*/target->getNumResults()
          ? target->getResult(0).getType()
          : Type{},
      /*captures=*/captures,
      [=](OpBuilder &builder, Location loc, BlockAndValueMapping &captures) {
        // Clone the target op into the FusedOp.
        auto clonedOp = builder.insert(target->clone(captures));
        // Yield the result of the cloned op (which may have none).
        builder.create<linalg::YieldOp>(
            target->getLoc(),
            clonedOp->getResults());
      });

  // Remove the target operation.
  target->replaceAllUsesWith(fusedOp);
  target->erase();
  return fusedOp;
}

/// Determines whether @p target contains wrapped op that can be unwrapped.
static bool canUnwrapFusedOp(FusedOp target) {
  auto body = target.getBody();

  // - exactly one wrapped op must be contained (plus terminator)
  if (body->getOperations().size() != 2)
    return false;

  // - the terminator must yield the result of that op.
  auto terminator = body->getTerminator();
  auto wrapped = &body->getOperations().front();
  return llvm::equal(terminator->getOperands(), wrapped->getResults());
}

/// Unwraps the single op inside the `linalg.fused` @p target op.
///
/// See canUnwrapFusedOp() on what scenarios this transformation is allowed in.
///
/// @pre      `canUnwrapFusedOp(target)`
///
/// Essentially exactly undoes what wrapInFusedOp() does.
static Operation* unwrapFusedOp(FusedOp target) {
  assert(canUnwrapFusedOp(target));

  // Compute the mapping from captured arguments to operands.
  auto unCaptureMapping = target.getUnCaptureMapping();

  // Clone the wrapped op out of the FusedOp.
  OpBuilder builder(target);
  auto unwrapped = builder.insert(
    target.getBody()->getOperations().front().clone(unCaptureMapping));

  // Remove the target operation
  target.replaceAllUsesWith(unwrapped);
  target.erase();
  return unwrapped;
}

/// Determines whether @p producer is duplicated when prepended to @p target .
static bool mustDuplicateProducer(FusedOp target, Operation *producer) {
  assert(producer);

  // Duplicates if there is one user of a result of producer that is not target.
  return llvm::any_of(
      producer->getResults(),
      [=](Value value) {
        return llvm::any_of(
            value.getUsers(),
            [=](auto user) { return user != target; });
      });
}

/// Determines whether @p producer can be prepended to @p target .
static bool canFuseProducer(FusedOp target, Operation *producer) {
  if (!producer)
    return false;

  // - producer may not be a user of target
  if (llvm::find(target->getUsers(), producer) != target->getUsers().end())
    return false;

  // - if producer will be duplicated, it must not have any side-effects
  return !mustDuplicateProducer(target, producer) || isSideEffectFree(producer);
}

/// Gets the index of the result that @p value has at the defining op.
static unsigned resultIndex(Value value) {
  auto parent = value.getDefiningOp();
  for (unsigned idx = 0; idx < parent->getNumResults(); ++idx) {
    if (parent->getResult(idx) == value)
      return idx;
  }

  llvm_unreachable("value not produced by defining op");
}

/// Fuses @p producer into @p target .
///
/// See canFuseProducer() on what scenarios this transformation is allowed in.
///
/// @pre      `canFuseProducer(target, producer)`
///
/// Given IR of the form:
/// ```mlir
/// %r_0, ..., %r_N = producer %o_0, ...
/// %z = linalg.fused (%x_0 = %y_0, ..., %x_i = %r_i, ...) {
///   ...
/// }
/// ```
///
/// This produces:
/// ```mlir
/// %z = linalg.fused (%x_0 = %y_0, ..., %o_0, ...) {
///   %x_i, ... = producer %o_0, ...
///   ...
/// }
/// ```
///
/// If @p producer had any users beside @p target , a duplicate of the operation
/// will remain. canFuseProducer() therefore checks if this does not cause
/// added side-effects.
static FusedOp fuseProducer(FusedOp target, Operation *producer) {
  assert(canFuseProducer(target, producer));

  // Determine which captures will still be needed and what is dropped.
  SmallVector<Value> newCaptures;
  DenseMap<BlockArgument, unsigned> argToResultIdx;
  for (unsigned idx = 0; idx < target.captures().size(); ++idx) {
    auto capture = target.captures()[idx];
    if (capture.getDefiningOp() == producer) {
      // Capture will be replaced by op result.
      argToResultIdx.try_emplace(
          target.getCaptureArgs()[idx],
          resultIndex(capture));
      continue;
    }

    // Capture is still needed.
    newCaptures.push_back(capture);
  }

  // Ensure all operands to producer are captured.
  collectUses(producer, producer, newCaptures);

  // Create the new FusedOp.
  OpBuilder builder(target);
  auto result = builder.create<FusedOp>(
      target.getLoc(),
      /*resultType=*/target.result() ? target.result().getType() : Type{},
      newCaptures,
      [&](OpBuilder &builder, Location loc, BlockAndValueMapping &captures) {
        // Clone the op to be prepended into the new block.
        auto prepended = builder.insert(producer->clone(captures));

        // Remap the capture arguments of the old FusedOp.
        for (unsigned idx = 0; idx < target.captures().size(); ++idx) {
          captures.map(
              target.getCaptureArgs()[idx],
              captures.lookup(target.captures()[idx]));
        }

        // Update the capture mapping with results of the prepended op.
        for (auto &pair : argToResultIdx) {
          captures.map(pair.getFirst(), prepended->getResult(pair.getSecond()));
        }

        // Clone the contents of the old body.
        for (auto &op : *target.getBody()) {
          builder.insert(op.clone(captures));
        }
      });

  // Remove the target operation.
  target.replaceAllUsesWith(result);
  target.erase();

  // If now unused, remove the producer operation.
  if (producer->getUsers().empty())
    producer->erase();
  return result;
}

/// Determines whether @p consumer can be appended to @p target .
static bool canFuseConsumer(FusedOp target, Operation *consumer) {
  if (!consumer)
    return false;

  // - target may not be a user of consumer
  if (llvm::find(consumer->getUsers(), target) != consumer->getUsers().end())
    return false;

  // - consumer must be the only user of target (if any).
  if (llvm::any_of(
      target->getUsers(),
      [&](auto user) { return user != consumer; }))
    return false;

  // - consumer must be wrappable
  return canWrapInFusedOp(consumer);
}

/// Fused @p consumer into @p target .
///
/// See canFuseConsumer() on what scenarios this transformation is allowed in.
///
/// @pre      `canFuseConsumer(target, consumer)`
///
/// Given IR of the form:
/// ```mlir
/// %z = linalg.fused (%x_0 = %y_0, ...) {
///   ...
/// }
/// %r = consumer %o_0, ..., %z, ...
/// ```
///
/// This produces:
/// ```mlir
/// %r = linalg.fused (%x_0 = %y_0, ..., %o_0, ...) {
///   ...
///   %r = consumer %o_0, ..., %z, ...
///   linalg.yield %r
/// }
/// ```
static FusedOp fuseConsumer(FusedOp target, Operation *consumer) {
  assert(canFuseConsumer(target, consumer));

  // Ensure that all operands of consumer are captured.
  auto newCaptures = llvm::to_vector(target.captures());
  collectUses(consumer, consumer, newCaptures);
  llvm::erase_if(
      newCaptures,
      [&](Value capture) { return capture == target.result(); });

  // Create the new FusedOp.
  OpBuilder builder(consumer);
  auto result = builder.create<FusedOp>(
      target.getLoc(),
      /*resultType=*/consumer->getNumResults()
          ? consumer->getResult(0).getType()
          : Type{},
      newCaptures,
      [&](OpBuilder &builder, Location loc, BlockAndValueMapping &captures) {
        // Remap the capture arguments of the old FusedOp.
        for (unsigned idx = 0; idx < target.captures().size(); ++idx) {
          captures.map(
              target.getCaptureArgs()[idx],
              captures.lookup(target.captures()[idx]));
        }

        // Clone the contents of the old body (without the terminator).
        for (auto &op : target.getBody()->without_terminator()) {
          builder.insert(op.clone(captures));
        }

        // If the old fused op had a result, add it to the mapping.
        auto terminator = target.getBody()->getTerminator();
        if (terminator->getNumOperands())
          captures.map(
              target.result(),
              captures.lookup(terminator->getOperand(0)));

        // Clone the op to be appended to the new block.
        auto appended = builder.insert(consumer->clone(captures));
        // Yield the result of the cloned op (which may have none).
        builder.create<linalg::YieldOp>(
            target->getLoc(),
            appended->getResults());
      });

  // Remove the consumer and then the target operation.
  consumer->replaceAllUsesWith(result);
  consumer->erase();
  target.erase();
  return result;
}

namespace {

struct Tactic {
  enum class Method {
    None = 0,
    Seed,
    FuseProducers,
    FuseProducersGreedy,
    FuseConsumers,
    FuseConsumersGreedy,
    Dissolve
  };

  Method method;
  OperatorClass filter;
};

using Strategy = SmallVector<Tactic>;

static FailureOr<Tactic> parseTactic(StringRef string) {
  const auto symbolize = [](StringRef item) {
    return llvm::StringSwitch<Optional<Tactic::Method>>(item)
        .Case("seed", Tactic::Method::Seed)
        .Case("producers", Tactic::Method::FuseProducers)
        .Case("producers*", Tactic::Method::FuseProducersGreedy)
        .Case("consumers", Tactic::Method::FuseConsumers)
        .Case("consumers*", Tactic::Method::FuseConsumersGreedy)
        .Case("dissolve", Tactic::Method::Dissolve)
        .Default(llvm::None);
  };

  auto [methodStr, tail] = string.split('(');
  if (tail.empty())
    return failure();

  auto method = symbolize(methodStr);
  if (!method.hasValue())
    return failure();

  auto [filterStr, rest] = tail.split(')');
  if (!rest.empty())
    return failure();

  FailureOr<OperatorClass> filter;
  if (filterStr.empty()) {
    filter = OperatorClass::None;
  } else {
    filter = parseOperatorClass(filterStr);
    if (failed(filter))
      return failure();
  }

  return Tactic{method.getValue(), filter.getValue()};
}

static FailureOr<Strategy> parseStrategy(ArrayRef<std::string> tactics) {
  Strategy result;
  for (auto &tactic : tactics) {
    auto parsed = parseTactic(tactic);
    if (failed(parsed))
      return failure();
    result.push_back(parsed.getValue());
  }
  return std::move(result);
}

struct LinalgStructuralFusionPass
    : public LinalgStructuralFusionBase<LinalgStructuralFusionPass> {
  void runOnOperation() override {
    if (!strategy) {
      if (!strategyString.empty()) {
        auto parsed = parseStrategy(strategyString);
        if (failed(parsed)) {
          llvm::WithColor::error()
            << "[linalg-structural-fusion] error parsing strategy\n";
          signalPassFailure();
          return;
        }

        strategy = std::move(parsed.getValue());
      } else {
        static Strategy defaultStrategy;
        if (defaultStrategy.empty()) {
          defaultStrategy.push_back(
            Tactic{Tactic::Method::Seed, OperatorClass::Convolution}
          );
          defaultStrategy.push_back(
            Tactic{Tactic::Method::FuseConsumers, OperatorClass::Activation}
          );
          defaultStrategy.push_back(
            Tactic{Tactic::Method::Dissolve, OperatorClass{}}
          );
          defaultStrategy.push_back(
            Tactic{Tactic::Method::FuseProducers, OperatorClass::Broadcast}
          );
          defaultStrategy.push_back(
            Tactic{Tactic::Method::FuseConsumers, OperatorClass::Padding}
          );
          defaultStrategy.push_back(
            Tactic{Tactic::Method::FuseConsumers, OperatorClass::Pooling}
          );
        }

        strategy = defaultStrategy;
      }
    }

    apply(strategy.getValue());
  }

private:
  Optional<Strategy> strategy;

  using WorkingSet = llvm::SmallVector<FusedOp>;

  size_t seed(WorkingSet &work, OperatorClass filter) {
    using llvm::operator&;

    size_t result = 0;
    work.clear();
    getOperation().walk([&](Operation *op, const WalkStage &stage) {
      if (isa<FusedOp>(op)) {
        // Do not look into fused ops.
        return WalkResult::skip();
      }

      if (matches(op, filter)) {
        // Seed this op, but do not look into it.
        if (canWrapInFusedOp(op)) {
          work.push_back(wrapInFusedOp(op));
          ++result;
        }
        return WalkResult::skip();
      }

      return WalkResult::advance();
    });

    return result;
  }

  FailureOr<FusedOp> fuseFirstProducer(FusedOp target, OperatorClass filter) {
    for (auto op : target.getOperands()) {
      auto candidate = op.getDefiningOp();
      if (!candidate
          || !matches(candidate, filter)
          || !canFuseProducer(target, candidate))
        continue;

      return ::fuseProducer(target, candidate);
    }

    return failure();
  }

  FailureOr<FusedOp> fuseProducers(FusedOp target, OperatorClass filter) {
    auto result = false;
    while (true) {
      auto fused = fuseFirstProducer(target, filter);
      if (succeeded(fused)) {
        target = fused.getValue();
        result = true;
        continue;
      }

      if (result)
        return target;

      return failure();
    }
  }

  size_t fuseProducers(WorkingSet &work, OperatorClass filter) {
    WorkingSet results;
    llvm::erase_if(
        work,
        [&](FusedOp op) {
          auto fused = fuseProducers(op, filter);
          if (failed(fused))
            return false;

          results.push_back(fused.getValue());
          return true;
        });
    work.append(results);
    return results.size();
  }

  size_t fuseProducersGreedy(WorkingSet &work, OperatorClass filter) {
    size_t result = 0;
    while (auto add = fuseProducers(work, filter)) {
      result += add;
    }
    return result;
  }

  FailureOr<FusedOp> fuseConsumer(FusedOp target, OperatorClass filter) {
    if (!target.result())
      return failure();

    assert(!target.result().getUsers().empty());
    Operation *candidate = *target.result().getUsers().begin();
    if (!matches(candidate, filter) || !canFuseConsumer(target, candidate))
      return failure();

    return ::fuseConsumer(target, candidate);
  }

  size_t fuseConsumers(WorkingSet &work, OperatorClass filter) {
    WorkingSet results;
    llvm::erase_if(
        work,
        [&](FusedOp op) {
          auto fused = fuseConsumer(op, filter);
          if (failed(fused))
            return false;

          results.push_back(fused.getValue());
          return true;
        });
    work.append(results);
    return results.size();
  }

  size_t fuseConsumersGreedy(WorkingSet &work, OperatorClass filter) {
    size_t result = 0;
    while (auto add = fuseConsumers(work, filter)) {
      result += add;
    }
    return result;
  }

  size_t dissolve(WorkingSet &work) {
    size_t result = 0;
    llvm::erase_if(
        work,
        [&](FusedOp op) {
          if (!canUnwrapFusedOp(op))
            return false;

          unwrapFusedOp(op);
          ++result;
          return true;
        });
    return result;
  }

  size_t apply(WorkingSet &work, const Tactic &tactic) {
    switch (tactic.method) {
    case Tactic::Method::Seed:
      return seed(work, tactic.filter);
    case Tactic::Method::FuseProducers:
      return fuseProducers(work, tactic.filter);
    case Tactic::Method::FuseProducersGreedy:
      return fuseProducersGreedy(work, tactic.filter);
    case Tactic::Method::FuseConsumers:
      return fuseConsumers(work, tactic.filter);
    case Tactic::Method::FuseConsumersGreedy:
      return fuseConsumersGreedy(work, tactic.filter);
    case Tactic::Method::Dissolve:
      return dissolve(work);
    default: llvm_unreachable("unknown tactic");
    }
  }

  size_t apply(WorkingSet &work, const Strategy &strategy) {
    size_t result = 0;
    for (auto &tactic : strategy) {
      result += apply(work, tactic);
    }
    return result;
  }

  size_t apply(const Strategy &strategy) {
    WorkingSet work;
    return apply(work, strategy);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgStructuralFusionPass() {
  return std::make_unique<LinalgStructuralFusionPass>();
}

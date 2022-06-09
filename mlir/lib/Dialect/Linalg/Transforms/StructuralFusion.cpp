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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-structural-fusion"

using namespace mlir;
using namespace mlir::linalg;

namespace {

struct LinalgStructuralFusionPass
    : public LinalgStructuralFusionBase<LinalgStructuralFusionPass> {
    void runOnOperation() override
    {
        // TODO: Implement.
    }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgStructuralFusionPass() {
    return std::make_unique<LinalgStructuralFusionPass>();
}

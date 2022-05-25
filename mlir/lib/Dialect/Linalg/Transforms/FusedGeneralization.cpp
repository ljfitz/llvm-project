//===- FusedGeneralization.cpp - linalg fused ops to generic ops  ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg fused generalization pass. It converts named
// Linalg fused ops to linalg.generic ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-fused-generalization"

using namespace mlir;
using namespace mlir::linalg;

namespace {

struct Conv2DReluLowering : OpRewritePattern<Conv2DReluOp> {
    using OpRewritePattern<Conv2DReluOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
        Conv2DReluOp op, 
        PatternRewriter &rewriter
    ) const override 
    {
        return failure();
    }
};

struct Conv2DLreluOpLowering : OpRewritePattern<Conv2DLreluOp> {
    using OpRewritePattern<Conv2DLreluOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
        Conv2DLreluOp op, 
        PatternRewriter &rewriter
    ) const override 
    {
        return failure();
    }
};

struct Conv2DLreluMaxpoolOpLowering : OpRewritePattern<Conv2DLreluMaxpoolOp> {
    using OpRewritePattern<Conv2DLreluMaxpoolOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
        Conv2DLreluMaxpoolOp op, 
        PatternRewriter &rewriter
    ) const override 
    {    
        return failure();
    }
};

struct LinalgFusedGeneralizationPass 
    : public LinalgFusedGeneralizationBase<LinalgFusedGeneralizationPass> {
    void runOnOperation() override 
    {
        RewritePatternSet patterns(&getContext());
        
        patterns.add<
            Conv2DReluLowering, 
            Conv2DLreluOpLowering, 
            Conv2DLreluMaxpoolOpLowering
        >(&getContext());

        (void)applyPatternsAndFoldGreedily(
            getOperation().getBody(), 
            std::move(patterns)
        );
    }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgFusedGeneralizationPass() {
    return std::make_unique<LinalgFusedGeneralizationPass>();
}

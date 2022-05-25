//===- Unfuse.cpp - linalg fused ops to simple ops  -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg unfuse pass. It converts the named fused
// ops into simple ones with generalizable lowering.
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

#define DEBUG_TYPE "linalg-unfuse"

using namespace mlir;
using namespace mlir::linalg;

namespace {

Value applyBias(Location loc, Value target, Value bias, OpBuilder &builder)
{
    auto targetTy = target.getType().cast<RankedTensorType>();
    auto biasTy = bias.getType().cast<RankedTensorType>();

    // target: NxFxHOxWO of T, bias: F of T
    assert(targetTy.getRank() == 4 && biasTy.getRank() == 1);
    assert(targetTy.getElementType() == biasTy.getElementType());
    assert(targetTy.getShape()[1] == biasTy.getShape()[0]);

    Value inputs[] = {
        /*IFM=*/target,
        /*bias=*/bias
    };
    return builder.create<ApplyBias2DFchwOp>(
        /*location=*/loc,
        /*resultTensorTypes=*/targetTy,
        /*inputs=*/inputs,
        /*outputs=*/target
    ).getResult(0);
}

struct Conv2DReluLowering : OpRewritePattern<Conv2DReluOp> {
    using OpRewritePattern<Conv2DReluOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
        Conv2DReluOp op,
        PatternRewriter &rewriter
    ) const override
    {
        assert(op.getNumInputs() == 3);

        // 1. Add the bias to the destination tensor.
        auto biasedDest = applyBias(
            op.getLoc(),
            op.getOutputOperand(0)->get(),
            op.getInputOperand(2)->get(),
            rewriter
        );

        // 2. Perform the regular convolution.
        Value convResult;
        {
            Value inputs[] = {
                /*I=*/op.getInputOperand(0)->get(),
                /*K=*/op.getInputOperand(1)->get()
            };
            NamedAttribute attributes[] = {
                rewriter.getNamedAttr("dilation", op.dilationAttr()),
                rewriter.getNamedAttr("stride", op.strideAttr())
            };
            convResult = rewriter.create<Conv2DNchwFchwOp>(
                op.getLoc(),
                /*resultTensorTypes=*/op.getOutputOperand(0)->get().getType(),
                /*inputs=*/inputs,
                /*outputs=*/biasedDest,
                /*attributes=*/attributes
            ).getResult(0);
        }

        // 3. Perform the activation.
        rewriter.replaceOpWithNewOp<Relu2DNchwOp>(
            op,
            /*resultTensorTypes=*/op.getOutputOperand(0)->get().getType(),
            /*inputs=*/convResult,
            /*outputs=*/convResult
        );

        return success();
    }
};

struct Conv2DLreluOpLowering : OpRewritePattern<Conv2DLreluOp> {
    using OpRewritePattern<Conv2DLreluOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
        Conv2DLreluOp op,
        PatternRewriter &rewriter
    ) const override
    {
        assert(op.getNumInputs() == 4);

        // 1. Add the bias to the destination tensor.
        auto biasedDest = applyBias(
            op.getLoc(),
            op.getOutputOperand(0)->get(),
            op.getInputOperand(2)->get(),
            rewriter
        );

        // 2. Perform the regular convolution.
        Value convResult;
        {
            Value inputs[] = {
                /*I=*/op.getInputOperand(0)->get(),
                /*K=*/op.getInputOperand(1)->get()
            };
            NamedAttribute attributes[] = {
                rewriter.getNamedAttr("dilation", op.dilationAttr()),
                rewriter.getNamedAttr("stride", op.strideAttr())
            };
            convResult = rewriter.create<Conv2DNchwFchwOp>(
                op.getLoc(),
                /*resultTensorTypes=*/op.getOutputOperand(0)->get().getType(),
                /*inputs=*/inputs,
                /*outputs=*/biasedDest,
                /*attributes=*/attributes
            ).getResult(0);
        }

        // 3. Perform the activation.
        {
            Value inputs[] = {
                /*ifm=*/convResult,
                /*alpha=*/op.getInputOperand(3)->get()
            };
            rewriter.replaceOpWithNewOp<Lrelu2DNchwOp>(
                op,
                /*resultTensorTypes=*/op.getOutputOperand(0)->get().getType(),
                /*inputs=*/inputs,
                /*outputs=*/convResult
            );
        }

        return success();
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

struct LinalgUnfusePass
    : public LinalgUnfuseBase<LinalgUnfusePass> {
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

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgUnfusePass() {
    return std::make_unique<LinalgUnfusePass>();
}

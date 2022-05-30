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
#include "mlir/Dialect/Tensor/Utils/Utils.h"
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

/// Obtain exactly N values from a DenseIntElementsAttr.
template<std::size_t N>
std::array<int64_t, N> getValues(DenseIntElementsAttr attr)
{
    // Sanity checks: this will actually produce two integers.
    auto attrTy = attr.getType().dyn_cast<ShapedType>();
    assert(
        attrTy && attrTy.getRank() == 1 && attrTy.getShape()[0] == N
        && attrTy.getElementType().isa<IntegerType>()
        && "expected Nxint"
    );

    std::array<int64_t, N> result;
    if (attr.isSplat()) {
        std::fill(result.begin(), result.end(), attr.getSplatValue<int64_t>());
    } else {
        llvm::copy(attr.getValues<int64_t>(), result.begin());
    }
    return result;
}

/// Given the convolution parameter types and attributes, compute the shape of
/// the result.
RankedTensorType infer2DConvolutionResultType(
    RankedTensorType input,
    RankedTensorType weights,
    DenseIntElementsAttr dilationAttr,
    DenseIntElementsAttr strideAttr
)
{
    // Sanity checks: these are actually valid parameters for a convolution.
    assert(input.getRank() == 4 && "expected NCHW");
    assert(weights.getRank() == 4 && "expected FCHW");
    assert(input.getShape()[1] == weights.getShape()[1] && "mismatched C");
    assert(
        input.getElementType() == weights.getElementType()
        && "mismatched T"
    );

    auto dilation = ::getValues<2>(dilationAttr);
    auto stride = ::getValues<2>(strideAttr);
    // Sanity checks: non-positive dilations and strides are not allowed.
    assert(llvm::all_of(dilation, [](auto x) { return x > 0; }));
    assert(llvm::all_of(stride, [](auto x) { return x > 0; }));

    // Compute result shape according to:
    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    std::array<int64_t, 4> resultShape = {
        /*N=*/input.getShape()[0],
        /*F=*/weights.getShape()[0],
        /*H=*/(input.getShape()[3] - dilation[1]*(weights.getShape()[3] - 1) - 1) / stride[1] + 1,
        /*W=*/(input.getShape()[2] - dilation[0]*(weights.getShape()[2] - 1) - 1) / stride[0] + 1
    };

    return RankedTensorType::get(resultShape, input.getElementType());
}

/// Unfuse the convolution part of @p op .
Value unfuse2DConvolution(
    OpBuilder &builder,
    Operation* op,
    Value ifm,
    Value weights,
    Value bias,
    Value dest = {}
)
{
    // Sanity check: none of these are optional.
    assert(ifm && weights && bias && "expected 3 operands");

    // Infer the result shape of the convolution.
    auto dilationAttr = op->getAttr("dilation").cast<DenseIntElementsAttr>();
    auto strideAttr = op->getAttr("stride").cast<DenseIntElementsAttr>();
    auto resultTy = infer2DConvolutionResultType(
        ifm.getType().cast<RankedTensorType>(),
        weights.getType().cast<RankedTensorType>(),
        dilationAttr,
        strideAttr
    );

    // Ensure we have an appropriately sized destination operand.
    if (!dest) {
        auto elementTy = ifm.getType().cast<ShapedType>().getElementType();
        auto zero = builder.create<arith::ConstantOp>(
            op->getLoc(),
            builder.getZeroAttr(elementTy)
        );
        dest = builder.create<tensor::SplatOp>(
            op->getLoc(),
            zero,
            resultTy
        ).getResult();
    }
    assert(dest.getType() == resultTy && "expected matching dest");

    // Apply the bias to dest.
    //  - We do this before the convolution, since there may be opportunities
    //    for constant folding / other optimizations.
    {
        Value inputs[] = { /*IFM=*/dest, /*bias=*/bias };
        dest = builder.create<ApplyBias2DFchwOp>(
            op->getLoc(),
            /*resultTensorTypes=*/resultTy,
            inputs,
            /*outputs=*/dest
        ).getResult(0);
    }

    // Insert the convolution operation.
    {
        Value inputs[] = { /*I=*/ifm, /*K=*/weights };
        NamedAttribute attributes[] = {
            builder.getNamedAttr("dilation", dilationAttr),
            builder.getNamedAttr("stride", strideAttr)
        };
        return builder.create<Conv2DNchwFchwOp>(
            op->getLoc(),
            /*resultTensorTypes=*/resultTy,
            inputs,
            /*outputs=*/dest,
            attributes
        ).getResult(0);
    }
}

struct Conv2DReluLowering : OpRewritePattern<Conv2DReluOp> {
    using OpRewritePattern<Conv2DReluOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
        Conv2DReluOp op,
        PatternRewriter &rewriter
    ) const override
    {
        // Sanity check: number of operands, none are optional!
        assert(op.getNumInputs() == 3 && "expected 3 inputs");
        assert(op.getNumOutputs() == 1 && "expected 1 output");

        // Unfuse the convolution.
        auto convResult = unfuse2DConvolution(
            rewriter,
            op,
            /*ifm=*/op.getInputOperand(0)->get(),
            /*weights=*/op.getInputOperand(1)->get(),
            /*bias=*/op.getInputOperand(2)->get(),
            /*dest=*/op.getOutputOperand(0)->get()
        );

        // Unfuse the ReLU.
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
        // Sanity check: number of operands, none are optional!
        assert(op.getNumInputs() == 4 && "expected 4 inputs");
        assert(op.getNumOutputs() == 1 && "expected 1 output");

        // Unfuse the convolution.
        auto convResult = unfuse2DConvolution(
            rewriter,
            op,
            /*ifm=*/op.getInputOperand(0)->get(),
            /*weights=*/op.getInputOperand(1)->get(),
            /*bias=*/op.getInputOperand(2)->get(),
            /*dest=*/op.getOutputOperand(0)->get()
        );

        // Unfuse the leaky ReLU.
        {
            Value inputs[] = {
                /*ifm=*/convResult,
                /*alpha=*/op.getInputOperand(3)->get()
            };
            rewriter.replaceOpWithNewOp<Lrelu2DNchwOp>(
                op,
                /*resultTensorTypes=*/convResult.getType(),
                inputs,
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
        // Sanity check: number of operands, none are optional!
        assert(op.getNumInputs() == 4 && "expected 4 inputs");
        assert(op.getNumOutputs() == 1 && "expected 1 output");

        // Unfuse the convolution.
        auto convResult = unfuse2DConvolution(
            rewriter,
            op,
            /*ifm=*/op.getInputOperand(0)->get(),
            /*weights=*/op.getInputOperand(1)->get(),
            /*bias=*/op.getInputOperand(2)->get()
        );

        // Unfuse the leaky ReLU.
        Value lreluResult;
        {
            Value inputs[] = {
                /*ifm=*/convResult,
                /*alpha=*/op.getInputOperand(3)->get()
            };
            lreluResult = rewriter.create<Lrelu2DNchwOp>(
                op->getLoc(),
                /*resultTensorTypes=*/convResult.getType(),
                inputs,
                /*outputs=*/convResult
            ).getResult(0);
        }

        // Unfuse the padding.
        Value padded = lreluResult;
        auto padding = ::getValues<4>(op.mp_padding());
        auto elementTy = lreluResult.getType()
            .cast<ShapedType>()
            .getElementType();
        if (llvm::any_of(padding, [](auto x) { return x != 0; })) {
            // Create the tensor.pad op.
            // TODO: Check the padding order.
            int64_t padLow[] = { 0, 0, padding[2], padding[0] };
            int64_t padHigh[] = { 0, 0, padding[3], padding[1] };
            auto padOp = rewriter.create<tensor::PadOp>(
                op.getLoc(),
                lreluResult,
                padLow,
                padHigh,
                ValueRange(),
                ValueRange()
            );

            // Generate a padding value.
            // BUG: xten -> linalg lowering does not pass along the padding
            //      value! But maxpool is ususally padded with -inf.
            Attribute padValue;
            if (auto floatTy = elementTy.cast<FloatType>()) {
                padValue = rewriter.getFloatAttr(
                    floatTy,
                    APFloat::getInf(floatTy.getFloatSemantics(), true)
                );
            } else if (auto intTy = elementTy.cast<IntegerType>()) {
                padValue = rewriter.getIntegerAttr(
                    intTy,
                    intTy.isSigned()
                        ? APInt::getSignedMinValue(intTy.getWidth()) 
                        : APInt::getMinValue(intTy.getWidth())
                );
            } else {
                llvm_unreachable("unsupported element type");
            }

            // Insert the body.
            OpBuilder::InsertionGuard guard(rewriter);
            auto &region = padOp.region();
            SmallVector<Type, 4> argTypes(4, rewriter.getIndexType());
            SmallVector<Location, 4> argLocs(4, op.getLoc());
            rewriter.createBlock(&region, region.end(), argTypes, argLocs);
            rewriter.create<tensor::YieldOp>(
                op.getLoc(),
                rewriter.create<arith::ConstantOp>(
                    op.getLoc(),
                    padValue
                )
            );

            padded = padOp.getResult();
        }

        // Unfuse the maxpool.
        {
            auto poolSize = ::getValues<2>(op.mp_kernel_size());
            Value inputs[] = {
                /*I=*/padded,
                /*K=*/rewriter.create<InitTensorOp>(
                    op.getLoc(),
                    poolSize,
                    elementTy
                )
            };
            NamedAttribute attributes[] = {
                rewriter.getNamedAttr("dilation", op->getAttr("mp_dilation")),
                rewriter.getNamedAttr("stride", op->getAttr("mp_stride"))
            };
            rewriter.replaceOpWithNewOp<PoolingNchwMaxOp>(
                op,
                /*resultTensorTypes=*/op.getOutputOperand(0)->get().getType(),
                inputs,
                /*outputs=*/op.getOutputOperand(0)->get(),
                /*attributes=*/attributes
            );
        }

        return success();
    }
};

struct LinalgUnfusePass
    : public LinalgUnfuseBase<LinalgUnfusePass> {
    void runOnOperation() override
    {
        RewritePatternSet patterns(&getContext());

        // BUG: Utterly fails for dynamic dimensions. Maybe reject patterns?

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

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLinalgUnfusePass() {
    return std::make_unique<LinalgUnfusePass>();
}

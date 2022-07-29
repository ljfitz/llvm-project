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
#include "mlir/Dialect/Math/IR/Math.h"
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
    Value bias)
{
    // Sanity check: none of these are optional.
    assert(ifm && weights && bias && "expected 3 operands");

    // Infer the result shape of the convolution.
    auto dilationAttr = op->getAttr("dilations").cast<DenseIntElementsAttr>();
    auto strideAttr = op->getAttr("strides").cast<DenseIntElementsAttr>();
    auto resultTy = infer2DConvolutionResultType(
        ifm.getType().cast<RankedTensorType>(),
        weights.getType().cast<RankedTensorType>(),
        dilationAttr,
        strideAttr
    );

    // Ensure we have an appropriately sized destination operand.
    Value dest = builder.create<InitTensorOp>(op->getLoc(), resultTy.getShape(), resultTy.getElementType());

    // Apply the bias to dest.
    //  - We do this before the convolution, since we assume this is the canonical evaluation order.
    {
        Value inputs[] = { /*bias=*/bias };
        dest = builder.create<BroadcastBias2DFchwOp>(
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
            builder.getNamedAttr("dilations", dilationAttr),
            builder.getNamedAttr("strides", strideAttr)
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

struct Conv2DTensorAddLowering : OpRewritePattern<Conv2DTensorAddOp> {
    using OpRewritePattern<Conv2DTensorAddOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
        Conv2DTensorAddOp op,
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
            /*weights=*/op.getInputOperand(2)->get(),
            /*bias=*/op.getInputOperand(3)->get()
        );
        // Unfuse the Add.
        rewriter.replaceOpWithNewOp<arith::AddFOp>(
            op,
            convResult,
            op.getInputOperand(1)->get()
        );

        return success();
    }
};

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
            /*bias=*/op.getInputOperand(2)->get()
        );

        // Unfuse the ReLU.
        rewriter.replaceOpWithNewOp<Relu2DNchwOp>(
            op,
            /*resultTensorTypes=*/convResult.getType(),
            /*inputs=*/convResult,
            /*outputs=*/convResult
        );

        return success();
    }
};

struct Conv2DTensorAddReluLowering : OpRewritePattern<Conv2DTensorAddReluOp> {
    using OpRewritePattern<Conv2DTensorAddReluOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
        Conv2DTensorAddReluOp op,
        PatternRewriter &rewriter
    ) const override
    {
        // Sanity check: number of operands, none are optional!
        assert(op.getNumInputs() == 4 && "expected 4 inputs");
        assert(op.getNumOutputs() == 1 && "expected 1 output");

        // Unfuse the convolution and use the summand directly as destination.
        auto convResult = unfuse2DConvolution(
            rewriter,
            op,
            /*ifm=*/op.getInputOperand(0)->get(),
            /*weights=*/op.getInputOperand(2)->get(),
            /*bias=*/op.getInputOperand(3)->get()
        );

        // Unfuse the Add.
        auto addResult = rewriter.create<arith::AddFOp>(
            op->getLoc(),
            convResult,
            op.getInputOperand(1)->get()
        ).getResult();

        // Unfuse the ReLU.
        rewriter.replaceOpWithNewOp<Relu2DNchwOp>(
            op,
            /*resultTensorTypes=*/addResult.getType(),
            /*inputs=*/addResult,
            /*outputs=*/addResult
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
            /*bias=*/op.getInputOperand(2)->get()
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

struct Conv2DTensorAddLreluLowering : OpRewritePattern<Conv2DTensorAddLreluOp> {
    using OpRewritePattern<Conv2DTensorAddLreluOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
        Conv2DTensorAddLreluOp op,
        PatternRewriter &rewriter
    ) const override
    {
        // Sanity check: number of operands, none are optional!
        assert(op.getNumInputs() == 5 && "expected 5 inputs");
        assert(op.getNumOutputs() == 1 && "expected 1 output");

        // Unfuse the convolution and use the summand directly as destination.
        auto convResult = unfuse2DConvolution(
            rewriter,
            op,
            /*ifm=*/op.getInputOperand(0)->get(),
            /*weights=*/op.getInputOperand(2)->get(),
            /*bias=*/op.getInputOperand(3)->get()
        );

        // Unfuse the Add.
        auto addResult = rewriter.create<arith::AddFOp>(
            op->getLoc(),
            convResult,
            op.getInputOperand(1)->get()
        ).getResult();

        // Unfuse the leaky ReLU.
        {
            Value inputs[] = {
                /*ifm=*/addResult,
                /*alpha=*/op.getInputOperand(4)->get()
            };
            rewriter.replaceOpWithNewOp<Lrelu2DNchwOp>(
                op,
                /*resultTensorTypes=*/addResult.getType(),
                inputs,
                /*outputs=*/addResult
            );
        }

        return success();
    }
};

void sanityCheckInputs(Conv2DLreluMaxpoolOp op) {
  // Sanity check: number of operands, none are optional!
  assert(op.getNumInputs() == 4 && "expected 4 inputs");
}

void sanityCheckInputs(Conv2DReluMaxpoolOp op) {
  // Sanity check: number of operands, none are optional!
  assert(op.getNumInputs() == 3 && "expected 3 inputs");
}

Value unfuseActivation(Conv2DLreluMaxpoolOp op, Value ifm,
                       PatternRewriter &rewriter) {
  Value inputs[] = {/*ifm=*/ifm,
                    /*alpha=*/op.getInputOperand(3)->get()};
  return rewriter
      .create<Lrelu2DNchwOp>(op->getLoc(),
                             /*resultTensorTypes=*/ifm.getType(), inputs,
                             /*outputs=*/ifm)
      .getResult(0);
}

Value unfuseActivation(Conv2DReluMaxpoolOp op, Value ifm,
                       PatternRewriter &rewriter) {
  Value inputs[] = {/*ifm=*/ifm};
  return rewriter
      .create<Relu2DNchwOp>(op->getLoc(),
                            /*resultTensorTypes=*/ifm.getType(), inputs,
                            /*outputs=*/ifm)
      .getResult(0);
}

template<class T>
struct Conv2DActivationMaxpoolOpLowering : OpRewritePattern<T> {
    using OpRewritePattern<T>::OpRewritePattern;
    LogicalResult matchAndRewrite(
        T op,
        PatternRewriter &rewriter
    ) const override
    {
        sanityCheckInputs(op);
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
        Value lreluResult = unfuseActivation(op, convResult, rewriter);

        // Unfuse the padding.
        Value padded = lreluResult;
        auto padding = ::getValues<4>(op.mp_padding());
        auto elementTy = lreluResult.getType()
            .cast<ShapedType>()
            .getElementType();
        if (llvm::any_of(padding, [](auto x) { return x != 0; })) {
            // Create the tensor.pad op.
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
            //      value! But maxpool is usually padded with -inf.
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
                rewriter.getNamedAttr("dilations", op->getAttr("mp_dilations")),
                rewriter.getNamedAttr("strides", op->getAttr("mp_strides"))
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

Value sumKeepDim(OpBuilder &builder, Location loc, Value in, unsigned dim)
{
    assert(in);
    auto inTy = in.getType().dyn_cast<RankedTensorType>();
    assert(inTy && "expected tensor operand");
    const auto rank = static_cast<unsigned>(inTy.getRank());
    assert(dim < rank && "expected valid dim index");

    // Compute the result type.
    auto sumShape = llvm::to_vector(inTy.getShape());
    sumShape[dim] = 1;
    auto sumTy = RankedTensorType::get(sumShape, inTy.getElementType());

    // Initialize the reduction accumulator.
    Value accu;
    {
        auto zero = builder.create<arith::ConstantOp>(
            loc,
            builder.getZeroAttr(inTy.getElementType())
        );
        accu = builder.create<tensor::SplatOp>(loc, zero, sumTy).getResult(); 
    }

    // Compute the indexing maps.
    // inMap ::= (o_0, ..., o_r, r) -> (o_0, ..., r, ..., o_r)
    // outMap ::= (o_0, ..., o_r, r) -> (o_0, ..., 0, ..., o_r)
    AffineMap inMap, outMap;
    {
        SmallVector<AffineExpr> inBuilder, outBuilder;
        for (auto [idx, inIdx] = std::make_pair(0U, 0U); idx < rank; ++idx) {
            if (idx == dim) {
                inBuilder.push_back(
                    getAffineDimExpr(rank - 1, builder.getContext())
                );
                outBuilder.push_back(
                    getAffineConstantExpr(0, builder.getContext())
                );
                continue;
            }

            auto inDim = getAffineDimExpr(inIdx++, builder.getContext());
            inBuilder.push_back(inDim);
            outBuilder.push_back(inDim);
        }
        inMap = AffineMap::get(rank, 0, inBuilder, builder.getContext());
        outMap = AffineMap::get(rank, 0, outBuilder, builder.getContext());
    }
    AffineMap indexingMaps[] = { inMap, outMap };

    // Compute the iterator types.
    SmallVector<StringRef> iteratorTypes(rank, "parallel");
    iteratorTypes.back() = "reduction";

    return builder.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/sumTy,
        /*inputs=*/in, 
        /*outputs=*/accu, 
        indexingMaps, 
        iteratorTypes, 
        [](OpBuilder &builder, Location loc, ValueRange args) {
            auto add = builder.create<arith::AddFOp>(
                loc,
                args[0],
                args[1]
            ).getResult();
            builder.create<linalg::YieldOp>(loc, add);
        }
    ).getResult(0);
}

struct SoftmaxLowering : OpRewritePattern<SoftmaxOp> {
    using OpRewritePattern<SoftmaxOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(
        SoftmaxOp op,
        PatternRewriter &rewriter
    ) const override
    {
        auto in = op.getOperand();
        auto inTy = in.getType().dyn_cast<RankedTensorType>();
        assert(inTy && "expected tensor operand");
        
        auto dim = op.dimAttr().getValue().getSExtValue();
        dim = dim < 0 ? dim + inTy.getRank() : dim;
        assert(dim < inTy.getRank() && "expected valid dim index");

        // exp(x)
        auto exp = rewriter.create<math::ExpOp>(op.getLoc(), in).getResult();

        // sum(exp(x), dim, keepDim=true)
        auto sum = sumKeepDim(rewriter, op.getLoc(), exp, static_cast<unsigned>(dim));

        // exp(x) / sum(exp(x), dim, keepDim=true)
        {
            const auto rank = static_cast<unsigned>(inTy.getRank());
            // Compute the indexing maps.
            // inMap ::= (o_0, ..., r, ..., o_r) -> (o_0, ..., 0, ..., o_r)
            // outMap ::= (o_0, ..., r, ..., o_r) -> (o_0, ..., r, ..., o_r)
            AffineMap inMap;
            {
                SmallVector<AffineExpr> builder;
                for (unsigned idx = 0; idx < rank; ++idx) {
                    builder.push_back(
                        idx == static_cast<unsigned>(dim)
                        ? getAffineConstantExpr(0, rewriter.getContext())
                        : getAffineDimExpr(idx, rewriter.getContext())
                    );
                }
                inMap = AffineMap::get(rank, 0, builder, rewriter.getContext());
            }
            AffineMap indexingMaps[] = { 
                inMap, 
                AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
            };

            // Compute the iterator types.
            SmallVector<StringRef> iteratorTypes(rank, "parallel");

            rewriter.replaceOpWithNewOp<linalg::GenericOp>(
                op,
                /*resultTensorTypes=*/inTy,
                /*inputs=*/sum, 
                /*outputs=*/exp, 
                indexingMaps, 
                iteratorTypes, 
                [](OpBuilder &builder, Location loc, ValueRange args) {
                    auto div = builder.create<arith::DivFOp>(
                        loc,
                        args[0],
                        args[1]
                    ).getResult();
                    builder.create<linalg::YieldOp>(loc, div);
                }
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
            Conv2DTensorAddLowering,
            Conv2DReluLowering,
            Conv2DTensorAddReluLowering,
            Conv2DLreluOpLowering,
            Conv2DTensorAddLreluLowering,
            Conv2DActivationMaxpoolOpLowering<Conv2DLreluMaxpoolOp>,
            Conv2DActivationMaxpoolOpLowering<Conv2DReluMaxpoolOp>,
            SoftmaxLowering
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

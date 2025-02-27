; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; This is the loop in c++ being vectorize in this file with
;experimental.vector.reverse
;  #pragma clang loop vectorize_width(8, scalable)
;  for (int i = N-1; i >= 0; --i)
;    a[i] = b[i] + 1.0;

; RUN: opt -passes=loop-vectorize,dce,instcombine -mtriple aarch64-linux-gnu -S \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue < %s | FileCheck %s

define void @vector_reverse_f64(i64 %N, ptr noalias %a, ptr noalias %b) #0{
; CHECK-LABEL: @vector_reverse_f64(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP7:%.*]] = icmp sgt i64 [[N:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP7]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_COND_CLEANUP:%.*]]
; CHECK:       for.body.preheader:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = shl i64 [[TMP0]], 3
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ugt i64 [[TMP1]], [[N]]
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = shl i64 [[TMP2]], 3
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N]], [[TMP3]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N]], [[N_MOD_VF]]
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP4:%.*]] = xor i64 [[INDEX]], -1
; CHECK-NEXT:    [[TMP5:%.*]] = add i64 [[TMP4]], [[N]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds double, ptr [[B:%.*]], i64 [[TMP5]]
; CHECK-NEXT:    [[TMP7:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP8:%.*]] = shl i64 [[TMP7]], 3
; CHECK-NEXT:    [[TMP9:%.*]] = sub i64 1, [[TMP8]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds double, ptr [[TMP6]], i64 [[TMP9]]
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x double>, ptr [[TMP10]], align 8
; CHECK-NEXT:    [[TMP11:%.*]] = fadd <vscale x 8 x double> [[WIDE_LOAD]], shufflevector (<vscale x 8 x double> insertelement (<vscale x 8 x double> poison, double 1.000000e+00, i64 0), <vscale x 8 x double> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds double, ptr [[A:%.*]], i64 [[TMP5]]
; CHECK-NEXT:    [[TMP13:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP14:%.*]] = shl i64 [[TMP13]], 3
; CHECK-NEXT:    [[TMP15:%.*]] = sub i64 1, [[TMP14]]
; CHECK-NEXT:    [[TMP16:%.*]] = getelementptr inbounds double, ptr [[TMP12]], i64 [[TMP15]]
; CHECK-NEXT:    store <vscale x 8 x double> [[TMP11]], ptr [[TMP16]], align 8
; CHECK-NEXT:    [[TMP17:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP18:%.*]] = shl i64 [[TMP17]], 3
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP18]]
; CHECK-NEXT:    [[TMP19:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP19]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N_MOD_VF]], 0
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_COND_CLEANUP_LOOPEXIT:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_MOD_VF]], [[MIDDLE_BLOCK]] ], [ [[N]], [[FOR_BODY_PREHEADER]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.cond.cleanup.loopexit:
; CHECK-NEXT:    br label [[FOR_COND_CLEANUP]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
; CHECK:       for.body:
; CHECK-NEXT:    [[I_08_IN:%.*]] = phi i64 [ [[I_08:%.*]], [[FOR_BODY]] ], [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ]
; CHECK-NEXT:    [[I_08]] = add nsw i64 [[I_08_IN]], -1
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds double, ptr [[B]], i64 [[I_08]]
; CHECK-NEXT:    [[TMP20:%.*]] = load double, ptr [[ARRAYIDX]], align 8
; CHECK-NEXT:    [[ADD:%.*]] = fadd double [[TMP20]], 1.000000e+00
; CHECK-NEXT:    [[ARRAYIDX1:%.*]] = getelementptr inbounds double, ptr [[A]], i64 [[I_08]]
; CHECK-NEXT:    store double [[ADD]], ptr [[ARRAYIDX1]], align 8
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i64 [[I_08_IN]], 1
; CHECK-NEXT:    br i1 [[CMP]], label [[FOR_BODY]], label [[FOR_COND_CLEANUP_LOOPEXIT]], !llvm.loop [[LOOP4:![0-9]+]]
;
entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.08.in = phi i64 [ %i.08, %for.body ], [ %N, %entry ]
  %i.08 = add nsw i64 %i.08.in, -1
  %arrayidx = getelementptr inbounds double, ptr %b, i64 %i.08
  %0 = load double, ptr %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx1 = getelementptr inbounds double, ptr %a, i64 %i.08
  store double %add, ptr %arrayidx1, align 8
  %cmp = icmp sgt i64 %i.08.in, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}


define void @vector_reverse_i64(i64 %N, ptr %a, ptr %b) #0 {
; CHECK-LABEL: @vector_reverse_i64(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A2:%.*]] = ptrtoint ptr [[A:%.*]] to i64
; CHECK-NEXT:    [[B1:%.*]] = ptrtoint ptr [[B:%.*]] to i64
; CHECK-NEXT:    [[CMP8:%.*]] = icmp sgt i64 [[N:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP8]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_COND_CLEANUP:%.*]]
; CHECK:       for.body.preheader:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = shl i64 [[TMP0]], 3
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ugt i64 [[TMP1]], [[N]]
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_MEMCHECK:%.*]]
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = shl i64 [[TMP2]], 6
; CHECK-NEXT:    [[TMP4:%.*]] = shl i64 [[N]], 3
; CHECK-NEXT:    [[TMP5:%.*]] = add i64 [[TMP4]], [[B1]]
; CHECK-NEXT:    [[TMP6:%.*]] = add i64 [[TMP4]], [[A2]]
; CHECK-NEXT:    [[TMP7:%.*]] = sub i64 [[TMP5]], [[TMP6]]
; CHECK-NEXT:    [[DIFF_CHECK:%.*]] = icmp ult i64 [[TMP7]], [[TMP3]]
; CHECK-NEXT:    br i1 [[DIFF_CHECK]], label [[SCALAR_PH]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP8:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP9:%.*]] = shl i64 [[TMP8]], 3
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N]], [[TMP9]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N]], [[N_MOD_VF]]
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP10:%.*]] = xor i64 [[INDEX]], -1
; CHECK-NEXT:    [[TMP11:%.*]] = add i64 [[TMP10]], [[N]]
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds i64, ptr [[B]], i64 [[TMP11]]
; CHECK-NEXT:    [[TMP13:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP14:%.*]] = shl i64 [[TMP13]], 3
; CHECK-NEXT:    [[TMP15:%.*]] = sub i64 1, [[TMP14]]
; CHECK-NEXT:    [[TMP16:%.*]] = getelementptr inbounds i64, ptr [[TMP12]], i64 [[TMP15]]
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <vscale x 8 x i64>, ptr [[TMP16]], align 8
; CHECK-NEXT:    [[TMP17:%.*]] = add <vscale x 8 x i64> [[WIDE_LOAD]], shufflevector (<vscale x 8 x i64> insertelement (<vscale x 8 x i64> poison, i64 1, i64 0), <vscale x 8 x i64> poison, <vscale x 8 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP18:%.*]] = getelementptr inbounds i64, ptr [[A]], i64 [[TMP11]]
; CHECK-NEXT:    [[TMP19:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP20:%.*]] = shl i64 [[TMP19]], 3
; CHECK-NEXT:    [[TMP21:%.*]] = sub i64 1, [[TMP20]]
; CHECK-NEXT:    [[TMP22:%.*]] = getelementptr inbounds i64, ptr [[TMP18]], i64 [[TMP21]]
; CHECK-NEXT:    store <vscale x 8 x i64> [[TMP17]], ptr [[TMP22]], align 8
; CHECK-NEXT:    [[TMP23:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP24:%.*]] = shl i64 [[TMP23]], 3
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], [[TMP24]]
; CHECK-NEXT:    [[TMP25:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP25]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP5:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 [[N_MOD_VF]], 0
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_COND_CLEANUP_LOOPEXIT:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_MOD_VF]], [[MIDDLE_BLOCK]] ], [ [[N]], [[FOR_BODY_PREHEADER]] ], [ [[N]], [[VECTOR_MEMCHECK]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.cond.cleanup.loopexit:
; CHECK-NEXT:    br label [[FOR_COND_CLEANUP]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
; CHECK:       for.body:
; CHECK-NEXT:    [[I_09_IN:%.*]] = phi i64 [ [[I_09:%.*]], [[FOR_BODY]] ], [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ]
; CHECK-NEXT:    [[I_09]] = add nsw i64 [[I_09_IN]], -1
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i64, ptr [[B]], i64 [[I_09]]
; CHECK-NEXT:    [[TMP26:%.*]] = load i64, ptr [[ARRAYIDX]], align 8
; CHECK-NEXT:    [[ADD:%.*]] = add i64 [[TMP26]], 1
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds i64, ptr [[A]], i64 [[I_09]]
; CHECK-NEXT:    store i64 [[ADD]], ptr [[ARRAYIDX2]], align 8
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i64 [[I_09_IN]], 1
; CHECK-NEXT:    br i1 [[CMP]], label [[FOR_BODY]], label [[FOR_COND_CLEANUP_LOOPEXIT]], !llvm.loop [[LOOP6:![0-9]+]]
;
entry:
  %cmp8 = icmp sgt i64 %N, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.09.in = phi i64 [ %i.09, %for.body ], [ %N, %entry ]
  %i.09 = add nsw i64 %i.09.in, -1
  %arrayidx = getelementptr inbounds i64, ptr %b, i64 %i.09
  %0 = load i64, ptr %arrayidx, align 8
  %add = add i64 %0, 1
  %arrayidx2 = getelementptr inbounds i64, ptr %a, i64 %i.09
  store i64 %add, ptr %arrayidx2, align 8
  %cmp = icmp sgt i64 %i.09.in, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}

attributes #0 = { "target-cpu"="generic" "target-features"="+neon,+sve" }

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 8}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}


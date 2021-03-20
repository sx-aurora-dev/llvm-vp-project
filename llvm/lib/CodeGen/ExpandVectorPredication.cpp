//===----- CodeGen/ExpandVectorPredication.cpp - Expand VP intrinsics -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements IR expansion for vector predication intrinsics, allowing
// targets to enable vector predication until just before codegen.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ExpandVectorPredication.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

using VPLegalization = TargetTransformInfo::VPLegalization;
using VPTransform = TargetTransformInfo::VPLegalization::VPTransform;

// Keep this in sync with TargetTransformInfo::VPLegalization.
#define VPINTERNAL_VPLEGAL_CASES                                               \
  VPINTERNAL_CASE(Legal)                                                       \
  VPINTERNAL_CASE(Discard)                                                     \
  VPINTERNAL_CASE(Convert)

#define VPINTERNAL_CASE(X) "|" #X

// Override options.
static cl::opt<std::string> EVLTransformOverride(
    "expandvp-override-evl-transform", cl::init(""), cl::Hidden,
    cl::desc("Options: <empty>" VPINTERNAL_VPLEGAL_CASES
             ". If non-empty, ignore "
             "TargetTransformInfo and "
             "always use this transformation for the %evl parameter (Used in "
             "testing)."));

static cl::opt<std::string> MaskTransformOverride(
    "expandvp-override-mask-transform", cl::init(""), cl::Hidden,
    cl::desc("Options: <empty>" VPINTERNAL_VPLEGAL_CASES
             ". If non-empty, Ignore "
             "TargetTransformInfo and "
             "always use this transformation for the %mask parameter (Used in "
             "testing)."));

#undef VPINTERNAL_CASE
#define VPINTERNAL_CASE(X) .Case(#X, VPLegalization::X)

static VPTransform parseOverrideOption(const std::string TextOpt) {
  return StringSwitch<VPTransform>(TextOpt) VPINTERNAL_VPLEGAL_CASES;
}

#undef INTERNAL_VPLEGAL_CASES

// Whether any override options are set.
static bool anyExpandVPOverridesSet() {
  return (EVLTransformOverride != "") || (MaskTransformOverride != "");
}

#define DEBUG_TYPE "expandvp"

STATISTIC(NumFoldedVL, "Number of folded vector length params");
STATISTIC(NumLoweredVPOps, "Number of folded vector predication operations");

///// Helpers {

/// \returns Whether the vector mask \p MaskVal has all lane bits set.
static bool isAllTrueMask(Value *MaskVal) {
  auto *ConstVec = dyn_cast<ConstantVector>(MaskVal);
  if (!ConstVec)
    return false;
  return ConstVec->isAllOnesValue();
}

/// \returns A non-excepting divisor constant for this type.
static Constant *getSafeDivisor(Type *DivTy) {
  assert(DivTy->isIntOrIntVectorTy());
  return ConstantInt::get(DivTy, 1u, false);
}

/// Transfer operation properties from \p OldVPI to \p NewVal.
static void transferDecorations(Value &NewVal, VPIntrinsic &VPI) {
  auto *NewInst = dyn_cast<Instruction>(&NewVal);
  if (!NewInst || !isa<FPMathOperator>(NewVal))
    return;

  auto *OldFMOp = dyn_cast<FPMathOperator>(&VPI);
  if (!OldFMOp)
    return;

  NewInst->setFastMathFlags(OldFMOp->getFastMathFlags());
}

/// Transfer all properties from \p OldOp to \p NewOp and replace all uses.
/// OldVP gets erased.
static void replaceOperation(Value &NewOp, VPIntrinsic &OldOp) {
  transferDecorations(NewOp, OldOp);
  OldOp.replaceAllUsesWith(&NewOp);
  OldOp.eraseFromParent();
}

//// } Helpers

namespace {

// Expansion pass state at function scope.
struct CachingVPExpander {
  Function &F;
  const TargetTransformInfo &TTI;
  const DataLayout &DL;

  /// \returns A (fixed length) vector with ascending integer indices
  /// (<0, 1, ..., NumElems-1>).
  /// \p Builder
  ///    Used for instruction creation.
  /// \p LaneTy
  ///    Integer element type of the result vector.
  /// \p NumElems
  ///    Number of vector elements.
  Value *createStepVector(IRBuilder<> &Builder, Type *LaneTy, int32_t NumElems);

  /// \returns A bitmask that is true where the lane position is less-than \p
  /// EVLParam
  ///
  /// \p Builder
  ///    Used for instruction creation.
  /// \p VLParam
  ///    The explicit vector length parameter to test against the lane
  ///    positions.
  /// \p ElemCount
  ///    Static (potentially scalable) number of vector elements
  Value *convertEVLToMask(IRBuilder<> &Builder, Value *EVLParam,
                          ElementCount ElemCount);

  Value *foldEVLIntoMask(VPIntrinsic &VPI);

  /// "Remove" the %evl parameter of \p PI by setting it to the static vector
  /// length of the operation.
  void discardEVLParameter(VPIntrinsic &PI);

  /// \brief Lower this VP binary operator to a unpredicated binary operator.
  Value *expandPredicationInBinaryOperator(IRBuilder<> &Builder,
                                           VPIntrinsic &PI);

  Value *expandPredicationInMemoryIntrinsic(IRBuilder<> &Builder, VPIntrinsic &PI);
  Value *expandPredicationInUnfoldedLoadStore(IRBuilder<> &Builder, VPIntrinsic &PI);

  /// \brief query TTI and expand the vector predication in \p P accordingly.
  Value *expandPredication(VPIntrinsic &PI);

  // Determine how and whether the VPIntrinsic \p VPI shall be expanded.
  // This overrides TTI with the cl::opts listed at the top of this file.
  VPLegalization getVPLegalizationStrategy(const VPIntrinsic &VPI) const;
  bool UsingTTIOverrides;

public:
  CachingVPExpander(Function &F, const TargetTransformInfo &TTI, const DataLayout &DL)
      : F(F), TTI(TTI), DL(DL), UsingTTIOverrides(anyExpandVPOverridesSet()) {}

  // expand VP ops in \p F according to \p TTI.
  bool expandVectorPredication();
};

//// CachingVPExpander {

Value *CachingVPExpander::createStepVector(IRBuilder<> &Builder, Type *LaneTy,
                                           int32_t NumElems) {
  // TODO add caching
  SmallVector<Constant *, 16> ConstElems;

  for (int32_t Idx = 0; Idx < NumElems; ++Idx) {
    ConstElems.push_back(ConstantInt::get(LaneTy, Idx, false));
  }

  return ConstantVector::get(ConstElems);
}

Value *CachingVPExpander::convertEVLToMask(IRBuilder<> &Builder,
                                           Value *EVLParam,
                                           ElementCount ElemCount) {
  // TODO add caching
  // Scalable vector %evl conversion.
  if (ElemCount.isScalable()) {
    auto *M = Builder.GetInsertBlock()->getModule();
    Type *BoolVecTy = VectorType::get(Builder.getInt1Ty(), ElemCount);
    Function *ActiveMaskFunc = Intrinsic::getDeclaration(
        M, Intrinsic::get_active_lane_mask, {BoolVecTy, EVLParam->getType()});
    // `get_active_lane_mask` performs an implicit less-than comparison.
    Value *ConstZero = Builder.getInt32(0);
    return Builder.CreateCall(ActiveMaskFunc, {ConstZero, EVLParam});
  }

  // Fixed vector %evl conversion.
  Type *LaneTy = EVLParam->getType();
  unsigned NumElems = ElemCount.getFixedValue();
  Value *VLSplat = Builder.CreateVectorSplat(NumElems, EVLParam);
  Value *IdxVec = createStepVector(Builder, LaneTy, NumElems);
  return Builder.CreateICmp(CmpInst::ICMP_ULT, IdxVec, VLSplat);
}

Value *
CachingVPExpander::expandPredicationInBinaryOperator(IRBuilder<> &Builder,
                                                     VPIntrinsic &VPI) {
  assert((isSafeToSpeculativelyExecute(&VPI) ||
          VPI.canIgnoreVectorLengthParam()) &&
         "Implicitly dropping %evl in non-speculatable operator!");

  auto OC = static_cast<Instruction::BinaryOps>(VPI.getFunctionalOpcode());
  assert(Instruction::isBinaryOp(OC));

  Value *FirstOp = VPI.getOperand(0);
  Value *SndOp = VPI.getOperand(1);
  Value *Mask = VPI.getMaskParam();

  // Blend in safe operands
  if (Mask && !isAllTrueMask(Mask)) {
    switch (OC) {
    default:
      // can safely ignore the predicate
      break;

    // Division operators need a safe divisor on masked-off lanes (1)
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      // 2nd operand must not be zero
      Value *SafeDivisor = getSafeDivisor(VPI.getType());
      SndOp = Builder.CreateSelect(Mask, SndOp, SafeDivisor);
    }
  }

  Value *NewBinOp = Builder.CreateBinOp(OC, FirstOp, SndOp, VPI.getName());

  replaceOperation(*NewBinOp, VPI);
  return NewBinOp;
}

/// \brief Lower this llvm.vp.(load|store|gather|scatter) to a non-vp
/// instruction.
Value *
CachingVPExpander::expandPredicationInMemoryIntrinsic(IRBuilder<> &Builder, VPIntrinsic &VPI) {
  assert(VPI.canIgnoreVectorLengthParam());
  auto &I = cast<Instruction>(VPI);

  auto MaskParam = VPI.getMaskParam();
  auto PtrParam = VPI.getMemoryPointerParam();
  auto DataParam = VPI.getMemoryDataParam();
  bool IsUnmasked = isAllTrueMask(MaskParam);

  MaybeAlign AlignOpt = VPI.getPointerAlignment();

  Value *NewMemoryInst = nullptr;
  switch (VPI.getIntrinsicID()) {
  default:
    abort(); // not a VP memory intrinsic

  case Intrinsic::vp_store: {
    if (IsUnmasked) {
      StoreInst *NewStore = Builder.CreateStore(DataParam, PtrParam, false);
      if (AlignOpt.hasValue())
        NewStore->setAlignment(AlignOpt.getValue());
      NewMemoryInst = NewStore;
    } else {
      NewMemoryInst = Builder.CreateMaskedStore(
          DataParam, PtrParam, AlignOpt.valueOrOne(), MaskParam);
    }
  } break;

  case Intrinsic::vp_load: {
    if (IsUnmasked) {
      LoadInst *NewLoad = Builder.CreateLoad(PtrParam, false);
      if (AlignOpt.hasValue())
        NewLoad->setAlignment(AlignOpt.getValue());
      NewMemoryInst = NewLoad;
    } else {
      NewMemoryInst =
          Builder.CreateMaskedLoad(PtrParam, AlignOpt.valueOrOne(), MaskParam);
    }
  } break;

  case Intrinsic::vp_scatter: {
    NewMemoryInst = Builder.CreateMaskedScatter(DataParam, PtrParam,
                                                AlignOpt.valueOrOne(), MaskParam);
  } break;

  case Intrinsic::vp_gather: {
    NewMemoryInst = Builder.CreateMaskedGather(PtrParam, AlignOpt.valueOrOne(),
                                               MaskParam, nullptr, I.getName());
  } break;
  }

  assert(NewMemoryInst);
  replaceOperation(*NewMemoryInst, VPI);
  return NewMemoryInst;
}

// The following are helper functions for loading and storing subvectors with
// variable offsets. There is currently no support for shuffles with
// non-constant masks, so these operations have to be done lane by lane.

// Create a load into Dest from the subvector of src given by a variable Offset and constant Width.
// Src is a pointer; Dest is a fixed-width vector; Offset and Width are specified in lanes.
Value *LoadSubvector(Value *Dest, Value *Src, Value *Offset, unsigned Width,
                     MaybeAlign EltAlign, Type *OffsetTy, Instruction *InsertPt) {
  assert(OffsetTy->isIntegerTy()
         && "Offset must be an integer type!");
  assert(Src->getType()->isPointerTy()
         && "Source must be a pointer!");
  assert(Dest->getType()->isVectorTy()
         && "Destination must be a vector!");
  Type *EltTy = Dest->getType()->getScalarType();
  IRBuilder<> Builder(InsertPt);
  Builder.SetCurrentDebugLocation(InsertPt->getDebugLoc());
  Value *SrcEltPtr = Builder.CreatePointerCast(
      Src, EltTy->getPointerTo(Src->getType()->getPointerAddressSpace()));
  auto *SubvecSrc = Builder.CreateInBoundsGEP(EltTy, SrcEltPtr, Offset);
  Value *VResult = Dest;
  for (unsigned i = 0; i < Width; ++i) {
    Value *vi = ConstantInt::get(OffsetTy, i);
    auto *EltOffset = Builder.CreateAdd(Offset, vi);
    auto *EltPtr = Builder.CreateInBoundsGEP(EltTy, SubvecSrc, vi);
    Value *EltLoad = Builder.CreateAlignedLoad(EltTy, EltPtr, EltAlign);
    VResult = Builder.CreateInsertElement(VResult, EltLoad, EltOffset);
  }
  return VResult;
}

// Create a store into Dest of the subvector of Val given by a variable Offset and constant Width.
// Dest is a pointer; Val is a fixed-width vector; Offset and Width are specified in lanes.
void StoreSubvector(Value *Val, Value *Dest, Value *Offset, unsigned Width,
                    MaybeAlign EltAlign, Type *OffsetTy, Instruction *InsertPt) {
  assert(OffsetTy->isIntegerTy()
         && "Offset must be an integer type!");
  assert(Dest->getType()->isPointerTy()
         && "Destination must be a pointer!");
  assert(Val->getType()->isVectorTy()
         && "Value must be a vector!");
  Type *EltTy = Val->getType()->getScalarType();
  IRBuilder<> Builder(InsertPt);
  Builder.SetCurrentDebugLocation(InsertPt->getDebugLoc());
  Value *DestEltPtr = Builder.CreatePointerCast(
      Dest, EltTy->getPointerTo(Dest->getType()->getPointerAddressSpace()));
  auto *SubvecDest = Builder.CreateInBoundsGEP(EltTy, DestEltPtr, Offset);
  for (unsigned i = 0; i < Width; ++i) {
    Value *vi = ConstantInt::get(OffsetTy, i);
    auto *EltOffset = Builder.CreateAdd(Offset, vi);
    auto *EltPtr = Builder.CreateInBoundsGEP(EltTy, SubvecDest, vi);
    Value *EltLoad = Builder.CreateExtractElement(Val, EltOffset);
    Builder.CreateAlignedStore(EltLoad, EltPtr, EltAlign);
  }
  return;
}

// We can split a vector store with variable length into contiguous conditional
// stores of powers of 2, one for each active bit in the length value. The
// offsets of the stores can be computed unconditionally using bitmasks of the
// length. The resulting logic looks like this:
//  PreBB:
//    // ... before intrinsic call
//    goto HeadBB;
//  HeadBB:
//    if (Length == VectorWidth)
//      goto ShortBB;
//    else
//      goto LongBB;
//  ShortBB:
//    // load/store full vector
//    goto PostBB;
//  LongBB:
//    for (int i = 0; i < LengthBits; ++i) {
//      if (hasBitSet(Length, i))
//        // load/store subvector of width 2^i
//    }
//    goto PostBB;
//  PostBB:
//  // after the intrinsic call ...


Value *
CachingVPExpander::expandPredicationInUnfoldedLoadStore(IRBuilder<> &Builder, VPIntrinsic &VPI) {
  assert(!VPI.canIgnoreVectorLengthParam());
  unsigned OC = VPI.getFunctionalOpcode();
  auto &I = cast<Instruction>(VPI);

  auto VLParam = VPI.getVectorLengthParam();
  auto PtrParam = VPI.getMemoryPointerParam();
  auto DataParam = VPI.getMemoryDataParam();
  auto MaskParam = VPI.getMaskParam();
  assert(isAllTrueMask(MaskParam));

  MaybeAlign AlignOpt = VPI.getPointerAlignment();

  Value *NewMemoryInst = nullptr;
  char const *Prefix;

  switch (OC) {
  default:
    abort(); // not a VP load or store

  case Instruction::Load:
    Prefix = "vp.load.";
    break;
  case Instruction::Store:
    Prefix = "vp.store.";
    break;
  }

  bool isLoad = ( OC == Instruction::Load );

  auto *VecTy = cast<FixedVectorType>(DataParam->getType());
  unsigned VecNumElts = VecTy->getNumElements();
  Type *VecEltTy = VecTy->getElementType();
  Type *VLTy = VLParam->getType();

  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  if (isa<Constant>(VLParam)) {
    switch (OC) {
    case Instruction::Load: {
      LoadInst *NewLoad = Builder.CreateLoad(PtrParam, false);
      if (AlignOpt.hasValue())
        NewLoad->setAlignment(AlignOpt.getValue());
      NewMemoryInst = NewLoad;
    } break;
    case Instruction::Store: {
      StoreInst *NewStore = Builder.CreateStore(DataParam, PtrParam, false);
      if (AlignOpt.hasValue())
        NewStore->setAlignment(AlignOpt.getValue());
      NewMemoryInst = NewStore;
    } break;
    default: break;
    }
    replaceOperation(*NewMemoryInst, VPI);
    return NewMemoryInst;
  }

  Instruction *ShortTerm, *LongTerm, *ThenTerm;
  Value *Pred;
  const Align BranchAlignment = commonAlignment(AlignOpt.valueOrOne(), VecEltTy->getPrimitiveSizeInBits() / 8);

  Value *VResult = ( isLoad ? UndefValue::get(VecTy) : nullptr );

  Pred = Builder.CreateICmpEQ(VLParam, ConstantInt::get(VLTy, VecNumElts));
  SplitBlockAndInsertIfThenElse(Pred, &I, &ShortTerm, &LongTerm);
  ShortTerm->getParent()->setName(Twine(Prefix) + "short");
  LongTerm->getParent()->setName(Twine(Prefix) + "long");
  I.getParent()->setName(Twine(Prefix) + "exit");

  unsigned LastBranchBit = Log2_64_Ceil(VecNumElts);
  unsigned BranchMask = maskTrailingOnes<unsigned>(LastBranchBit);
  unsigned BranchBit = LastBranchBit;
  while (BranchBit--) { // postdecr to avoid compairing 0u-1
    unsigned BranchOffsetMask =
        maskTrailingOnes<unsigned>(BranchBit + 1) ^ BranchMask;
    unsigned BranchWidth = 1 << BranchBit;
    Value *BranchWidthValue = ConstantInt::get(VLTy, BranchWidth);
    Value *BranchOffsetMaskValue = ConstantInt::get(VLTy, BranchOffsetMask);

    BasicBlock *IfBB = LongTerm->getParent();
    Builder.SetInsertPoint(LongTerm);
    Pred = Builder.CreateICmpUGT(Builder.CreateAnd(VLParam, BranchWidthValue), ConstantInt::get(VLTy, 0));
    ThenTerm = SplitBlockAndInsertIfThen(Pred, LongTerm, /*Unreachable*/ false);
    ThenTerm->getParent()->setName(Twine(Prefix) + "branch");
    LongTerm->getParent()->setName(Twine(Prefix) + "long");
    Builder.SetInsertPoint(ThenTerm);
    Value *BranchOffsetValue = Builder.CreateAnd(VLParam, BranchOffsetMaskValue);

    if (isLoad) {
      Value *BranchVResult =
          LoadSubvector(VResult, PtrParam, BranchOffsetValue, BranchWidth, BranchAlignment, VLTy, ThenTerm);
      Builder.SetInsertPoint(LongTerm);
      PHINode *ThenPhi = Builder.CreatePHI(VecTy, 2);
      ThenPhi->addIncoming(BranchVResult, ThenTerm->getParent());
      ThenPhi->addIncoming(VResult, IfBB);
      VResult = ThenPhi;
    } else {
      StoreSubvector(DataParam, PtrParam, BranchOffsetValue, BranchWidth, BranchAlignment, VLTy, ThenTerm);
    }
  }

  Builder.SetInsertPoint(ShortTerm);
  Value *ShortVResult;
  if (isLoad) {
    LoadInst *ShortLoad = Builder.CreateLoad(PtrParam, false);
    if (AlignOpt.hasValue())
      ShortLoad->setAlignment(AlignOpt.getValue());
    ShortVResult = ShortLoad;
  } else {
    StoreInst *ShortStore = Builder.CreateStore(DataParam, PtrParam, false);
    if (AlignOpt.hasValue())
      ShortStore->setAlignment(AlignOpt.getValue());
  }

  if (isLoad) {
    Builder.SetInsertPoint(&I);
    PHINode *Phi = Builder.CreatePHI(VecTy, 2);
    Phi->addIncoming(VResult, LongTerm->getParent());
    Phi->addIncoming(ShortVResult, ShortTerm->getParent());
    VResult = Phi;
  }

  if (isLoad)
    NewMemoryInst = VResult;
  else
    NewMemoryInst = nullptr;
  replaceOperation(*NewMemoryInst, VPI);
  return NewMemoryInst;
}


void CachingVPExpander::discardEVLParameter(VPIntrinsic &VPI) {
  LLVM_DEBUG(dbgs() << "Discard EVL parameter in " << VPI << "\n");

  if (VPI.canIgnoreVectorLengthParam())
    return;

  Value *EVLParam = VPI.getVectorLengthParam();
  if (!EVLParam)
    return;

  ElementCount StaticElemCount = VPI.getStaticVectorLength();
  Value *MaxEVL = nullptr;
  Type *Int32Ty = Type::getInt32Ty(VPI.getContext());
  if (StaticElemCount.isScalable()) {
    // TODO add caching
    auto *M = VPI.getModule();
    Function *VScaleFunc = Intrinsic::getDeclaration(M, Intrinsic::vscale, Int32Ty);
    IRBuilder<> Builder(VPI.getParent(), VPI.getIterator());
    Value *FactorConst = Builder.getInt32(StaticElemCount.getKnownMinValue());
    Value *VScale = Builder.CreateCall(VScaleFunc, {}, "vscale");
    MaxEVL = Builder.CreateMul(VScale, FactorConst, "scalable_size",
                               /*NUW*/ true, /*NSW*/ false);
  } else {
    MaxEVL = ConstantInt::get(Int32Ty, StaticElemCount.getFixedValue(), false);
  }
  VPI.setVectorLengthParam(MaxEVL);
}

Value *CachingVPExpander::foldEVLIntoMask(VPIntrinsic &VPI) {
  LLVM_DEBUG(dbgs() << "Folding vlen for " << VPI << '\n');

  IRBuilder<> Builder(&VPI);

  // Ineffective %evl parameter and so nothing to do here.
  if (VPI.canIgnoreVectorLengthParam())
    return &VPI;

  // Only VP intrinsics can have a %evl parameter
  Value *OldMaskParam = VPI.getMaskParam();
  Value *OldEVLParam = VPI.getVectorLengthParam();
  assert(OldMaskParam && "no mask param to fold the vl param into");
  assert(OldEVLParam && "no EVL param to fold away");

  LLVM_DEBUG(dbgs() << "OLD evl: " << *OldEVLParam << '\n');
  LLVM_DEBUG(dbgs() << "OLD mask: " << *OldMaskParam << '\n');

  // Convert the %evl predication into vector mask predication.
  ElementCount ElemCount = VPI.getStaticVectorLength();
  Value *VLMask = convertEVLToMask(Builder, OldEVLParam, ElemCount);
  Value *NewMaskParam = Builder.CreateAnd(VLMask, OldMaskParam);
  VPI.setMaskParam(NewMaskParam);

  // Drop the %evl parameter.
  discardEVLParameter(VPI);
  assert(VPI.canIgnoreVectorLengthParam() &&
         "transformation did not render the evl param ineffective!");

  // Reassess the modified instruction.
  return &VPI;
}

Value *CachingVPExpander::expandPredication(VPIntrinsic &VPI) {
  LLVM_DEBUG(dbgs() << "Lowering to unpredicated op: " << VPI << '\n');

  IRBuilder<> Builder(&VPI);

  // Try lowering to a LLVM instruction first.
  unsigned OC = VPI.getFunctionalOpcode();
#define FIRST_BINARY_INST(X) unsigned FirstBinOp = X;
#define LAST_BINARY_INST(X) unsigned LastBinOp = X;
#include "llvm/IR/Instruction.def"

  if (FirstBinOp <= OC && OC <= LastBinOp) {
    return expandPredicationInBinaryOperator(Builder, VPI);
  }

  switch (OC) {
  default:
    abort(); // unexpected intrinsic
  case Instruction::Load:
  case Instruction::Store:
    if (!VPI.canIgnoreVectorLengthParam() && isAllTrueMask(VPI.getMaskParam())) {
      return expandPredicationInUnfoldedLoadStore(Builder, VPI);
    } else {
      return expandPredicationInMemoryIntrinsic(Builder, VPI);
    }
    break;
  }

  return &VPI;
}

//// } CachingVPExpander

struct TransformJob {
  VPIntrinsic *PI;
  TargetTransformInfo::VPLegalization Strategy;
  TransformJob(VPIntrinsic *PI, TargetTransformInfo::VPLegalization InitStrat)
      : PI(PI), Strategy(InitStrat) {}

  bool isDone() const { return Strategy.shouldDoNothing(); }
};

void sanitizeStrategy(Instruction &I, VPLegalization &LegalizeStrat) {
  // Speculatable instructions do not strictly need predication.
  if (isSafeToSpeculativelyExecute(&I)) {
    // Converting a speculatable VP intrinsic means dropping %mask and %evl.
    // No need to expand %evl into the %mask only to ignore that code.
    if (LegalizeStrat.OpStrategy == VPLegalization::Convert)
      LegalizeStrat.EVLParamStrategy = VPLegalization::Discard;
    return;
  }

  // We have to preserve the predicating effect of %evl for this
  // non-speculatable VP intrinsic.
  // 1) Never discard %evl.
  // 2) If this VP intrinsic will be expanded to non-VP code, make sure that
  //    %evl gets folded into %mask.
  if ((LegalizeStrat.EVLParamStrategy == VPLegalization::Discard) ||
      (LegalizeStrat.OpStrategy == VPLegalization::Convert)) {
    LegalizeStrat.EVLParamStrategy = VPLegalization::Convert;
  }
}

VPLegalization
CachingVPExpander::getVPLegalizationStrategy(const VPIntrinsic &VPI) const {
  auto VPStrat = TTI.getVPLegalizationStrategy(VPI);
  if (LLVM_LIKELY(!UsingTTIOverrides)) {
    // No overrides - we are in production.
    return VPStrat;
  }

  // Overrides set - we are in testing, the following does not need to be
  // efficient.
  VPStrat.EVLParamStrategy = parseOverrideOption(EVLTransformOverride);
  VPStrat.OpStrategy = parseOverrideOption(MaskTransformOverride);
  return VPStrat;
}

/// \brief Expand llvm.vp.* intrinsics as requested by \p TTI.
bool CachingVPExpander::expandVectorPredication() {
  SmallVector<TransformJob, 16> Worklist;

  // Collect all VPIntrinsics that need expansion and determine their expansion
  // strategy.
  for (auto &I : instructions(F)) {
    auto *VPI = dyn_cast<VPIntrinsic>(&I);
    if (!VPI)
      continue;
    auto VPStrat = getVPLegalizationStrategy(*VPI);
    sanitizeStrategy(I, VPStrat);
    if (!VPStrat.shouldDoNothing()) {
      Worklist.emplace_back(VPI, VPStrat);
    }
  }
  if (Worklist.empty())
    return false;

  // Transform all VPIntrinsics on the worklist.
  LLVM_DEBUG(dbgs() << "\n:::: Transforming instructions (" << Worklist.size()
                    << ") ::::\n");
  for (TransformJob Job : Worklist) {
    // Transform the EVL parameter.
    switch (Job.Strategy.EVLParamStrategy) {
    case VPLegalization::Legal:
      break;
    case VPLegalization::Discard:
      discardEVLParameter(*Job.PI);
      break;
    case VPLegalization::Convert:
      if (foldEVLIntoMask(*Job.PI))
        ++NumFoldedVL;
      break;
    }
    Job.Strategy.EVLParamStrategy = VPLegalization::Legal;

    // Replace with a non-predicated operation.
    switch (Job.Strategy.OpStrategy) {
    case VPLegalization::Legal:
      break;
    case VPLegalization::Discard:
      llvm_unreachable("Invalid strategy for operators.");
    case VPLegalization::Convert:
      expandPredication(*Job.PI);
      ++NumLoweredVPOps;
      break;
    }
    Job.Strategy.OpStrategy = VPLegalization::Legal;

    assert(Job.isDone() && "incomplete transformation");
  }

  return true;
}
class ExpandVectorPredication : public FunctionPass {
public:
  static char ID;
  ExpandVectorPredication() : FunctionPass(ID) {
    initializeExpandVectorPredicationPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    const auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    const auto &DL = F.getParent()->getDataLayout();
    CachingVPExpander VPExpander(F, *TTI, DL);
    return VPExpander.expandVectorPredication();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesCFG();
  }
};
} // namespace

char ExpandVectorPredication::ID;
INITIALIZE_PASS_BEGIN(ExpandVectorPredication, "expandvp",
                      "Expand vector predication intrinsics", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(ExpandVectorPredication, "expandvp",
                    "Expand vector predication intrinsics", false, false)

FunctionPass *llvm::createExpandVectorPredicationPass() {
  return new ExpandVectorPredication();
}

PreservedAnalyses
ExpandVectorPredicationPass::run(Function &F, FunctionAnalysisManager &AM) {
  const auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  const auto &DL = F.getParent()->getDataLayout();
  CachingVPExpander VPExpander(F, TTI, DL);
  if (!VPExpander.expandVectorPredication())
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

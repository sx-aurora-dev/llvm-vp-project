#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/FPEnv.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/VectorBuilder.h>

namespace {
using namespace llvm;
using ShortTypeVec = VPIntrinsic::ShortTypeVec;
using ShortValueVec = SmallVector<Value *, 4>;
} // namespace

namespace llvm {

Module &VectorBuilder::getModule() const {
  return *Builder.GetInsertBlock()->getParent()->getParent();
}

Value *VectorBuilder::getAllTrueMask() {
  auto *boolTy = Builder.getInt1Ty();
  auto *maskTy = VectorType::get(boolTy, StaticVectorLength);
  return ConstantInt::getAllOnesValue(maskTy);
}

Value &VectorBuilder::RequestPred() {
  if (Mask)
    return *Mask;

  return *getAllTrueMask();
}

Value &VectorBuilder::RequestEVL() {
  if (ExplicitVectorLength)
    return *ExplicitVectorLength;

  assert(!StaticVectorLength.isScalable() && "TODO vscale lowering");
  auto *intTy = Builder.getInt32Ty();
  return *ConstantInt::get(intTy, StaticVectorLength.getFixedValue());
}

Value *VectorBuilder::createVectorCopy(Instruction &Inst, ValArray VecOpArray, Twine Name) {
  auto OC = Inst.getOpcode();
  auto VPID = VPIntrinsic::GetForOpcode(OC);
  if (VPID == Intrinsic::not_intrinsic) {
    return nullptr;
  }

  Optional<int> MaskPosOpt = VPIntrinsic::GetMaskParamPos(VPID);
  Optional<int> VLenPosOpt = VPIntrinsic::GetVectorLengthParamPos(VPID);

  Optional<int> CmpPredPos = None;
  if (isa<CmpInst>(Inst)) {
    CmpPredPos = 2;
  }

  // TODO transfer alignment

  // construct VP vector operands (including pred and evl)
  SmallVector<Value *, 6> VecParams;
  for (size_t i = 0; i < Inst.getNumOperands() + 5; ++i) {
    if (MaskPosOpt && (i == (size_t)MaskPosOpt.getValue())) {
      // First operand of select is mask (singular exception)
      if (VPID != Intrinsic::vp_select) {
        VecParams.push_back(&RequestPred());
      }
    }
    if (VLenPosOpt && (i == (size_t)VLenPosOpt.getValue())) {
      VecParams.push_back(&RequestEVL());
    }
    if (CmpPredPos && (i == (size_t)CmpPredPos.getValue())) {
      auto &CmpI = cast<CmpInst>(Inst);
      VecParams.push_back(ConstantInt::get(
          Type::getInt8Ty(Builder.getContext()), CmpI.getPredicate()));
    }
    if (i < VecOpArray.size())
      VecParams.push_back(VecOpArray[i]);
  }

  Type *ScaRetTy = Inst.getType();
  Type *VecRetTy = ScaRetTy->isVoidTy() ? ScaRetTy : &getVectorType(*ScaRetTy);
  auto &M = *Builder.GetInsertBlock()->getParent()->getParent();
  auto VPDecl =
      VPIntrinsic::getDeclarationForParams(&M, VPID, VecParams, VecRetTy);

#if 0
  // Prepare constraint fp params
  // FIXME: \p Inst could also be just another VP intrinsic.
  const auto *CFPIntrin = dyn_cast<ConstrainedFPIntrinsic>(&Inst);
  SmallVector<OperandBundleDef, 2> ConstraintBundles;
  if (CFPIntrin) {
    auto RoundOpt = CFPIntrin->getRoundingMode();
    if (RoundOpt) {
      auto *RoundParam =
          GetConstrainedFPRounding(Builder.getContext(), RoundOpt.getValue());
      ConstraintBundles.emplace_back("cfp-round", RoundParam);
    }
    auto ExceptOpt = CFPIntrin->getExceptionBehavior();
    if (ExceptOpt) {
      auto *ExceptParam =
          GetConstrainedFPExcept(Builder.getContext(), ExceptOpt.getValue());
      ConstraintBundles.emplace_back("cfp-except", ExceptParam);
    }
  }

  // Transfer FMF flags
  auto VPCall = Builder.CreateCall(VPDecl, VecParams, ConstraintBundles,
                                   Inst.getName() + ".vp");
#else
  // Transfer FMF flags
  auto VPCall = Builder.CreateCall(VPDecl, VecParams, Name);
#endif

  auto FPOp = dyn_cast<FPMathOperator>(&Inst);
  if (FPOp && isa<FPMathOperator>(VPCall)) {
    VPCall->setFastMathFlags(FPOp->getFastMathFlags());
  }

  return VPCall;
}

VectorType &VectorBuilder::getVectorType(Type &ElementTy) {
  return *VectorType::get(&ElementTy, StaticVectorLength);
}

static Type&
getScalarType(Type &Ty) {
  if (auto *VecTy = dyn_cast<VectorType>(&Ty))
    return *VecTy->getElementType();
  return Ty;
}

Instruction &VectorBuilder::createContiguousStore(Value &Val,
                                                  Value &ElemPointer,
                                                  MaybeAlign AlignOpt) {
  auto &PointerTy = cast<PointerType>(*ElemPointer.getType());
  auto &VecTy = *Val.getType();
  auto *VecPtrTy = VecTy.getPointerTo(PointerTy.getAddressSpace());
  auto *VecPtr = Builder.CreatePointerCast(&ElemPointer, VecPtrTy);
  assert((&getScalarType(*PointerTy.getElementType()) ==
          &getScalarType(*Val.getType())) &&
         "Element type mismatch");

  auto *StoreFunc = Intrinsic::getDeclaration(&getModule(), Intrinsic::vp_store,
                                              {&VecTy, VecPtrTy});
  ShortValueVec Args{&Val, VecPtr, &RequestPred(), &RequestEVL()};
  CallInst &StoreCall = *Builder.CreateCall(StoreFunc, Args);
  if (AlignOpt.hasValue()) {
    unsigned PtrPos =
        VPIntrinsic::GetMemoryPointerParamPos(Intrinsic::vp_store).getValue();
    StoreCall.addParamAttr(
        PtrPos, Attribute::getWithAlignment(getContext(), AlignOpt.getValue()));
  }
  return StoreCall;
}

Instruction &VectorBuilder::createContiguousLoad(Value &ElemPointer,
                                                 MaybeAlign AlignOpt,
                                                 Twine Name) {
  auto &PointerTy = cast<PointerType>(*ElemPointer.getType());
  auto &VecTy = getVectorType(getScalarType(*PointerTy.getElementType()));
  auto *VecPtrTy = VecTy.getPointerTo(PointerTy.getAddressSpace());
  auto *VecPtr = Builder.CreatePointerCast(&ElemPointer, VecPtrTy);

  auto *LoadFunc = Intrinsic::getDeclaration(&getModule(), Intrinsic::vp_load,
                                             {&VecTy, VecPtrTy});
  ShortValueVec Args{VecPtr, &RequestPred(), &RequestEVL()};
  CallInst &LoadCall = *Builder.CreateCall(LoadFunc, Args, Name);
  if (AlignOpt.hasValue()) {
    unsigned PtrPos =
        VPIntrinsic::GetMemoryPointerParamPos(Intrinsic::vp_load).getValue();
    LoadCall.addParamAttr(
        PtrPos, Attribute::getWithAlignment(getContext(), AlignOpt.getValue()));
  }
  return LoadCall;
}

Instruction &VectorBuilder::createScatter(Value &Val, Value &PointerVec,
                                          MaybeAlign AlignOpt) {
  auto *ScatterFunc =
      Intrinsic::getDeclaration(&getModule(), Intrinsic::vp_scatter,
                                {Val.getType(), PointerVec.getType()});
  ShortValueVec Args{&Val, &PointerVec, &RequestPred(), &RequestEVL()};
  CallInst &ScatterCall = *Builder.CreateCall(ScatterFunc, Args);
#if 0
  // TODO: 'align' unsupported on vector-of-pointers.
  if (AlignOpt.hasValue()) {
    unsigned PtrPos =
        VPIntrinsic::GetMemoryPointerParamPos(Intrinsic::vp_scatter).getValue();
    // ScatterCall.addParamAttr(
    //     PtrPos, Attribute::getWithAlignment(getContext(), AlignOpt.getValue()));
  }
#endif
  return ScatterCall;
}

Instruction &VectorBuilder::createGather(Value &PointerVec, MaybeAlign AlignOpt,
                                         Twine Name) {
  auto &PointerVecTy = cast<VectorType>(*PointerVec.getType());
  auto &ElemTy = *cast<PointerType>(*PointerVecTy.getElementType())
                      .getPointerElementType();
  auto &VecTy = *VectorType::get(&ElemTy, PointerVecTy.getElementCount());
  auto *GatherFunc = Intrinsic::getDeclaration(
      &getModule(), Intrinsic::vp_gather, {&VecTy, &PointerVecTy});

  ShortValueVec Args{&PointerVec, &RequestPred(), &RequestEVL()};
  CallInst &GatherCall = *Builder.CreateCall(GatherFunc, Args, Name);
#if 0
  // TODO: 'align' unsupported on vector-of-pointers.
  if (AlignOpt.hasValue()) {
    unsigned PtrPos =
        VPIntrinsic::GetMemoryPointerParamPos(Intrinsic::vp_gather).getValue();
    // FIXME 'align' invalid here.
    // GatherCall.addParamAttr(
    //     PtrPos, Attribute::getWithAlignment(getContext(), AlignOpt.getValue()));
  }
#endif
  return GatherCall;
}

Value *VectorBuilder::createVectorShift(Value *SrcVal, Value *Amount, Twine Name) {
  auto D = VPIntrinsic::getDeclarationForParams(
      &getModule(), Intrinsic::vp_vshift, {SrcVal, Amount});
  return Builder.CreateCall(D, {SrcVal, Amount, &RequestPred(), &RequestEVL()},
                            Name);
}

} // namespace llvm

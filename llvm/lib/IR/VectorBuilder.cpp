#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/FPEnv.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/VectorBuilder.h>

namespace {
using namespace llvm;
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

Value &VectorBuilder::requestPred() {
  if (Mask)
    return *Mask;

  return *getAllTrueMask();
}

Value &VectorBuilder::requestEVL() {
  if (ExplicitVectorLength)
    return *ExplicitVectorLength;

  assert(!StaticVectorLength.isScalable() && "TODO vscale lowering");
  auto *intTy = Builder.getInt32Ty();
  return *ConstantInt::get(intTy, StaticVectorLength.getFixedValue());
}

Value *VectorBuilder::createVectorInstruction(unsigned Opcode,
                                              ValArray VecOpArray, Twine Name) {
  auto VPID = VPIntrinsic::getForOpcode(Opcode);
  if (VPID == Intrinsic::not_intrinsic)
    llvm_unreachable("No VP intrinsic for this opcode");

  auto MaskPosOpt = VPIntrinsic::getMaskParamPos(VPID);
  auto VLenPosOpt = VPIntrinsic::getVectorLengthParamPos(VPID);

  auto *VPDecl =
      VPIntrinsic::getDeclarationForParams(&getModule(), VPID, VecOpArray);

  // Attach mask and evl operands.
  SmallVector<Value *, 6> VecParams;
  for (size_t i = 0; i < VecOpArray.size() + 2; ++i) {
    if (MaskPosOpt && (i == (size_t)MaskPosOpt.getValue())) {
      VecParams.push_back(&requestPred());
    }
    if (VLenPosOpt && (i == (size_t)VLenPosOpt.getValue())) {
      VecParams.push_back(&requestEVL());
    }
    if (i < VecOpArray.size())
      VecParams.push_back(VecOpArray[i]);
  }
  return Builder.CreateCall(VPDecl, VecParams, Name);
}

} // namespace llvm

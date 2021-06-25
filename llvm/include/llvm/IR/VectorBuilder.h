#ifndef LLVM_IR_VECTORBUILDER_H
#define LLVM_IR_VECTORBUILDER_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/IR/Value.h>

namespace llvm {

using ValArray = ArrayRef<Value *>;

class VectorBuilder {
  IRBuilder<> &Builder;

  // Explicit mask parameter
  Value *Mask;
  // Explicit vector length parameter
  Value *ExplicitVectorLength;
  // Compile-time vector length
  ElementCount StaticVectorLength;

  // get a valid mask/evl argument for the current predication contet
  Value &requestPred();
  Value &requestEVL();

public:
  VectorBuilder(IRBuilder<> &Builder)
      : Builder(Builder), Mask(nullptr), ExplicitVectorLength(nullptr),
        StaticVectorLength(ElementCount::getFixed(0)) {}

  Module &getModule() const;
  LLVMContext &getContext() const { return Builder.getContext(); }

  // The cannonical vector type for this \p ElementTy
  VectorType &getVectorType(Type &ElementTy);

  Value *getAllTrueMask();

  // Predication context tracker
  VectorBuilder &setMask(Value *_Mask) {
    Mask = _Mask;
    return *this;
  }
  VectorBuilder &setEVL(Value *_ExplicitVectorLength) {
    ExplicitVectorLength = _ExplicitVectorLength;
    return *this;
  }
  VectorBuilder &setStaticVL(unsigned FixedVL) {
    StaticVectorLength = ElementCount::getFixed(FixedVL);
    return *this;
  }
  VectorBuilder &setStaticVL(ElementCount ScalableVL) {
    assert(!ScalableVL.isScalable() &&
           "TODO implement symbolic vlens (aka scalable)");
    StaticVectorLength = ScalableVL;
    return *this;
  }

  // Create a map-vectorized copy of the instruction \p Inst with the underlying
  // IRBuilder instance. This operation may return nullptr if the instruction
  // could not be vectorized.
  Value *createVectorCopy(Instruction &Inst, ValArray VecOpArray,
                          Twine Name = "");

  // Emit a VP intrinsic call that mimics a regular instruction.
  // \p Opcode      The functional instruction opcode of the emitted intrinsic.
  // \p VecOpArray  The operand list.
  Value *createVectorInstruction(unsigned Opcode, ValArray VecOpArray,
                                 Twine Name = "");
};

} // namespace llvm

#endif // LLVM_IR_VECTORBUILDER_H

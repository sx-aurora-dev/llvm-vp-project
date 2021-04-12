#ifndef LLVM_IR_VECTORBUILDER_H
#define LLVM_IR_VECTORBUILDER_H

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/PatternMatch.h>

namespace llvm {

using ValArray = ArrayRef<Value*>;

class VectorBuilder {
  IRBuilder<> & Builder;

  // Explicit mask parameter
  Value * Mask;
  // Explicit vector length parameter
  Value * ExplicitVectorLength;
  // Compile-time vector length
  ElementCount StaticVectorLength;

  // get a valid mask/evl argument for the current predication contet
  Value& RequestPred();
  Value& RequestEVL();

public:
  VectorBuilder(IRBuilder<> &_builder)
      : Builder(_builder), Mask(nullptr), ExplicitVectorLength(nullptr),
        StaticVectorLength(ElementCount::getFixed(0)) {}

  Module & getModule() const;
  LLVMContext & getContext() const { return Builder.getContext(); }

  // The cannonical vector type for this \p ElementTy
  VectorType& getVectorType(Type &ElementTy);

  Value* getAllTrueMask();

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
    assert(!ScalableVL.isScalable() && "TODO implement symbolic vlens (aka scalable)");
    StaticVectorLength = ScalableVL;
    return *this;
  }

  // Create a map-vectorized copy of the instruction \p Inst with the underlying IRBuilder instance.
  // This operation may return nullptr if the instruction could not be vectorized.
  Value* createVectorCopy(Instruction & Inst, ValArray VecOpArray);

  // shift the elements in \p SrcVal by Amount where the result lane is true.
  Value* createVectorShift(Value *SrcVal, Value *Amount, Twine Name="");

  // Memory
  Instruction &createContiguousStore(Value &Val, Value &Pointer,
                                     MaybeAlign Alignment);
  Instruction &createContiguousLoad(Value &Pointer, MaybeAlign Alignment,
                                    Twine Name = "");
  Instruction &createScatter(Value &Val, Value &PointerVec,
                             MaybeAlign Alignment);
  Instruction &createGather(Value &PointerVec, MaybeAlign Alignment,
                            Twine Name = "");
};

} // namespace llvm

#endif // LLVM_IR_VECTORBUILDER_H

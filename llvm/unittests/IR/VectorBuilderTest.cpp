//===--------- VectorBuilderTest.cpp - VectorBuilder unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/VectorBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <sstream>

using namespace llvm;

namespace {

static int VectorNumElements = 8;

class VectorBuilderTest : public testing::Test {
protected:
  LLVMContext Context;

  VectorBuilderTest() : Context() {}

  std::unique_ptr<Module> createBuilderModule(Function *&Func, BasicBlock *&BB,
                                              Value *&Mask, Value *&EVL) {
    auto *Mod = new Module("TestModule", Context);
    auto *Int32Ty = Type::getInt32Ty(Context);
    auto *Mask8Ty = FixedVectorType::get(Type::getInt1Ty(Context), VectorNumElements);
    auto *VoidFuncTy =
        FunctionType::get(Type::getVoidTy(Context), {Mask8Ty, Int32Ty}, false);
    Func =
        Function::Create(VoidFuncTy, GlobalValue::ExternalLinkage, "bla", Mod);
    Mask = Func->getArg(0);
    EVL = Func->getArg(1);
    BB = BasicBlock::Create(Context, "entry", Func);

    return std::unique_ptr<Module>(Mod);
  }
};

/// Check that the property scopes include/llvm/IR/VPIntrinsics.def are closed.
TEST_F(VectorBuilderTest, TestCreateBinaryInstructions) {
  Function *F;
  BasicBlock *BB;
  Value *Mask, *EVL;
  auto Mod = createBuilderModule(F, BB, Mask, EVL);

  IRBuilder<> Builder(BB);
  VectorBuilder VBuild(Builder);
  VBuild.setMask(Mask).setEVL(EVL);

  auto *FloatVecTy = FixedVectorType::get(Type::getFloatTy(Context), VectorNumElements);
  auto *IntVecTy = FixedVectorType::get(Type::getInt32Ty(Context), VectorNumElements);

#define HANDLE_BINARY_INST(NUM, OPCODE, INSTCLASS)                             \
  {                                                                            \
    auto VPID = VPIntrinsic::getForOpcode(Instruction::OPCODE);                \
    bool IsFP = (#INSTCLASS)[0] == 'F';                                        \
    Value *Op = UndefValue::get(IsFP ? FloatVecTy : IntVecTy);                 \
    auto *VPIntrin =                                                           \
        VBuild.createVectorInstruction(Instruction::OPCODE, {Op, Op});         \
    ASSERT_TRUE(isa<VPIntrinsic>(VPIntrin));                                   \
    ASSERT_EQ(cast<VPIntrinsic>(VPIntrin)->getIntrinsicID(), VPID);            \
  }
#include "llvm/IR/Instruction.def"
}

} // end anonymous namespace

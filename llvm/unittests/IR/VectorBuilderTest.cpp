//===--------- VectorBuilderTest.cpp - VectorBuilder unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

class VectorBuilderTest : public testing::Test {
protected:
  LLVMContext Context;

  VectorBuilderTest() : Context() {}

  LLVMContext C;
};

/// Check that the property scopes include/llvm/IR/VPIntrinsics.def are closed.
TEST_F(VectorBuilderTest, TestCreateInstruction) {
  std::set<unsigned> VPOpcodes;
#define HANDLE_VP_TO_OPC(VP, OPC) VPOpcodes.insert(OPC);
#include "llvm/IR/VPIntrinsics.def"
}

} // end anonymous namespace

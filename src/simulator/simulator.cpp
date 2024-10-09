
#include "slap.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace {

llvm::FunctionType *getExternalAccessType(llvm::LLVMContext &ctx) {
  /* void slap_sim_access(slap_sim_context_t, void* node_handle, size_t memref,
                   size_t offset);*/
  auto void_type = llvm::Type::getVoidTy(ctx);
  auto size_t_ty = llvm::Type::getInt64Ty(ctx);
  auto ptr_ty = void_type->getPointerTo();
  return llvm::FunctionType::get(
      void_type, {ptr_ty, size_t_ty, size_t_ty, size_t_ty}, false);
}

llvm::FunctionType *getFunctionType(llvm::LLVMContext &ctx) {
  auto void_type = llvm::Type::getVoidTy(ctx);
  auto ptr_ty = void_type->getPointerTo();
  auto func_ptr_ty = getExternalAccessType(ctx)->getPointerTo();
  return llvm::FunctionType::get(void_type, {ptr_ty, func_ptr_ty}, false);
}

class CodegenContext {
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<llvm::Module> module;
  llvm::Function *func;
  llvm::IRBuilder<> builder;
  llvm::DenseMap<slap_graph_t, llvm::BasicBlock *> state_map;
  llvm::DenseMap<size_t, llvm::AllocaInst *> ivar_map;
  llvm::BasicBlock *entry;

public:
  CodegenContext()
      : ctx(std::make_unique<llvm::LLVMContext>()),
        module(std::make_unique<llvm::Module>("simulator", *ctx)),
        func(llvm::Function::Create(getFunctionType(*ctx),
                                    llvm::Function::ExternalLinkage,
                                    "simulation_entrypoint", *module)),
        builder(*ctx), state_map() {}

private:
  llvm::BasicBlock *getBasicBlock(slap_graph_t node) {
    auto it = state_map.find(node);
    if (it == state_map.end())
      return nullptr;
    return it->second;
  }

  llvm::Value *getCtxArg() { return func->arg_begin(); }
  llvm::Value *getExternalAccessArg() {
    auto it = func->arg_begin();
    return ++it;
  }

  llvm::BasicBlock *newBasicBlock(slap_graph_t node) {
    std::string name =
        "state_" + std::to_string(reinterpret_cast<uintptr_t>(node));
    auto bb = llvm::BasicBlock::Create(*ctx, name, func);
    state_map[node] = bb;
    return bb;
  }

  llvm::AllocaInst *getIVarAlloca(size_t ivar) {
    auto it = ivar_map.find(ivar);
    if (it == ivar_map.end()) {
      auto currentInsertPoint = builder.saveIP();
      builder.SetInsertPoint(entry);
      auto alloca = builder.CreateAlloca(llvm::Type::getInt64Ty(*ctx));
      ivar_map[ivar] = alloca;
      builder.restoreIP(currentInsertPoint);
      return alloca;
    }
    return it->second;
  }

  llvm::Value *emitExpr(slap_expr_t expr) {
    llvm::ArrayRef<ssize_t> coeffs(slap_expr_get_coefficients(expr),
                                   slap_expr_get_length(expr));
    ssize_t bias = slap_expr_get_bias(expr);
    // zero
    llvm::Value *acc = builder.getInt64(0);
    llvm::Type *int64_ty = llvm::Type::getInt64Ty(*ctx);
    for (auto [ivar, coeff] : llvm::enumerate(coeffs)) {
      if (coeff == 0)
        continue;
      auto alloca = getIVarAlloca(ivar);
      auto load = builder.CreateLoad(int64_ty, alloca);
      auto mul = builder.CreateMul(load, builder.getInt64(coeff));
      acc = builder.CreateAdd(acc, mul);
    }
    if (bias != 0)
      acc = builder.CreateAdd(acc, builder.getInt64(bias));
    return acc;
  }

  llvm::BasicBlock *emitSimulation(slap_graph_t node) {
    if (auto *bb = this->getBasicBlock(node))
      return bb;

    auto *bb = this->newBasicBlock(node);
    switch (slap_graph_get_kind(node)) {
    case SLAP_GRAPH_START: {
      bb->setName("start");
      entry = bb;
      auto next = slap_graph_get_next(node);
      auto next_bb = this->emitSimulation(next);
      this->builder.SetInsertPoint(bb);
      this->builder.CreateBr(next_bb);
      break;
    }
    case SLAP_GRAPH_END: {
      this->builder.SetInsertPoint(bb);
      this->builder.CreateRetVoid();
      break;
    }
    case SLAP_GRAPH_ACCESS: {
      this->builder.SetInsertPoint(bb);
      auto expr = slap_graph_get_expr(node);
      auto offset = this->emitExpr(expr);
      auto memref = this->builder.getInt64(slap_graph_get_identifer(node));
      auto raw_handle =
          this->builder.getInt64(reinterpret_cast<uintptr_t>(node));
      auto access = this->getExternalAccessArg();
      this->builder.CreateCall(getExternalAccessType(*ctx), access,
                               {getCtxArg(), raw_handle, memref, offset});
      auto next = slap_graph_get_next(node);
      auto next_bb = this->emitSimulation(next);
      this->builder.SetInsertPoint(bb);
      this->builder.CreateBr(next_bb);
      break;
    }
    case SLAP_GRAPH_UPDATE: {
      this->builder.SetInsertPoint(bb);
      auto expr = slap_graph_get_expr(node);
      auto updated = this->emitExpr(expr);
      auto ivar = slap_graph_get_identifer(node);
      auto alloca = this->getIVarAlloca(ivar);
      this->builder.CreateStore(updated, alloca);
      auto next = slap_graph_get_next(node);
      auto next_bb = this->emitSimulation(next);
      this->builder.SetInsertPoint(bb);
      this->builder.CreateBr(next_bb);
      break;
    }
    case SLAP_GRAPH_BRANCH: {
      this->builder.SetInsertPoint(bb);
      auto expr = slap_graph_get_expr(node);
      auto bound = this->emitExpr(expr);
      auto ivar = slap_graph_get_identifer(node);
      auto alloca = this->getIVarAlloca(ivar);
      auto load = this->builder.CreateLoad(this->builder.getInt64Ty(), alloca);
      auto cmp = this->builder.CreateICmpSLT(load, bound);
      auto then_ = this->emitSimulation(slap_graph_get_then(node));
      auto else_ = this->emitSimulation(slap_graph_get_else(node));
      this->builder.SetInsertPoint(bb);
      this->builder.CreateCondBr(cmp, then_, else_);
      break;
    }
    }
    return bb;
  }
  void optimize() {
    using namespace llvm;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    // Create the new pass manager builder.
    // Take a look at the PassBuilder constructor parameters for more
    // customization, e.g. specifying a TargetMachine or various debugging
    // options.
    PassBuilder PB;

    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Create the pass manager.
    // This one corresponds to a typical -O2 optimization pipeline.
    ModulePassManager MPM =
        PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);

    // Optimize the IR!
    MPM.run(*module, MAM);
  }

public:
  void process(slap_graph_t node) {
    emitSimulation(node);
    optimize();
  }

  void run(slap_sim_context_t sim_ctx) {
    auto jit = llvm::orc::LLJITBuilder().create();
    if (!jit)
      llvm::report_fatal_error("Failed to create JIT");
    auto res = jit->get()->addIRModule(
        llvm::orc::ThreadSafeModule(std::move(module), std::move(ctx)));
    if (res)
      llvm::report_fatal_error("Failed to add module to JIT");
    auto symbol = jit->get()->lookup("simulation_entrypoint");
    if (!symbol)
      llvm::report_fatal_error("Failed to find symbol");
    auto function = symbol.get().toPtr<void (*)(void *, void *)>();
    function(sim_ctx, reinterpret_cast<void *>(slap_sim_access));
  }
};

} // namespace

extern "C" void slap_initialize_llvm() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
}

extern "C" void slap_run_simulation(slap_sim_context_t ctx,
                                    slap_graph_t graph) {
  CodegenContext cg_ctx;
  cg_ctx.process(graph);
  cg_ctx.run(ctx);
}

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Visitors.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Parser/Parser.h>

#include <memory>
#include <slap.h>

namespace {
using namespace mlir;
void initialize(MLIRContext &context) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

class ExtractContext {
  slap_context_t slap_ctx;
  llvm::DenseMap<Value, size_t> ivars;
  affine::AffineForOp parent;
  slap_graph_t parent_epilogue;
  void record_ivar(Value ivar) {
    if (ivars.contains(ivar))
      return;
    ivars[ivar] = ivars.size();
  }

public:
  ExtractContext(slap_context_t ctx, affine::AffineForOp entry)
      : slap_ctx(ctx), ivars{}, parent(nullptr), parent_epilogue(nullptr) {
    record_ivar(entry.getInductionVar());
    entry.walk(
        [&](affine::AffineForOp op) { record_ivar(op.getInductionVar()); });
  }
  slap_context_t getSLAPContext() { return slap_ctx; }
  size_t getNumOfIvars() { return ivars.size(); }
  size_t getIVar(Value ivar) {
    auto it = ivars.find(ivar);
    if (it == ivars.end())
      llvm_unreachable("induction variable not found");
    return it->second;
  }
  slap_graph_t getParentEpilogue() { return parent_epilogue; }
  void setParentCondEpilogque(slap_graph_t parent) { parent_epilogue = parent; }
  affine::AffineForOp getParent() { return parent; }
  void setParent(affine::AffineForOp parent) { this->parent = parent; }
};

class ParentGuard {
  ExtractContext &ctx;
  slap_graph_t old_parent;
  affine::AffineForOp old_loop;

public:
  ParentGuard(ExtractContext &ctx, slap_graph_t parent,
              affine::AffineForOp loop)
      : ctx(ctx), old_parent(ctx.getParentEpilogue()),
        old_loop(ctx.getParent()) {
    ctx.setParentCondEpilogque(parent);
    ctx.setParent(loop);
  }
  ~ParentGuard() {
    ctx.setParentCondEpilogque(old_parent);
    ctx.setParent(old_loop);
  }
};

struct AffineBoundContext {
  ExtractContext &ext_ctx;
  affine::AffineBound bound;
  std::optional<AffineExpr> parent;
  llvm::SmallVector<ssize_t> coeff;
  ssize_t bias;

  size_t getIVar(Value ivar) { return ext_ctx.getIVar(ivar); }
};

void extractAffineExpr(AffineExpr expr, AffineBoundContext &ctx) {
  switch (expr.getKind()) {
  case AffineExprKind::Add: {
    auto add = cast<AffineBinaryOpExpr>(expr);
    extractAffineExpr(add.getLHS(), ctx);
    extractAffineExpr(add.getRHS(), ctx);
    break;
  }
  case AffineExprKind::Mul: {
    auto mul = cast<AffineBinaryOpExpr>(expr);
    if (!isa<AffineDimExpr>(mul.getLHS()))
      llvm_unreachable(
          "nested affine expression under multiplication is not supported");
    auto old_parent = ctx.parent;
    ctx.parent = expr;
    extractAffineExpr(mul.getRHS(), ctx);
    ctx.parent = old_parent;
  }
  case AffineExprKind::Constant: {
    auto constant = cast<AffineConstantExpr>(expr);
    if (!ctx.parent || ctx.parent->getKind() == AffineExprKind::Add)
      ctx.bias = constant.getValue();
    else {
      auto mul = cast<AffineBinaryOpExpr>(*ctx.parent);
      auto dim = cast<AffineDimExpr>(mul.getLHS());
      auto pos = dim.getPosition();
      auto value = ctx.bound.getOperand(pos);
      ctx.coeff[ctx.getIVar(value)] = constant.getValue();
    }
    break;
  }
  case AffineExprKind::DimId: {
    auto dim = cast<AffineDimExpr>(expr);
    auto pos = dim.getPosition();
    auto value = ctx.bound.getOperand(pos);
    ctx.coeff[ctx.getIVar(value)] = 1;
    break;
  }
  case AffineExprKind::SymbolId:
  case AffineExprKind::Mod:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
    llvm_unreachable("mod, floordiv, ceildiv, symbols are not supported");
  }
}

slap_expr_t extractAffineExpr(affine::AffineBound expr, ExtractContext &ctx) {
  auto map = expr.getMap();
  if (map.getNumResults() != 1)
    llvm_unreachable("only single result affine map is supported");
  auto result = map.getResult(0);
  auto affine_ctx = AffineBoundContext{
      .ext_ctx = ctx,
      .bound = expr,
      .parent = std::nullopt,
      .coeff = llvm::SmallVector<ssize_t>(ctx.getNumOfIvars(), 0),
      .bias = 0,
  };
  extractAffineExpr(result, affine_ctx);
  return slap_expr_new(ctx.getSLAPContext(), affine_ctx.coeff.data(),
                       affine_ctx.coeff.size(), affine_ctx.bias);
}

slap_graph_t createEpilogue(affine::AffineForOp loop, ExtractContext &ctx,
                            slap_graph_t cond_node) {
  size_t ivar = ctx.getIVar(loop.getInductionVar());
  ssize_t step = loop.getStepAsInt();
  llvm::SmallVector<ssize_t> coeff(ctx.getNumOfIvars(), 0);
  coeff[ivar] = 1;
  slap_expr_t expr =
      slap_expr_new(ctx.getSLAPContext(), coeff.data(), coeff.size(), step);
  return slap_graph_new_update(ctx.getSLAPContext(), ivar, expr, cond_node);
}

slap_graph_t extractOperation(Operation *op, ExtractContext &ctx);
slap_graph_t extractFromLoop(affine::AffineForOp loop, ExtractContext &ctx);

slap_graph_t extractTerminator(Operation *op, ExtractContext &ctx) {
  if (isa<func::ReturnOp>(op))
    return slap_graph_new_end(ctx.getSLAPContext());
  if (isa<affine::AffineYieldOp>(op))
    return ctx.getParentEpilogue();
  llvm_unreachable("unsupported terminator");
}

slap_graph_t extractOperation(Operation *op, ExtractContext &ctx) {
  for (;;) {
    if (op->hasTrait<OpTrait::IsTerminator>())
      return extractTerminator(op, ctx);
    if (isa<affine::AffineForOp>(op))
      return extractFromLoop(cast<affine::AffineForOp>(op), ctx);
    if (isa<RegionBranchOpInterface>(op))
      llvm_unreachable(
          "region branch other than affine for is not supported yet");
    op = op->getNextNode();
  }
}

slap_graph_t extractFromLoop(affine::AffineForOp loop, ExtractContext &ctx) {
  affine::AffineBound lb = loop.getLowerBound();
  AffineMap map = lb.getMap();
  auto ub_expr = extractAffineExpr(loop.getUpperBound(), ctx);
  slap_graph_t cond_node = slap_graph_new_branch(
      ctx.getSLAPContext(), ctx.getIVar(loop.getInductionVar()), ub_expr,
      nullptr, nullptr);
  slap_graph_t epilogue = createEpilogue(loop, ctx, cond_node);
  {
    ParentGuard guard(ctx, epilogue, loop);
    slap_graph_t body = extractOperation(&loop.getBody()->front(), ctx);
    slap_graph_branch_set_then(cond_node, body);
  }
  auto next = extractOperation(loop->getNextNode(), ctx);
  slap_graph_branch_set_else(cond_node, next);
  auto lb_expr = extractAffineExpr(lb, ctx);
  slap_graph_t lb_graph = slap_graph_new_update(
      ctx.getSLAPContext(), ctx.getIVar(loop.getInductionVar()), lb_expr,
      cond_node);
  return lb_graph;
}
slap_graph_t extractFromEntry(affine::AffineForOp entry, slap_context_t ctx) {
  ExtractContext extract_ctx{ctx, entry};
  auto loop = extractFromLoop(entry, extract_ctx);
  auto start = slap_graph_new_start(ctx, loop);
  return start;
}
} // namespace

extern "C" {
slap_graph_t slap_extract_affine_loop(slap_context_t ctx, char *path,
                                      size_t length) {
  using namespace mlir;
  using namespace llvm;
  MLIRContext context;
  initialize(context);
  llvm::StringRef filepath(path, length);
  ErrorOr<std::unique_ptr<MemoryBuffer>> buffer =
      MemoryBuffer::getFile(filepath);
  if (!buffer)
    return nullptr;
  SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(std::move(*buffer), SMLoc());
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(source_mgr, &context);
  if (!module)
    return nullptr;
  affine::AffineForOp entry;
  module->walk([&entry](affine::AffineForOp forOp) {
    if (forOp->hasAttr("slap.extract")) {
      entry = forOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!entry)
    return nullptr;

  return extractFromEntry(entry, ctx);
}
}

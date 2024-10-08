#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Visitors.h"
#include <llvm-20/llvm/ADT/DenseMap.h>
#include <llvm-20/llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Parser/Parser.h>

#include <memory>
#include <slpn.h>

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
  slpn_context_t slpn_ctx;
  llvm::DenseMap<Value, size_t> ivars;
  void record_ivar(Value ivar) {
    if (ivars.contains(ivar))
      return;
    ivars[ivar] = ivars.size();
  }

public:
  ExtractContext(slpn_context_t ctx, affine::AffineForOp entry)
      : slpn_ctx(ctx), ivars{} {
    record_ivar(entry.getInductionVar());
    entry.walk(
        [&](affine::AffineForOp op) { record_ivar(op.getInductionVar()); });
  }
  slpn_context_t getSLPNContext() { return slpn_ctx; }
  size_t getNumOfIvars() { return ivars.size(); }
  size_t getIVar(Value ivar) {
    auto it = ivars.find(ivar);
    if (it == ivars.end())
      llvm_unreachable("induction variable not found");
    return it->second;
  }
};

struct AffineBoundContext {
  affine::AffineBound bound;
  std::optional<AffineExpr> parent;
  llvm::SmallVector<ssize_t> coeff;
  ssize_t bias;
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
      ctx.coeff[pos] = constant.getValue();
    }
    break;
  }
  case AffineExprKind::DimId: {
    auto dim = cast<AffineDimExpr>(expr);
    auto pos = dim.getPosition();
    auto value = ctx.bound.getOperand(pos);
    ctx.coeff[pos] = 1;
    break;
  }
  case AffineExprKind::SymbolId:
  case AffineExprKind::Mod:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
    llvm_unreachable("mod, floordiv, ceildiv, symbols are not supported");
  }
}

slpn_expr_t extractAffineExpr(affine::AffineBound expr, ExtractContext &ctx) {
  auto map = expr.getMap();
  if (map.getNumResults() != 1)
    llvm_unreachable("only single result affine map is supported");
  auto result = map.getResult(0);
  auto affine_ctx = AffineBoundContext{
      .bound = expr,
      .parent = std::nullopt,
      .coeff = llvm::SmallVector<ssize_t>(ctx.getNumOfIvars(), 0),
      .bias = 0,
  };
  extractAffineExpr(result, affine_ctx);
  return slpn_expr_new(ctx.getSLPNContext(), affine_ctx.coeff.data(),
                       affine_ctx.coeff.size(), affine_ctx.bias);
}

slpn_graph_t extractFromLoop(affine::AffineForOp entry, ExtractContext &ctx) {
  affine::AffineBound lb = entry.getLowerBound();
  AffineMap map = lb.getMap();
  auto lb_expr = extractAffineExpr(lb, ctx);
  slpn_graph_t lb_graph = slpn_graph_new_update(
      ctx.getSLPNContext(), ctx.getIVar(entry.getInductionVar()), lb_expr,
      slpn_graph_new_end(ctx.getSLPNContext()));
  return lb_graph;
}
slpn_graph_t extractFromEntry(affine::AffineForOp entry, slpn_context_t ctx) {
  ExtractContext extract_ctx{ctx, entry};
  auto loop = extractFromLoop(entry, extract_ctx);
  auto start = slpn_graph_new_start(ctx, loop);
  return start;
}
} // namespace

extern "C" {
slpn_graph_t slpn_extract_affine_loop(slpn_context_t ctx, char *path,
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
    if (forOp->hasAttr("slpn.extract")) {
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

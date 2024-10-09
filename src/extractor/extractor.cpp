#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Parser/Parser.h>

#include <memory>
#include <numeric>
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
  llvm::DenseMap<Value, size_t> memrefs;
  affine::AffineForOp parent;
  slap_graph_t parent_epilogue;
  void recordIVar(Value ivar) { ivars.try_emplace(ivar, ivars.size()); }

public:
  ExtractContext(slap_context_t ctx, affine::AffineForOp entry)
      : slap_ctx(ctx), ivars{}, parent(nullptr), parent_epilogue(nullptr) {
    recordIVar(entry.getInductionVar());
    entry.walk(
        [&](affine::AffineForOp op) { recordIVar(op.getInductionVar()); });
  }
  slap_context_t getSLAPContext() { return slap_ctx; }
  size_t getNumOfIvars() { return ivars.size(); }
  size_t getIVar(Value ivar) {
    auto it = ivars.find(ivar);
    if (it == ivars.end()) {
      llvm_unreachable("induction variable not found");
    }
    return it->second;
  }
  slap_graph_t getParentEpilogue() { return parent_epilogue; }
  void setParentCondEpilogque(slap_graph_t parent) { parent_epilogue = parent; }
  affine::AffineForOp getParent() { return parent; }
  void setParent(affine::AffineForOp parent) { this->parent = parent; }
  size_t getMemRef(Value memref) {
    return memrefs.try_emplace(memref, memrefs.size()).first->second;
  }
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

struct AffineContext {
  ExtractContext &ext_ctx;
  std::optional<AffineExpr> parent;
  llvm::SmallVector<ssize_t> coeff;
  ssize_t bias;
  OperandRange operands;

  size_t getIVar(Value ivar) { return ext_ctx.getIVar(ivar); }
};

void extractAffineExpr(AffineExpr expr, AffineContext &ctx) {
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
    break;
  }
  case AffineExprKind::Constant: {
    auto constant = cast<AffineConstantExpr>(expr);
    if (!ctx.parent || ctx.parent->getKind() == AffineExprKind::Add)
      ctx.bias = constant.getValue();
    else {
      auto mul = cast<AffineBinaryOpExpr>(*ctx.parent);
      auto dim = cast<AffineDimExpr>(mul.getLHS());
      auto pos = dim.getPosition();
      auto value = ctx.operands[pos];
      ctx.coeff[ctx.getIVar(value)] = constant.getValue();
    }
    break;
  }
  case AffineExprKind::DimId: {
    auto dim = cast<AffineDimExpr>(expr);
    auto pos = dim.getPosition();
    auto value = ctx.operands[pos];
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
  auto affine_ctx = AffineContext{
      .ext_ctx = ctx,
      .parent = std::nullopt,
      .coeff = llvm::SmallVector<ssize_t>(ctx.getNumOfIvars(), 0),
      .bias = 0,
      .operands = expr.getOperands(),
  };
  extractAffineExpr(result, affine_ctx);
  return slap_expr_new(ctx.getSLAPContext(), affine_ctx.coeff.data(),
                       affine_ctx.coeff.size(), affine_ctx.bias);
}

slap_graph_t extractOperation(Operation *op, ExtractContext &ctx);
slap_graph_t extractFromLoop(affine::AffineForOp loop, ExtractContext &ctx);

slap_graph_t extractAffineAccess(Value memref, AffineMap map,
                                 OperandRange operands, Operation *next_node,
                                 MLIRContext *mctx, ExtractContext &ctx) {
  if (auto memTy = dyn_cast<MemRefType>(memref.getType())) {
    if (!memTy.hasStaticShape())
      llvm_unreachable("dynamic shape memref is not supported");
    auto layout = memTy.getLayout();
    if (layout.isIdentity()) {
      auto shape = memTy.getShape();
      auto exprs = map.getResults();
      auto zip = llvm::zip(shape, exprs);
      auto acc = std::accumulate(
          zip.begin(), zip.end(), getAffineConstantExpr(0, mctx),
          [](AffineExpr acc, std::tuple<int64_t, AffineExpr> pair) {
            auto [size, expr] = pair;
            return acc * size + expr;
          });
      auto affine_ctx = AffineContext{
          .ext_ctx = ctx,
          .parent = std::nullopt,
          .coeff = llvm::SmallVector<ssize_t>(ctx.getNumOfIvars(), 0),
          .bias = 0,
          .operands = operands,
      };
      acc = acc * (memTy.getElementTypeBitWidth() / 8);
      acc = simplifyAffineExpr(acc, map.getNumDims(), map.getNumSymbols());
      extractAffineExpr(acc, affine_ctx);
      auto expr = slap_expr_new(ctx.getSLAPContext(), affine_ctx.coeff.data(),
                                affine_ctx.coeff.size(), affine_ctx.bias);
      auto memref_id = ctx.getMemRef(memref);
      auto next = extractOperation(next_node, ctx);
      return slap_graph_new_access(ctx.getSLAPContext(), memref_id, expr, next);
    } else if (auto strided = dyn_cast<StridedLayoutAttr>(layout)) {
      //   if (!strided.hasStaticLayout())
      //     llvm_unreachable("dynamic layout is not supported");
      //   auto offset = strided.getOffset();
      //   auto stride = strided.getStrides();
      //   auto exprs = map.getResults();
      //   auto zip = llvm::zip(stride, exprs);
      //   auto acc = std::accumulate(
      //       zip.begin(), zip.end(), getAffineConstantExpr(offset, mctx),
      //       [](AffineExpr acc, std::tuple<int64_t, AffineExpr> pair) {
      //         auto [strided, expr] = pair;
      //         return acc + strided * expr;
      //       });
      //   auto affine_ctx = AffineContext{
      //       .ext_ctx = ctx,
      //       .parent = std::nullopt,
      //       .coeff = llvm::SmallVector<ssize_t>(ctx.getNumOfIvars(), 0),
      //       .bias = 0,
      //       .operands = operands,
      //   };
      //   extractAffineExpr(acc, affine_ctx);
      //   auto expr = slap_expr_new(ctx.getSLAPContext(),
      //   affine_ctx.coeff.data(),
      //                             affine_ctx.coeff.size(), affine_ctx.bias);
      //   auto memref_id = ctx.getMemRef(memref);
      //   auto next = extractOperation(next_node, ctx);
      //   return slap_graph_new_access(ctx.getSLAPContext(), memref_id, expr,
      //   next);
      llvm_unreachable("strided layout is not supported");
    }
    llvm_unreachable("unsupported layout");
  }
  llvm_unreachable("unsupported memref type");
}

slap_graph_t extractAffineAccess(affine::AffineLoadOp load,
                                 ExtractContext &ctx) {
  return extractAffineAccess(load.getMemRef(), load.getAffineMap(),
                             load->getOperands().drop_front(),
                             load->getNextNode(), load.getContext(), ctx);
}

slap_graph_t extractAffineAccess(affine::AffineStoreOp store,
                                 ExtractContext &ctx) {
  return extractAffineAccess(store.getMemRef(), store.getAffineMap(),
                             store.getOperands().drop_front(2),
                             store->getNextNode(), store.getContext(), ctx);
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
    if (auto loop = dyn_cast<affine::AffineForOp>(op))
      return extractFromLoop(loop, ctx);
    if (auto load = dyn_cast<affine::AffineLoadOp>(op))
      return extractAffineAccess(load, ctx);
    if (auto store = dyn_cast<affine::AffineStoreOp>(op))
      return extractAffineAccess(store, ctx);
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

#pragma once

#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slpn_expr *slpn_expr_t;
typedef struct slpn_graph *slpn_graph_t;
typedef struct slpn_context const *slpn_context_t;

slpn_expr_t slpn_expr_new(slpn_context_t, ssize_t *coeffs, size_t n,
                          ssize_t bias);

slpn_graph_t slpn_graph_new_start(slpn_context_t, slpn_graph_t next);
slpn_graph_t slpn_graph_new_end(slpn_context_t);
slpn_graph_t slpn_graph_new_access(slpn_context_t, size_t memref,
                                   slpn_expr_t offset, slpn_graph_t next);
slpn_graph_t slpn_graph_new_update(slpn_context_t, size_t ivar,
                                   slpn_expr_t expr, slpn_graph_t next);
slpn_graph_t slpn_graph_new_branch(slpn_context_t, size_t ivar,
                                   slpn_expr_t bound, slpn_graph_t then_,
                                   slpn_graph_t else_);

slpn_graph_t slpn_graph_start_set_next(slpn_graph_t, slpn_graph_t);
slpn_graph_t slpn_graph_new_access_set_next(slpn_graph_t, slpn_graph_t);
slpn_graph_t slpn_graph_new_update_set_next(slpn_graph_t, slpn_graph_t);
slpn_graph_t slpn_graph_new_branch_set_then(slpn_graph_t, slpn_graph_t);
slpn_graph_t slpn_graph_new_branch_set_else(slpn_graph_t, slpn_graph_t);

slpn_graph_t slpn_extract_affine_loop(slpn_context_t, char *path,
                                      size_t length);

#ifdef __cplusplus
}
#endif

#pragma once

#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slap_expr *slap_expr_t;
typedef struct slap_graph *slap_graph_t;
typedef struct slap_context const *slap_context_t;

slap_expr_t slap_expr_new(slap_context_t, ssize_t *coeffs, size_t n,
                          ssize_t bias);

slap_graph_t slap_graph_new_start(slap_context_t, slap_graph_t next);
slap_graph_t slap_graph_new_end(slap_context_t);
slap_graph_t slap_graph_new_access(slap_context_t, size_t memref,
                                   slap_expr_t offset, slap_graph_t next);
slap_graph_t slap_graph_new_update(slap_context_t, size_t ivar,
                                   slap_expr_t expr, slap_graph_t next);
slap_graph_t slap_graph_new_branch(slap_context_t, size_t ivar,
                                   slap_expr_t bound, slap_graph_t then_,
                                   slap_graph_t else_);

slap_graph_t slap_graph_start_set_next(slap_graph_t, slap_graph_t);
slap_graph_t slap_graph_access_set_next(slap_graph_t, slap_graph_t);
slap_graph_t slap_graph_update_set_next(slap_graph_t, slap_graph_t);
slap_graph_t slap_graph_branch_set_then(slap_graph_t, slap_graph_t);
slap_graph_t slap_graph_branch_set_else(slap_graph_t, slap_graph_t);

slap_graph_t slap_extract_affine_loop(slap_context_t, char *path,
                                      size_t length);

#ifdef __cplusplus
}
#endif

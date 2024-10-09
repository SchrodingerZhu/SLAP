#pragma once

#include <cstddef>
#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct slap_expr *slap_expr_t;
typedef struct slap_graph *slap_graph_t;
typedef struct slap_context const *slap_context_t;
typedef struct slap_sim_context *slap_sim_context_t;

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
size_t *slap_allocate_index_array(slap_context_t, size_t n);
typedef enum : int {
  SLAP_GRAPH_START,
  SLAP_GRAPH_END,
  SLAP_GRAPH_ACCESS,
  SLAP_GRAPH_UPDATE,
  SLAP_GRAPH_BRANCH,
} slap_graph_kind;

slap_graph_kind slap_graph_get_kind(slap_graph_t);
slap_graph_t slap_graph_get_next(slap_graph_t);
slap_graph_t slap_graph_get_then(slap_graph_t);
slap_graph_t slap_graph_get_else(slap_graph_t);
slap_expr_t slap_graph_get_expr(slap_graph_t);
size_t slap_graph_get_identifer(slap_graph_t);

ssize_t *slap_expr_get_coefficients(slap_expr_t);
size_t slap_expr_get_length(slap_expr_t);
ssize_t slap_expr_get_bias(slap_expr_t);

slap_graph_t slap_extract_affine_loop(slap_context_t, char *path, size_t length,
                                      size_t **vaddr, size_t *vaddr_len);

void slap_sim_access(slap_sim_context_t, void *node_handle, size_t memref,
                     size_t offset);

#ifdef __cplusplus
}
#endif

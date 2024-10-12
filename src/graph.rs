use std::{cell::UnsafeCell, collections::HashSet, ptr::NonNull};

use rustc_hash::FxHashMap;

use crate::{affine::Expr, simulator::SimulationCtx, Context};

#[derive(Clone)]
pub enum Graph<'a> {
    Start(Option<&'a Self>),
    End,
    Access {
        memref: usize,
        offset: &'a Expr<'a>,
        next: Option<&'a Self>,
    },
    Update {
        ivar: usize,
        expr: &'a Expr<'a>,
        next: Option<&'a Self>,
    },
    Branch {
        ivar: usize,
        bound: &'a Expr<'a>,
        then: Option<&'a Self>,
        r#else: Option<&'a Self>,
    },
}

impl<'a> Graph<'a> {
    fn populate_adjacency(&self, result: &mut FxHashMap<usize, Box<[usize]>>) {
        let token = self as *const _ as usize;
        match result.entry(token) {
            std::collections::hash_map::Entry::Occupied(_) => (),
            std::collections::hash_map::Entry::Vacant(entry) => match *self {
                Graph::Start(Some(next)) => {
                    entry.insert(vec![next as *const _ as usize].into_boxed_slice());
                    next.populate_adjacency(result);
                }
                Graph::Access {
                    next: Some(next), ..
                } => {
                    entry.insert(vec![next as *const _ as usize].into_boxed_slice());
                    next.populate_adjacency(result);
                }
                Graph::Update {
                    next: Some(next), ..
                } => {
                    entry.insert(vec![next as *const _ as usize].into_boxed_slice());
                    next.populate_adjacency(result);
                }
                Graph::Branch { then, r#else, .. } => {
                    let mut vec = Vec::new();
                    if let Some(then) = then {
                        vec.push(then as *const _ as usize);
                    }
                    if let Some(r#else) = r#else {
                        vec.push(r#else as *const _ as usize);
                    }
                    entry.insert(vec.into_boxed_slice());
                    if let Some(then) = then {
                        then.populate_adjacency(result);
                    }
                    if let Some(r#else) = r#else {
                        r#else.populate_adjacency(result);
                    }
                }
                _ => {
                    entry.insert(Default::default());
                }
            },
        }
    }
    pub fn adjacency(&self) -> FxHashMap<usize, Box<[usize]>> {
        let mut result = FxHashMap::default();
        self.populate_adjacency(&mut result);
        result
    }
    pub fn format(
        &self,
        writer: &mut std::fmt::Formatter<'_>,
        visited: &mut HashSet<NonNull<Self>>,
    ) -> std::fmt::Result {
        let token = NonNull::from(self);
        if visited.contains(&token) {
            write!(writer, "...")?;
            return Ok(());
        }
        visited.insert(token);
        match self {
            Graph::Start(next) => {
                write!(writer, "Start(")?;
                if let Some(next) = next {
                    next.format(writer, visited)?;
                }
                write!(writer, ")")
            }
            Graph::End => {
                write!(writer, "End")
            }
            Graph::Access {
                memref,
                offset,
                next,
            } => {
                write!(writer, "Access({}, {:?}, ", memref, offset)?;
                if let Some(next) = next {
                    next.format(writer, visited)?;
                }
                write!(writer, ")")
            }
            Graph::Update { ivar, expr, next } => {
                write!(writer, "Update({}, {:?}, ", ivar, expr)?;
                if let Some(next) = next {
                    next.format(writer, visited)?;
                }
                write!(writer, ")")
            }
            Graph::Branch {
                ivar,
                bound,
                then,
                r#else,
            } => {
                write!(writer, "Branch({}, {:?}, ", ivar, bound)?;
                if let Some(then) = then {
                    then.format(writer, visited)?;
                    write!(writer, ", ")?;
                }
                if let Some(r#else) = r#else {
                    r#else.format(writer, visited)?;
                }
                write!(writer, ")")
            }
        }
    }
    pub fn get_affine_dim(&self) -> usize {
        match self {
            Graph::Access { offset, .. } => offset.affine_dim(),
            Graph::Update { expr, .. } => expr.affine_dim(),
            Graph::Branch { bound, .. } => bound.affine_dim(),
            Graph::Start(n) => n.map_or(0, |x| x.get_affine_dim()),
            Graph::End => 0,
        }
    }
    pub fn vectorize_all(
        &self,
        ctx: &SimulationCtx,
        average: bool,
    ) -> FxHashMap<usize, Box<[f64]>> {
        let mut result = FxHashMap::default();
        self.vectorize_all_impl(ctx, &mut result, self.get_affine_dim(), average);
        result
    }
    fn vectorize_all_impl(
        &self,
        ctx: &SimulationCtx,
        result: &mut FxHashMap<usize, Box<[f64]>>,
        affine_dim: usize,
        average: bool,
    ) {
        let token = self as *const _ as usize;
        if result.contains_key(&token) {
            return;
        }
        let vector = self.vectorize(ctx, affine_dim, average);
        result.insert(token, vector);
        match self {
            Graph::Start(Some(next)) => next.vectorize_all_impl(ctx, result, affine_dim, average),
            Graph::Access {
                next: Some(next), ..
            } => {
                next.vectorize_all_impl(ctx, result, affine_dim, average);
            }
            Graph::Update {
                next: Some(next), ..
            } => {
                next.vectorize_all_impl(ctx, result, affine_dim, average);
            }
            Graph::Branch { then, r#else, .. } => {
                if let Some(then) = then {
                    then.vectorize_all_impl(ctx, result, affine_dim, average);
                }
                if let Some(r#else) = r#else {
                    r#else.vectorize_all_impl(ctx, result, affine_dim, average);
                }
            }
            _ => (),
        }
    }
    pub fn vectorize(&self, ctx: &SimulationCtx, affine_dim: usize, average: bool) -> Box<[f64]> {
        let distro = ctx.get_node_dist(self);
        let target_value = if average {
            distro
                .and_then(|distro| {
                    let mut total_sum = 0usize;
                    let mut total_freq = 0usize;
                    for (x, y) in distro.iter() {
                        total_sum += x * y;
                        total_freq += y;
                    }
                    total_sum.checked_div(total_freq)
                })
                .map(|x| x as f64)
                .unwrap_or(-1.0)
        } else {
            distro
                .and_then(|x| x.iter().max_by_key(|x| x.1))
                .map(|x| *x.0 as f64)
                .unwrap_or(-1.0)
        };
        let mut result = vec![];
        let kind: f64 = match self {
            Graph::Start(_) => 0.0,
            Graph::End => 1.0,
            Graph::Access { .. } => 2.0,
            Graph::Update { .. } => 3.0,
            Graph::Branch { .. } => 4.0,
        };
        result.push(kind);
        let ivar = match self {
            Graph::Access { memref, .. } => *memref as f64,
            Graph::Update { ivar, .. } => *ivar as f64,
            Graph::Branch { ivar, .. } => *ivar as f64,
            _ => -1.0,
        };
        result.push(ivar);
        match self {
            Graph::Access { offset, .. } => {
                offset.vectorize_into(&mut result);
            }
            Graph::Update { expr, .. } => {
                expr.vectorize_into(&mut result);
            }
            Graph::Branch { bound, .. } => {
                bound.vectorize_into(&mut result);
            }
            _ => {
                for _ in 0..affine_dim {
                    result.push(-1.0);
                }
            }
        }
        result.push(target_value);
        result.into_boxed_slice()
    }
}

impl std::fmt::Debug for Graph<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format(f, &mut HashSet::new())
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_new_start(
    ctx: *const Context,
    next: *mut Graph<'_>,
) -> *mut Graph<'_> {
    let ctx = &*ctx;
    let next = NonNull::new(next).map(|ptr| ptr.as_ref());
    ctx.arena
        .alloc(UnsafeCell::new(Graph::Start(next)))
        .get_mut()
}

#[no_mangle]
pub extern "C" fn slap_graph_new_end<'a>(ctx: *const Context) -> *mut Graph<'a> {
    let ctx = unsafe { &*ctx };
    ctx.arena.alloc(UnsafeCell::new(Graph::End)).get_mut()
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_new_access<'a>(
    ctx: *const Context,
    memref: usize,
    offset: *mut Expr<'a>,
    next: *mut Graph<'a>,
) -> *mut Graph<'a> {
    let ctx = &*ctx;
    ctx.arena
        .alloc(UnsafeCell::new(Graph::Access {
            memref,
            offset: &*offset,
            next: NonNull::new(next).map(|ptr| ptr.as_ref()),
        }))
        .get_mut()
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_new_update<'a>(
    ctx: *const Context,
    ivar: usize,
    expr: *mut Expr<'a>,
    next: *mut Graph<'a>,
) -> *mut Graph<'a> {
    let ctx = &*ctx;
    ctx.arena
        .alloc(UnsafeCell::new(Graph::Update {
            ivar,
            expr: &*expr,
            next: NonNull::new(next).map(|ptr| ptr.as_ref()),
        }))
        .get_mut()
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_new_branch<'a>(
    ctx: *const Context,
    ivar: usize,
    bound: *mut Expr<'a>,
    then: *mut Graph<'a>,
    r#else: *mut Graph<'a>,
) -> *mut Graph<'a> {
    let ctx = &*ctx;
    ctx.arena
        .alloc(UnsafeCell::new(Graph::Branch {
            ivar,
            bound: &*bound,
            then: NonNull::new(then).map(|ptr| ptr.as_ref()),
            r#else: NonNull::new(r#else).map(|ptr| ptr.as_ref()),
        }))
        .get_mut()
}

#[allow(improper_ctypes)]
extern "C" {
    fn slap_extract_affine_loop<'a>(
        ctx: *const Context,
        filename: *const std::os::raw::c_char,
        length: usize,
        vaddr: *mut *const usize,
        vaddr_len: *mut usize,
    ) -> Option<NonNull<Graph<'a>>>;
}

impl<'a> Graph<'a> {
    pub fn new_from_file(ctx: &'a Context, filename: &str) -> Option<(&'a Self, &'a [usize])> {
        let filename = std::ffi::CString::new(filename).unwrap();
        unsafe {
            let mut vaddr_cell = std::ptr::null();
            let mut vaddr_len = 0;
            let graph = slap_extract_affine_loop(
                ctx,
                filename.as_ptr(),
                filename.as_bytes().len(),
                &mut vaddr_cell as _,
                &mut vaddr_len,
            );
            graph
                .map(|graph| graph.as_ref())
                .map(|graph| (graph, std::slice::from_raw_parts(vaddr_cell, vaddr_len)))
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_start_set_next<'a>(
    start: *mut Graph<'a>,
    next: *mut Graph<'a>,
) {
    if let Some(mut start) = NonNull::new(start) {
        let start = start.as_mut();
        if let Graph::Start(ref mut field) = start {
            *field = NonNull::new(next).map(|ptr| ptr.as_ref());
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_access_set_next<'a>(
    access: *mut Graph<'a>,
    next: *mut Graph<'a>,
) {
    if let Some(mut access) = NonNull::new(access) {
        let access = access.as_mut();
        if let Graph::Access {
            next: ref mut field,
            ..
        } = access
        {
            *field = NonNull::new(next).map(|ptr| ptr.as_ref());
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_update_set_next<'a>(
    update: *mut Graph<'a>,
    next: *mut Graph<'a>,
) {
    if let Some(mut update) = NonNull::new(update) {
        let update = update.as_mut();
        if let Graph::Update {
            next: ref mut field,
            ..
        } = update
        {
            *field = NonNull::new(next).map(|ptr| ptr.as_ref());
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_branch_set_then(branch: *mut Graph<'_>, then: *mut Graph) {
    if let Some(mut branch) = NonNull::new(branch) {
        let branch = branch.as_mut();
        if let Graph::Branch {
            then: ref mut field,
            ..
        } = branch
        {
            *field = NonNull::new(then).map(|ptr| ptr.as_ref());
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_branch_set_else(branch: *mut Graph<'_>, r#else: *mut Graph) {
    if let Some(mut branch) = NonNull::new(branch) {
        let branch = branch.as_mut();
        if let Graph::Branch {
            r#else: ref mut field,
            ..
        } = branch
        {
            *field = NonNull::new(r#else).map(|ptr| ptr.as_ref());
        }
    }
}

/*
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
*/

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum GraphKind {
    Start = 0,
    End = 1,
    Access = 2,
    Update = 3,
    Branch = 4,
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_get_kind(graph: *const Graph<'_>) -> GraphKind {
    let graph = &*graph;
    match graph {
        Graph::Start(_) => GraphKind::Start,
        Graph::End => GraphKind::End,
        Graph::Access { .. } => GraphKind::Access,
        Graph::Update { .. } => GraphKind::Update,
        Graph::Branch { .. } => GraphKind::Branch,
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_get_expr(graph: *const Graph<'_>) -> *const Expr<'_> {
    let graph = &*graph;
    match *graph {
        Graph::Access { offset, .. } => offset,
        Graph::Update { expr, .. } => expr,
        Graph::Branch { bound, .. } => bound,
        _ => std::ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_get_next(graph: *const Graph<'_>) -> *const Graph<'_> {
    let graph = &*graph;
    match graph {
        Graph::Start(next) => next.map(|ptr| ptr as _).unwrap_or(std::ptr::null()),
        Graph::Access { next, .. } => next.map(|ptr| ptr as _).unwrap_or(std::ptr::null()),
        Graph::Update { next, .. } => next.map(|ptr| ptr as _).unwrap_or(std::ptr::null()),
        _ => std::ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_get_then(graph: *const Graph<'_>) -> *const Graph<'_> {
    let graph = &*graph;
    match graph {
        Graph::Branch { then, .. } => then.map(|ptr| ptr as _).unwrap_or(std::ptr::null()),
        _ => std::ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_get_else(graph: *const Graph<'_>) -> *const Graph<'_> {
    let graph = &*graph;
    match graph {
        Graph::Branch { r#else, .. } => r#else.map(|ptr| ptr as _).unwrap_or(std::ptr::null()),
        _ => std::ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_graph_get_identifer(graph: *const Graph<'_>) -> usize {
    let graph = &*graph;
    match *graph {
        Graph::Access { memref, .. } => memref,
        Graph::Update { ivar, .. } => ivar,
        Graph::Branch { ivar, .. } => ivar,
        _ => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_allocate_index_array(ctx: *const Context, len: usize) -> *mut usize {
    let ctx = &*ctx;
    ctx.arena.alloc_slice_fill_default(len).as_mut_ptr()
}

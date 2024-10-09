use std::{cell::UnsafeCell, collections::BTreeMap, ptr::NonNull};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::Graph;

#[derive(Default)]
pub struct NodeInfo {
    logic_time: FxHashMap<usize, usize>,
    reuse_interval: BTreeMap<usize, usize>,
}

impl std::fmt::Debug for NodeInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeInfo")
            .field("reuse_interval", &self.reuse_interval)
            .finish()
    }
}

#[derive(Debug)]
pub struct SimulationCtx<'a> {
    pub(crate) block_size: usize,
    pub(crate) vaddrs: &'a [usize],
    pub(crate) logic_time: usize,
    pub(crate) node_info: bumpalo::collections::Vec<'a, NodeInfo>,
    pub(crate) address_map: FxHashMap<NonNull<Graph<'a>>, usize>,
}

impl<'a> SimulationCtx<'a> {
    unsafe fn access(&mut self, node_id: usize, block_id: usize) {
        let time = self.logic_time;
        self.logic_time += 1;
        let node_info = self.node_info.get_unchecked_mut(node_id);
        let last_access = node_info.logic_time.entry(block_id).or_insert(time);
        if *last_access != time {
            let interval = time - *last_access;
            node_info
                .reuse_interval
                .entry(interval)
                .and_modify(|e| *e += 1)
                .or_insert(1);
            *last_access = time;
        }
    }
    pub fn new(ctx: &'a crate::Context, block_size: usize, vaddrs: &'a [usize]) -> Self {
        Self {
            block_size,
            vaddrs,
            logic_time: 0,
            node_info: bumpalo::collections::Vec::new_in(&ctx.arena),
            address_map: FxHashMap::default(),
        }
    }
    fn populate_node_info_impl(
        &mut self,
        g: &'a Graph<'a>,
        visited: &mut FxHashSet<NonNull<Graph<'a>>>,
    ) {
        if !visited.insert(NonNull::from(g)) {
            return;
        }
        match g {
            Graph::Start(Some(x)) => self.populate_node_info_impl(x, visited),
            Graph::Access { next, .. } => {
                let nonnull = NonNull::from(g);
                self.address_map.entry(nonnull).or_insert_with(|| {
                    let res = self.node_info.len();
                    self.node_info.push(NodeInfo::default());
                    res
                });
                if let Some(x) = next {
                    self.populate_node_info_impl(x, visited);
                }
            }
            Graph::Update { next: Some(x), .. } => self.populate_node_info_impl(x, visited),
            Graph::Branch {
                then: Some(x),
                r#else: Some(y),
                ..
            } => {
                self.populate_node_info_impl(x, visited);
                self.populate_node_info_impl(y, visited);
            }
            _ => (),
        }
    }

    pub fn populate_node_info(&mut self, g: &'a Graph<'a>) {
        self.populate_node_info_impl(g, &mut FxHashSet::default());
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_sim_access(
    ctx: *const UnsafeCell<SimulationCtx>,
    node_id: usize,
    block_id: usize,
) {
    let ctx = &mut *(*ctx).get();
    ctx.access(node_id, block_id);
}

#[no_mangle]
pub unsafe extern "C" fn slap_sim_get_memref_vaddr(
    ctx: *const UnsafeCell<SimulationCtx>,
    memref_id: usize,
) -> usize {
    let ctx = &mut *(*ctx).get();
    ctx.vaddrs[memref_id]
}

#[no_mangle]
pub unsafe extern "C" fn slap_sim_get_node_id(
    ctx: *const UnsafeCell<SimulationCtx>,
    graph: *mut Graph,
) -> usize {
    let ctx = &mut *(*ctx).get();
    ctx.address_map[&NonNull::new_unchecked(graph)]
}

#[no_mangle]
pub unsafe extern "C" fn slap_sim_get_block_size(ctx: *const UnsafeCell<SimulationCtx>) -> usize {
    let ctx = &mut *(*ctx).get();
    ctx.block_size
}

#[allow(improper_ctypes)]
extern "C" {
    pub fn slap_initialize_llvm();
    pub fn slap_run_simulation<'a>(ctx: *const UnsafeCell<SimulationCtx>, graph: *const Graph<'a>);
}

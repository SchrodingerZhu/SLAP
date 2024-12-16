use std::{cell::UnsafeCell, collections::BTreeMap, ptr::NonNull};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::Graph;

#[derive(Debug)]
pub struct SimulationCtx<'a> {
    block_size: usize,
    vaddrs: &'a [usize],
    pub(crate) node_info: bumpalo::collections::Vec<'a, usize>,
    pub(crate) address_map: FxHashMap<NonNull<Graph<'a>>, usize>,
    cache_size: usize,
    logic_time: usize,
    cache: FxHashMap<usize, usize>,
    access_time_to_address: BTreeMap<usize, usize>,
}

impl<'a> SimulationCtx<'a> {
    unsafe fn access(&mut self, node_id: usize, block_id: usize) {
        self.logic_time += 1;
        let pre_insert_len = self.cache.len();
        let to_evict = match self.cache.entry(block_id) {
            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                let old_time = occupied.insert(self.logic_time);
                self.access_time_to_address
                    .remove(&old_time)
                    .expect("block not found in access_time_to_address");
                self.access_time_to_address
                    .insert(self.logic_time, block_id);
                None
            }
            std::collections::hash_map::Entry::Vacant(vacant) => {
                vacant.insert(self.logic_time);
                self.access_time_to_address
                    .insert(self.logic_time, block_id);
                *self.node_info.get_unchecked_mut(node_id) += 1;
                if pre_insert_len == self.cache_size {
                    self.access_time_to_address.pop_first().map(|x| x.1)
                } else {
                    None
                }
            }
        };
        if let Some(block_id) = to_evict {
            self.cache.remove(&block_id);
        }
    }
    pub fn new(
        ctx: &'a crate::Context,
        block_size: usize,
        cache_size: usize,
        vaddrs: &'a [usize],
    ) -> Self {
        Self {
            block_size,
            vaddrs,
            node_info: bumpalo::collections::Vec::new_in(&ctx.arena),
            address_map: FxHashMap::default(),
            cache_size,
            logic_time: 0,
            cache: FxHashMap::default(),
            access_time_to_address: BTreeMap::new(),
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
                    self.node_info.push(Default::default());
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

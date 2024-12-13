use std::{cell::UnsafeCell, collections::BTreeMap, collections::LinkedList, ptr::NonNull};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::Graph;

#[derive(Debug)]
pub struct SimulationCtx<'a> {
    block_size: usize,
    vaddrs: &'a [usize],
    pub(crate) node_info: bumpalo::collections::Vec<'a, BTreeMap<usize, usize>>,
    pub(crate) address_map: FxHashMap<NonNull<Graph<'a>>, usize>,
    cache_size: usize,
    cache_stack:  LinkedList<usize>,
}

impl<'a> SimulationCtx<'a> {
    unsafe fn access(&mut self, node_id: usize, block_id: usize) {
        let mut found = false;
        for (index, &value) in self.cache_stack.iter().enumerate() {
            if value == block_id {
                found = true;
                let mut split = self.cache_stack.split_off(index);
                split.pop_front();
                self.cache_stack.extend(split);
                break;
            }
        }
        
        if self.cache_stack.len() == self.cache_size {
            self.cache_stack.pop_back();
        }
        self.cache_stack.push_back(block_id);
        
        let node_info = self.node_info.get_unchecked_mut(node_id);
        if found {
            node_info
                .entry(1)
                .and_modify(|e| *e += 1)
                .or_insert(1);
        }
        else {
            node_info
                .entry(0)
                .and_modify(|e| *e += 1)
                .or_insert(1);
        }
        
        
    }
    pub fn new(ctx: &'a crate::Context, block_size: usize, cache_size:usize, vaddrs: &'a [usize]) -> Self {
        Self {
            block_size,
            vaddrs,
            node_info: bumpalo::collections::Vec::new_in(&ctx.arena),
            address_map: FxHashMap::default(),
            cache_size,
            cache_stack: LinkedList::default(),
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

    pub fn get_node_dist(&self, g: &Graph<'a>) -> Option<&BTreeMap<usize, usize>> {
        self.address_map
            .get(&NonNull::from(g))
            .map(|x| &self.node_info[*x])
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

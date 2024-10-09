use std::{cell::UnsafeCell, collections::BTreeMap, ffi::c_void};

use rustc_hash::FxHashMap;

use crate::graph::Graph;

#[derive(Debug, Default)]
pub struct NodeInfo {
    logic_time: FxHashMap<usize, usize>,
    reuse_interval: BTreeMap<usize, usize>,
}

pub struct SimulationCtx {
    pub(crate) block_size: usize,
    pub(crate) virtual_addr: FxHashMap<usize, usize>,
    pub(crate) logic_time: usize,
    pub(crate) node_info: FxHashMap<*const c_void, UnsafeCell<NodeInfo>>,
}

impl SimulationCtx {
    fn access(&mut self, node: *const c_void, memref: usize, offset: usize) {
        let time = self.logic_time;
        self.logic_time += 1;
        let vaddr = *self.virtual_addr.get(&memref).unwrap();
        let block_id = (vaddr + offset) / self.block_size;
        let node_info = self.node_info.entry(node).or_default();
        let node_info = unsafe { &mut *node_info.get() };
        let logic_time = node_info.logic_time.entry(block_id).or_insert(time);
        if *logic_time != time {
            let interval = time - *logic_time;
            node_info
                .reuse_interval
                .entry(interval)
                .and_modify(|e| *e += 1)
                .or_insert(1);
            *logic_time = time;
        }
    }
}

impl std::fmt::Debug for SimulationCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimulationCtx")
            .field("block_size", &self.block_size)
            .field("virtual_addr", &self.virtual_addr)
            .field("logic_time", &self.logic_time)
            .finish()?;
        writeln!(f)?;
        for (node, info) in &self.node_info {
            f.debug_struct("NodeInfo")
                .field("node", &node)
                .field("reuse_interval", unsafe { &(*info.get()).reuse_interval })
                .finish()?;
            writeln!(f)?;
        }
        Ok(())
    }
}

#[no_mangle]
pub unsafe extern "C" fn slap_sim_access(
    ctx: *const UnsafeCell<SimulationCtx>,
    node: *const c_void,
    memref: usize,
    offset: usize,
) {
    let ctx = &mut *(*ctx).get();
    ctx.access(node, memref, offset);
}

#[allow(improper_ctypes)]
extern "C" {
    pub fn slap_initialize_llvm();
    pub fn slap_run_simulation<'a>(ctx: *const UnsafeCell<SimulationCtx>, graph: *const Graph<'a>);
}

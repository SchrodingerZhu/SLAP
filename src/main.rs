mod affine;
mod graph;
mod simulator;

struct Context {
    arena: bumpalo::Bump,
}

fn main() {
    unsafe {
        simulator::slap_initialize_llvm();
    }
    let ctx = Context {
        arena: bumpalo::Bump::new(),
    };
    let (g, vaddrs) =
        graph::Graph::new_from_file(&ctx, "example/gemm.mlir").expect("failed to parse mlir");
    unsafe {
        let ctx = simulator::SimulationCtx {
            block_size: 64,
            virtual_addr: vaddrs.iter().copied().enumerate().collect(),
            logic_time: 0,
            node_info: Default::default(),
        };
        let cell = std::cell::UnsafeCell::new(ctx);
        simulator::slap_run_simulation(&cell, g);
        println!("{:?}", &*cell.get());
    }
    println!("{g:?}");
}

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
        let mut ctx = simulator::SimulationCtx::new(&ctx, 64, vaddrs);
        ctx.populate_node_info(g);
        let cell = std::cell::UnsafeCell::new(ctx);
        simulator::slap_run_simulation(&cell, g);
        println!("{:#?}", &*cell.get());
    }
    println!("{g:?}");
}

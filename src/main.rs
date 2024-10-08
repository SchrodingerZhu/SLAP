mod affine;
mod graph;

struct Context {
    arena: bumpalo::Bump,
}

fn main() {
    let ctx = Context {
        arena: bumpalo::Bump::new(),
    };
    let g = graph::Graph::new_from_file(&ctx, "/tmp/test.mlir").expect("failed to parse mlir");
    println!("{g:?}");
}

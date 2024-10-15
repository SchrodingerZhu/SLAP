use std::{cell::UnsafeCell, path::PathBuf};

use clap::Parser;

mod affine;
mod graph;
mod simulator;

pub struct Context {
    arena: bumpalo::Bump,
    dump_node: bool,
    printer: UnsafeCell<Box<dyn std::io::Write>>,
}

#[derive(clap::Parser)]
enum Command {
    /// Generate RI distribution for the given affine program
    Distribution {
        #[clap(short, long)]
        /// Path to the affine program
        input: PathBuf,
        #[clap(short, long)]
        /// Path to the output file, if not provided, the result will be printed to stdout
        output: Option<PathBuf>,
    },
    /// Vectorize the given affine program into training data
    Vectorize {
        #[clap(short, long)]
        /// Path to the affine program
        input: PathBuf,
        /// Path to the adjacency output file, if not provided, the result will be printed to stdout
        #[clap(short, long)]
        adjacency: Option<PathBuf>,
        /// Path to the node data output file, if not provided, the result will be printed to stdout
        #[clap(short, long)]
        data: Option<PathBuf>,
        /// Print average RI instead of max RI
        #[clap(short = 'A', long)]
        average: bool,
    },
}

#[no_mangle]
unsafe extern "C" fn slap_dump_node_of_affine_access(ctx: *const Context) -> bool {
    (*ctx).dump_node
}

// void slap_print_callback(const char *, size_t, void *);
#[no_mangle]
unsafe extern "C" fn slap_print_callback(data: *const u8, len: usize, ctx: *mut std::ffi::c_void) {
    let ctx = &mut *(ctx as *mut Context);
    let data = std::slice::from_raw_parts(data, len);
    (*ctx.printer.get()).write_all(data).unwrap();
}

fn main() {
    let cmd = Command::parse();
    unsafe {
        simulator::slap_initialize_llvm();
    }
    match cmd {
        Command::Distribution { input, output } => {
            let ctx = Context {
                arena: bumpalo::Bump::new(),
                dump_node: true,
                printer: UnsafeCell::new(
                    output
                        .map(|x| {
                            Box::new(std::fs::File::create(x).unwrap()) as Box<dyn std::io::Write>
                        })
                        .unwrap_or_else(|| Box::new(std::io::stdout())),
                ),
            };
            let (g, vaddrs) = graph::Graph::new_from_file(&ctx, &format!("{}", input.display()))
                .expect("failed to parse mlir");
            unsafe {
                let mut sctx = simulator::SimulationCtx::new(&ctx, 64, vaddrs);
                sctx.populate_node_info(g);
                let cell = std::cell::UnsafeCell::new(sctx);
                simulator::slap_run_simulation(&cell, g);
                writeln!(&mut *ctx.printer.get(), "{{").unwrap();
                for (i, (k, v)) in (*cell.get()).address_map.iter().enumerate() {
                    write!(
                        &mut *ctx.printer.get(),
                        "\t\"{}\" : {{",
                        k.as_ptr() as usize
                    )
                    .unwrap();
                    for (i, x) in (*cell.get()).node_info[*v].iter().enumerate() {
                        write!(&mut *ctx.printer.get(), "\"{}\" : {}", x.0, x.1).unwrap();
                        if i != (*cell.get()).node_info[*v].len() - 1 {
                            write!(&mut *ctx.printer.get(), ", ").unwrap();
                        }
                    }
                    if i != (*cell.get()).address_map.len() - 1 {
                        writeln!(&mut *ctx.printer.get(), "}},").unwrap();
                    } else {
                        writeln!(&mut *ctx.printer.get(), "}}").unwrap();
                    }
                }
                writeln!(&mut *ctx.printer.get(), "}}").unwrap();
            }
        }
        Command::Vectorize {
            input,
            adjacency,
            data,
            average,
        } => {
            let ctx = Context {
                arena: bumpalo::Bump::new(),
                dump_node: false,
                printer: UnsafeCell::new(Box::new(std::io::stderr())),
            };
            let (g, vaddrs) = graph::Graph::new_from_file(&ctx, &format!("{}", input.display()))
                .expect("failed to parse mlir");
            let adj = g.adjacency();
            let mut adj_writer = adjacency
                .map(|x| Box::new(std::fs::File::create(x).unwrap()) as Box<dyn std::io::Write>)
                .unwrap_or_else(|| Box::new(std::io::stdout()));
            let adj_json = serde_json::to_string_pretty(&adj).unwrap();
            adj_writer.write_all(adj_json.as_bytes()).unwrap();
            let mut data_writer = data
                .map(|x| Box::new(std::fs::File::create(x).unwrap()) as Box<dyn std::io::Write>)
                .unwrap_or_else(|| Box::new(std::io::stdout()));
            unsafe {
                let mut sctx = simulator::SimulationCtx::new(&ctx, 64, vaddrs);
                sctx.populate_node_info(g);
                let cell = std::cell::UnsafeCell::new(sctx);
                simulator::slap_run_simulation(&cell, g);
                let vectorized = g.vectorize_all(&*cell.get(), average);
                let json = serde_json::to_string_pretty(&vectorized).unwrap();
                data_writer.write_all(json.as_bytes()).unwrap();
            }
        }
    }
}

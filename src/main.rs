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
    #[clap(name = "distribution")]
    /// Generate RI distribution for the given affine program
    Distribution {
        #[clap(short, long)]
        /// Path to the affine program
        input: PathBuf,
        #[clap(short, long)]
        /// Path to the output file, if not provided, the result will be printed to stdout
        output: Option<PathBuf>,
    },
    #[clap(name = "vectorize")]
    /// Vectorize the given affine program into training data
    Vectorize {
        #[clap(short, long)]
        /// Path to the affine program
        input: PathBuf,
        /// Path to the output file, if not provided, the result will be printed to stdout
        #[clap(short, long)]
        output: Option<PathBuf>,
    },
}

#[no_mangle]
pub unsafe extern "C" fn slap_dump_node_of_affine_access(ctx: *const Context) -> bool {
    (*ctx).dump_node
}

// void slap_print_callback(const char *, size_t, void *);
#[no_mangle]
pub unsafe extern "C" fn slap_print_callback(
    data: *const std::os::raw::c_char,
    len: usize,
    ctx: *mut std::ffi::c_void,
) {
    let ctx = &mut *(ctx as *mut Context);
    let data = std::slice::from_raw_parts(data as *const u8, len);
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
                for (k, v) in (*cell.get()).address_map.iter() {
                    writeln!(
                        &mut *ctx.printer.get(),
                        "{:?} -> {:?}",
                        k.as_ptr() as usize,
                        (*cell.get()).node_info[*v]
                    )
                    .unwrap();
                }
            }
        }
        Command::Vectorize { input, output } => {
            todo!()
        }
    }
}

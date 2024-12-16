use std::{cell::UnsafeCell, path::PathBuf};

use burn_dataset::SqliteDatasetWriter;
use clap::Parser;

mod affine;
mod dataset;
mod graph;
mod simulator;

pub struct Context {
    arena: bumpalo::Bump,
    dump_node: bool,
    printer: UnsafeCell<Box<dyn std::io::Write>>,
}

#[derive(clap::Parser)]
enum Command {
    /// Generate Miss Count Distribution
    MissCount {
        #[clap(short, long)]
        /// Path to the affine program
        input: PathBuf,
        #[clap(short, long)]
        cache_size: usize,
        #[clap(short, long)]
        block_size: usize,
        #[clap(short, long)]
        /// Path to the output file, if not provided, the result will be printed to stdout
        output: Option<PathBuf>,
    },
    /// Generate Training Dataset
    Dataset {
        #[clap(short, long)]
        /// Path to the affine program
        input: PathBuf,
        #[clap(short, long)]
        /// Path to the output file
        output: PathBuf,
        // Simulation parameters
        #[clap(short, long)]
        config: PathBuf,
    },
}

#[derive(serde::Deserialize)]
struct SimilationConfig {
    replacements: Vec<dataset::Replacement>,
    block_size: usize,
    cache_sizes: Vec<usize>,
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

fn init() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| unsafe {
        simulator::slap_initialize_llvm();
    });
}

fn main() {
    let cmd = Command::parse();
    init();
    match cmd {
        Command::MissCount {
            input,
            output,
            cache_size,
            block_size,
        } => {
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
                let mut sctx = simulator::SimulationCtx::new(&ctx, block_size, cache_size, vaddrs);
                sctx.populate_node_info(g);
                let cell = std::cell::UnsafeCell::new(sctx);
                simulator::slap_run_simulation(&cell, g);
                for (k, v) in (*cell.get()).address_map.iter() {
                    writeln!(
                        &mut *ctx.printer.get(),
                        "id: {}, node: {}, miss: {}",
                        v,
                        k.as_ptr() as usize,
                        (*cell.get()).node_info[*v]
                    )
                    .unwrap();
                }
            }
        }
        Command::Dataset {
            input,
            output,
            config,
        } => {
            let config: SimilationConfig =
                serde_json::from_reader(std::fs::File::open(config).unwrap()).unwrap();
            let input = std::fs::read_to_string(input).unwrap();
            let mut writer = SqliteDatasetWriter::new(output, true).unwrap();
            dataset::generate_dataset(
                &input,
                &config.replacements,
                config.block_size,
                &config.cache_sizes,
                &mut writer,
            );
        }
    }
}

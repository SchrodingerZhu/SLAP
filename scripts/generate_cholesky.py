import argparse
import asyncio
import aiofiles

SRC = """
module {
    func.func @cholesky(%b: memref<[:N:]xf32>, %L: memref<[:N:]x[:N:]xf32>, %x: memref<[:N:]xf32>) {
        // Allocate a local variable for loop index i and j
        affine.for %i = 0 to [:N:] {
        // x[i] = b[i]
        %b_i = affine.load %b[%i] : memref<[:N:]xf32>
        affine.store %b_i, %x[%i] : memref<[:N:]xf32>
        
        affine.for %j = 0 to affine_map<(d0) -> (d0)> (%i) {
            %L_ij = affine.load %L[%i, %j] : memref<[:N:]x[:N:]xf32>
            %x_j = affine.load %x[%j] : memref<[:N:]xf32>
            %prod = arith.mulf %L_ij, %x_j : f32
            %x_i = affine.load %x[%i] : memref<[:N:]xf32> 
            %tmp = arith.subf %x_i, %prod : f32
            affine.store %tmp, %x[%i] : memref<[:N:]xf32>
        }

        // x[i] = x[i] / L[i][i]
        %L_ii = affine.load %L[%i, %i] : memref<[:N:]x[:N:]xf32>
        %x_i_final = affine.load %x[%i] : memref<[:N:]xf32>
        %final_x_i = arith.divf %x_i_final, %L_ii : f32
        affine.store %final_x_i, %x[%i] : memref<[:N:]xf32>
        } { slap.extract }
    return
    }
}




"""


def gen(n):
    src = SRC.replace("[:N:]", str(n))
    return src

async def single_task(args, src, n):
    async with aiofiles.tempfile.NamedTemporaryFile("w") as f:
        await f.write(src)
        await f.flush()
        proc_args = [
            args.generator, "vectorize", "-c", "-i", f.name, "-a", 
            f"{args.output_dir}/cholesky{n}.adj.json", "-d", f"{args.output_dir}/cholesky{n}.data.json"
        ]
        if args.average:
            proc_args.append("-A")
        proc = await asyncio.subprocess.create_subprocess_exec(*proc_args)
        await proc.wait()

async def gen_all(args):
    tasks = []
    for n in range(args.n_start, args.n_end + 1, args.n_step):
        src = gen(n)
        tasks.append(single_task(args, src, n))
    for i in range(0, len(tasks), args.batch_size):
        lauched = [asyncio.create_task(t) for t in tasks[i:i+args.batch_size]]
        await asyncio.gather(*lauched)
        

def main():
    parser = argparse.ArgumentParser(description='Generate cholesky')
    parser.add_argument('--n-start', type=int, default=5)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--n-end', type=int, default=512)
    parser.add_argument('--output-dir', type=str, default="/tmp")
    parser.add_argument('--generator', type=str, default="target/release/slap")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--average', action='store_true')
    args = parser.parse_args()
    asyncio.run(gen_all(args))


if __name__ == "__main__":
    main()

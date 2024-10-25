import argparse
import asyncio
import aiofiles

SRC = """
module { 
    func.func @floyd_warshall(%path: memref<[:N:]x[:N:]xi32>) {
        affine.for %k = 0 to [:N:] {
            affine.for %i = 0 to [:N:] {
                affine.for %j = 0 to [:N:] {
                    %ij = affine.load %path[%i, %j] : memref<[:N:]x[:N:]xi32>
                    %ik = affine.load %path[%i, %k] : memref<[:N:]x[:N:]xi32>
                    %kj = affine.load %path[%k, %j] : memref<[:N:]x[:N:]xi32>
                    
                    %new = arith.addi %ik, %kj : i32
                    
                    %cond = arith.cmpi ule, %new, %ij : i32
                    
                    %final_result = arith.select %cond, %new, %ij : i32
                    
                    affine.store %final_result, %path[%i, %j] : memref<[:N:]x[:N:]xi32>
                }
            }
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
            f"{args.output_dir}/floyd_warshall{n}.adj.json", "-d", f"{args.output_dir}/floyd_warshall{n}.data.json"
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
    parser = argparse.ArgumentParser(description='Generate floyd_warshall')
    parser.add_argument('--n-start', type=int, default=5)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--n-end', type=int, default=128)
    parser.add_argument('--output-dir', type=str, default="/tmp")
    parser.add_argument('--generator', type=str, default="target/release/slap")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--average', action='store_true')
    args = parser.parse_args()
    asyncio.run(gen_all(args))


if __name__ == "__main__":
    main()

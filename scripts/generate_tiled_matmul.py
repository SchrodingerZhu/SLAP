import argparse
import asyncio
import aiofiles

SRC = """
module {
  func.func @matmul_tiled(%A: memref<[:M:]x[:K:]xf32>, %B: memref<[:K:]x[:N:]xf32>, %C: memref<[:M:]x[:N:]xf32>) {
    // Outer loops iterate over 16x16 tiles in both row and column of matrix C
    affine.for %ii = 0 to [:M:] step [:TM:] {
      affine.for %jj = 0 to [:N:] step [:TN:] {

        // Inner loop processes the product for each tile
        affine.for %kk = 0 to [:K:] step [:TK:] {
          // Perform the 16x16 matrix multiplication for this block
          affine.for %i = 0 to [:TM:] {
            affine.for %j = 0 to [:TN:] {
              affine.for %k = 0 to [:TK:] {
                // Load the current value of C[%ii + %i, %jj + %j] (accumulation)
                %c_val = affine.load %C[%ii + %i, %jj + %j] : memref<[:M:]x[:N:]xf32>
                
                // Load values from A and B with the respective offsets
                %a_val = affine.load %A[%ii + %i, %kk + %k] : memref<[:M:]x[:K:]xf32>
                %b_val = affine.load %B[%kk + %k, %jj + %j] : memref<[:K:]x[:N:]xf32>

                // Multiply and accumulate
                %prod = arith.mulf %a_val, %b_val : f32
                %c_new = arith.addf %c_val, %prod : f32

                // Store the updated value in C
                affine.store %c_new, %C[%ii + %i, %jj + %j] : memref<[:M:]x[:N:]xf32>
              }
            }
          }
        }
      }
    } { slap.extract }
    return
  }
}
"""

def gen(M, K, N, TM, TN, TK):
    src = SRC.replace("[:M:]", str(M)).replace("[:K:]", str(K)).replace("[:N:]", str(N))
    src = src.replace("[:TM:]", str(TM)).replace("[:TN:]", str(TN)).replace("[:TK:]", str(TK))
    return src

async def single_task(args, src, M, K, N, TM, TN, TK):
    async with aiofiles.tempfile.NamedTemporaryFile("w") as f:
        await f.write(src)
        await f.flush()
        proc_args = [
            args.generator, "vectorize", "-c", "-i", f.name, "-a", 
            f"{args.output_dir}/tiled_matmul_{M}_{K}_{N}_{TM}_{TN}_{TK}.adj.json", "-d", f"{args.output_dir}/tiled_matmul_{M}_{K}_{N}_{TM}_{TN}_{TK}.data.json"
        ]
        if args.average:
            proc_args.append("-A")
        proc = await asyncio.subprocess.create_subprocess_exec(*proc_args)
        await proc.wait()

async def gen_all(args):
    tasks = []
    for M in range(args.m_start, args.m_end + 1, args.m_step):
        for K in range(args.k_start, args.k_end + 1, args.k_step):
            for N in range(args.n_start, args.n_end + 1, args.n_step):
                for TM in range(1, M + 1):
                    for TN in range(1, N + 1):
                        for TK in range(1, K + 1):
                            if M % TM != 0 or N % TN != 0 or K % TK != 0:
                                continue
                            src = gen(M, K, N, TM, TN, TK)
                            tasks.append(single_task(args, src, M, K, N, TM, TN, TK))
    for i in range(0, len(tasks), args.batch_size):
        lauched = [asyncio.create_task(t) for t in tasks[i:i+args.batch_size]]
        await asyncio.gather(*lauched)
        

def main():
    # start N, start M, start K, step N, step M, step K
    # start I, start J, start K, step I, step J, step K
    parser = argparse.ArgumentParser(description='Generate a tiled matrix multiplication function')
    parser.add_argument('--m-start', type=int, default=10)
    parser.add_argument('--m-step', type=int, default=10)
    parser.add_argument('--m-end', type=int, default=100)
    parser.add_argument('--k-start', type=int, default=10)
    parser.add_argument('--k-step', type=int, default=10)
    parser.add_argument('--k-end', type=int, default=100)
    parser.add_argument('--n-start', type=int, default=10)
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--n-end', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default="/tmp")
    parser.add_argument('--generator', type=str, default="target/release/slap")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--average', action='store_true')
    args = parser.parse_args()
    asyncio.run(gen_all(args))


if __name__ == "__main__":
    main()

import argparse
import tempfile
import subprocess
import asyncio
import aiofiles

SRC = """
module {
  func.func @gemm(%A: memref<[:M:]x[:K:]xf32>, %B: memref<[:K:]x[:N:]xf32>, %C: memref<[:M:]x[:N:]xf32>, %alpha: f32, %beta: f32) {
    affine.for %i = 0 to [:M:] step [:SI:] {
      affine.for %j = 0 to [:N:] step [:SJ:] {
        // Load the value from C and scale it by beta
        %c_val = affine.load %C[%i, %j] : memref<[:M:]x[:N:]xf32>
        %c_scaled = arith.mulf %c_val, %beta : f32
        
        // Initialize the accumulator
        %acc0 = arith.constant 0.0 : f32
        %sum = affine.for %k = 0 to [:K:] step [:SK:] iter_args(%acc = %acc0) -> f32 {
          // Load values from A and B
          %a_val = affine.load %A[%i, %k] : memref<[:M:]x[:K:]xf32>
          %b_val = affine.load %B[%k, %j] : memref<[:K:]x[:N:]xf32>
          
          // Multiply and accumulate
          %prod = arith.mulf %a_val, %b_val : f32
          %new_acc = arith.addf %acc, %prod : f32
          
          // Yield the new accumulated value
          affine.yield %new_acc : f32
        }
        
        // Multiply the sum by alpha
        %result = arith.mulf %sum, %alpha : f32
        
        // Add the scaled C matrix value to the result
        %final_val = arith.addf %c_scaled, %result : f32
        
        // Store the final result back to matrix C
        affine.store %final_val, %C[%i, %j] : memref<[:M:]x[:N:]xf32>
      }
    } { slap.extract }
    return
  }
}
"""

def gen(M, K, N, SI, SJ, SK):
    src = SRC.replace("[:M:]", str(M)).replace("[:K:]", str(K)).replace("[:N:]", str(N))
    src = src.replace("[:SI:]", str(SI)).replace("[:SJ:]", str(SJ)).replace("[:SK:]", str(SK))
    return src

async def single_task(args, src, M, K, N, SI, SJ, SK):
    async with aiofiles.tempfile.NamedTemporaryFile("w") as f:
        await f.write(src)
        await f.flush()
        proc = await asyncio.subprocess.create_subprocess_exec(args.generator, "vectorize", "-i", f.name, "-a", 
            f"{args.output_dir}/matmul_{M}_{K}_{N}_{SI}_{SJ}_{SK}.adj.json", "-d", f"{args.output_dir}/matmul_{M}_{K}_{N}_{SI}_{SJ}_{SK}.data.json")
        await proc.wait()

async def gen_all(args):
    tasks = []
    for M in range(args.m_start, args.m_end + 1, args.m_step):
        for K in range(args.k_start, args.k_end + 1, args.k_step):
            for N in range(args.n_start, args.n_end + 1, args.n_step):
                for SI in range(args.si_start, args.si_end + 1, args.si_step):
                    for SJ in range(args.sj_start, args.sj_end + 1, args.sj_step):
                        for SK in range(args.sk_start, args.sk_end + 1, args.sk_step):
                            src = gen(M, K, N, SI, SJ, SK)
                            tasks.append(single_task(args, src, M, K, N, SI, SJ, SK))
    for i in range(0, len(tasks), args.batch_size):
        lauched = [asyncio.create_task(t) for t in tasks[i:i+args.batch_size]]
        await asyncio.gather(*lauched)
        

def main():
    # start N, start M, start K, step N, step M, step K
    # start I, start J, start K, step I, step J, step K
    parser = argparse.ArgumentParser(description='Generate a matrix multiplication function')
    parser.add_argument('--m-start', type=int, default=10)
    parser.add_argument('--m-step', type=int, default=10)
    parser.add_argument('--m-end', type=int, default=100)
    parser.add_argument('--k-start', type=int, default=10)
    parser.add_argument('--k-step', type=int, default=10)
    parser.add_argument('--k-end', type=int, default=100)
    parser.add_argument('--n-start', type=int, default=10)
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--n-end', type=int, default=100)
    parser.add_argument('--si-start', type=int, default=1)
    parser.add_argument('--si-step', type=int, default=1)
    parser.add_argument('--si-end', type=int, default=5)
    parser.add_argument('--sj-start', type=int, default=1)
    parser.add_argument('--sj-step', type=int, default=1)
    parser.add_argument('--sj-end', type=int, default=5)
    parser.add_argument('--sk-start', type=int, default=1)
    parser.add_argument('--sk-step', type=int, default=1)
    parser.add_argument('--sk-end', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default="/tmp")
    parser.add_argument('--generator', type=str, default="target/release/slap")
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    asyncio.run(gen_all(args))


if __name__ == "__main__":
    main()

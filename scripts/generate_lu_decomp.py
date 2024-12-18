import argparse
import asyncio
import aiofiles

SRC = """
!memref = memref<[:SIZE:]x[:SIZE:]xf32>
module {
  func.func @lu_decomposition(%A: !memref) {
    // Iterate over columns (k-th column of A)
    affine.for %k = 0 to [:SIZE:] {
      // Update the upper triangular part (U) in the k-th row
      affine.for %j = affine_map<(d0) -> (d0)> (%k) to [:SIZE:] {
        // A[k, j] remains as U[k, j] for k-th row
        %A_kj = affine.load %A[%k, %j] : !memref
        affine.store %A_kj, %A[%k, %j] : !memref
      }

      // Update the lower triangular part (L) in the k-th column
      affine.for %i = affine_map<(d0) -> (d0 + 1)> (%k) to [:SIZE:] {
        // Compute L[i, k] = A[i, k] / U[k, k]
        %A_ik = affine.load %A[%i, %k] : !memref
        %A_kk = affine.load %A[%k, %k] : !memref
        %L_ik = arith.divf %A_ik, %A_kk : f32
        affine.store %L_ik, %A[%i, %k] : !memref

        // Update the rest of the A matrix using the new L and U values
        affine.for %j = affine_map<(d0) -> (d0 + 1)> (%k) to [:SIZE:] {
          // A[i, j] -= L[i, k] * U[k, j]
          %A_ij = affine.load %A[%i, %j] : !memref
          %U_kj = affine.load %A[%k, %j] : !memref
          %L_ik_new = affine.load %A[%i, %k] : !memref
          %product = arith.mulf %L_ik_new, %U_kj : f32
          %A_ij_new = arith.subf %A_ij, %product : f32
          affine.store %A_ij_new, %A[%i, %j] : !memref
        }
      }
    } { slap.extract }
    return
  }
}
"""

def gen(SIZE):
    src = SRC.replace("[:SIZE:]", str(SIZE))
    return src

async def single_task(args, src, SIZE):
    async with aiofiles.tempfile.NamedTemporaryFile("w") as f:
        await f.write(src)
        await f.flush()
        proc_args = [
            args.generator, "vectorize", "-c", "-i", f.name, "-a", 
            f"{args.output_dir}/lu_decomp_{SIZE}.adj.json", "-d", f"{args.output_dir}/lu_decomp_{SIZE}.data.json"
        ]
        if args.average:
            proc_args.append("-A")
        proc = await asyncio.subprocess.create_subprocess_exec(*proc_args)
        await proc.wait()

async def gen_all(args):
    tasks = []
    for s in range(args.size_start, args.size_end + 1, args.size_step):
        src = gen(s)
        tasks.append(single_task(args, src, s))
    for i in range(0, len(tasks), args.batch_size):
        lauched = [asyncio.create_task(t) for t in tasks[i:i+args.batch_size]]
        await asyncio.gather(*lauched)
        

def main():
    parser = argparse.ArgumentParser(description='Generate a LU decomposition function')
    parser.add_argument('--size-start', type=int, default=2)
    parser.add_argument('--size-step', type=int, default=1)
    parser.add_argument('--size-end', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default="/tmp")
    parser.add_argument('--generator', type=str, default="target/release/slap")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--average', action='store_true')
    args = parser.parse_args()
    asyncio.run(gen_all(args))


if __name__ == "__main__":
    main()

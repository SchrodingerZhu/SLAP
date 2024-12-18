import argparse
import asyncio
import aiofiles

SRC = """
!memref = memref<[:SIZE:]x[:SIZE:]xf32>
module {
  func.func @heat_distribution(%A: !memref) {
    // Iterate over 100 rounds
    affine.for %r = 0 to [:ROUND:] {
      
      // First update even-indexed elements (skipping the boundary elements)
      affine.for %i = 1 to [:SIZE-1:] step 2 {
        affine.for %j = 1 to [:SIZE-1:] step 2 {
          // Load neighbors (top, bottom, left, right, and center)
          %top    = affine.load %A[%i - 1, %j] : !memref
          %bottom = affine.load %A[%i + 1, %j] : !memref
          %left   = affine.load %A[%i, %j - 1] : !memref
          %right  = affine.load %A[%i, %j + 1] : !memref
          %center = affine.load %A[%i, %j] : !memref

          // Compute the sum and average the values for heat distribution
          %c5 = arith.constant 5.0 : f32
          %sum1    = arith.addf %top, %bottom : f32
          %sum2    = arith.addf %sum1, %left : f32
          %sum3    = arith.addf %sum2, %right : f32
          %sum4    = arith.addf %sum3, %center : f32
          %avg     = arith.divf %sum4, %c5 : f32

          // Store the updated value back to matrix A
          affine.store %avg, %A[%i, %j] : !memref
        }
      }

      // Then update odd-indexed elements (skipping the boundary elements)
      affine.for %i = 2 to [:SIZE-2:] step 2 {
        affine.for %j = 2 to [:SIZE-2:] step 2 {
          // Load neighbors (top, bottom, left, right, and center)
          %top_odd    = affine.load %A[%i - 1, %j] : !memref
          %bottom_odd = affine.load %A[%i + 1, %j] : !memref
          %left_odd   = affine.load %A[%i, %j - 1] : !memref
          %right_odd  = affine.load %A[%i, %j + 1] : !memref
          %center_odd = affine.load %A[%i, %j] : !memref

          // Compute the sum and average the values for heat distribution
          %c5 = arith.constant 5.0 : f32
          %sum1_odd    = arith.addf %top_odd, %bottom_odd : f32
          %sum2_odd    = arith.addf %sum1_odd, %left_odd : f32
          %sum3_odd    = arith.addf %sum2_odd, %right_odd : f32
          %sum4_odd    = arith.addf %sum3_odd, %center_odd : f32
          %avg_odd     = arith.divf %sum4_odd, %c5 : f32

          // Store the updated value back to matrix A
          affine.store %avg_odd, %A[%i, %j] : !memref
        }
      }
    } { slap.extract }
    return
  }
}
"""

def gen(SIZE, ROUND):
    src = SRC.replace("[:SIZE:]", str(SIZE)).replace("[:ROUND:]", str(ROUND))
    src = src.replace("[:SIZE-1:]", str(SIZE - 1)).replace("[:SIZE-2:]", str(SIZE - 2))
    return src

async def single_task(args, src, SIZE, ROUND):
    async with aiofiles.tempfile.NamedTemporaryFile("w") as f:
        await f.write(src)
        await f.flush()
        proc_args = [
            args.generator, "vectorize", "-c", "-i", f.name, "-a", 
            f"{args.output_dir}/heat_dist_{SIZE}_{ROUND}.adj.json", "-d", f"{args.output_dir}/heat_dist_{SIZE}_{ROUND}.data.json"
        ]
        if args.average:
            proc_args.append("-A")
        proc = await asyncio.subprocess.create_subprocess_exec(*proc_args)
        await proc.wait()

async def gen_all(args):
    tasks = []
    for s in range(args.size_start, args.size_end + 1, args.size_step):
        for r in range(args.round_start, args.round_end + 1, args.round_step):
            src = gen(s, r)
            tasks.append(single_task(args, src, s, r))
    for i in range(0, len(tasks), args.batch_size):
        lauched = [asyncio.create_task(t) for t in tasks[i:i+args.batch_size]]
        await asyncio.gather(*lauched)
        

def main():
    parser = argparse.ArgumentParser(description='Generate a heat distribution multiplication function')
    parser.add_argument('--size-start', type=int, default=4)
    parser.add_argument('--size-step', type=int, default=1)
    parser.add_argument('--size-end', type=int, default=500)
    parser.add_argument('--round-start', type=int, default=10)
    parser.add_argument('--round-step', type=int, default=1)
    parser.add_argument('--round-end', type=int, default=200)
    parser.add_argument('--output-dir', type=str, default="/tmp")
    parser.add_argument('--generator', type=str, default="target/release/slap")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--average', action='store_true')
    args = parser.parse_args()
    asyncio.run(gen_all(args))


if __name__ == "__main__":
    main()

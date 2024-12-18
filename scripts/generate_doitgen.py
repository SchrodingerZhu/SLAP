import argparse
import asyncio
import aiofiles

SRC = """
module {
  func.func @doitgen(%A: memref<[:R:]x[:Q:]x[:P:]xf32>, %C4: memref<[:P:]x[:P:]xf32>, %sum: memref<[:P:]xf32>) {
    // Loop over r
    affine.for %r = 0 to [:R:] {
      // Loop over q
      affine.for %q = 0 to [:Q:] {
        // Initialize sum to zero for each p
        affine.for %p = 0 to [:P:] {
          // Set sum[p] to 0.0
          %zero = arith.constant 0.0 : f32
          affine.store %zero, %sum[%p] : memref<[:P:]xf32>

          affine.for %s = 0 to [:P:] {
            // Compute sum[p] += A[r][q][s] * C4[s][p]
            %A_val = affine.load %A[%r, %q, %s] : memref<[:R:]x[:Q:]x[:P:]xf32>
            %C4_val = affine.load %C4[%s, %p] : memref<[:P:]x[:P:]xf32>
            %product = arith.mulf %A_val, %C4_val : f32
            %current_sum = affine.load %sum[%p] : memref<[:P:]xf32>
            %new_sum = arith.addf %current_sum, %product : f32
            affine.store %new_sum, %sum[%p] : memref<[:P:]xf32>
          }
        }
        affine.for %p = 0 to [:P:] {
          // Assign sum to A[r][q][p]
          %final_value = affine.load %sum[%p] : memref<[:P:]xf32>
          affine.store %final_value, %A[%r, %q, %p] : memref<[:R:]x[:Q:]x[:P:]xf32>
        }
      }
    }{ slap.extract }
    return
  }
}


"""


def gen(r, q, p):
    src = SRC.replace("[:R:]", str(r)).replace("[:Q:]", str(q)).replace("[:P:]", str(p))
    return src

async def single_task(args, src, r, q, p):
    async with aiofiles.tempfile.NamedTemporaryFile("w") as f:
        await f.write(src)
        await f.flush()
        proc_args = [
            args.generator, "vectorize", "-c", "-i", f.name, "-a", 
            f"{args.output_dir}/doitgen_{r}_{q}_{p}.adj.json", "-d", f"{args.output_dir}/doitgen_{r}_{q}_{p}.data.json"
        ]
        if args.average:
            proc_args.append("-A")
        proc = await asyncio.subprocess.create_subprocess_exec(*proc_args)
        await proc.wait()

async def gen_all(args):
    tasks = []
    for r in range(args.r_start, args.r_end + 1, args.r_step):
        for q in range(args.q_start, args.q_end + 1, args.q_step):
            for p in range(args.p_start, args.p_end + 1, args.p_step):
                src = gen(r, q, p)
                tasks.append(single_task(args, src, r, q, p))
    for i in range(0, len(tasks), args.batch_size):
        lauched = [asyncio.create_task(t) for t in tasks[i:i+args.batch_size]]
        await asyncio.gather(*lauched)
        

def main():
    parser = argparse.ArgumentParser(description='Generate doitgen')
    parser.add_argument('--r-start', type=int, default=5)
    parser.add_argument('--r-step', type=int, default=1)
    parser.add_argument('--r-end', type=int, default=50)
    parser.add_argument('--q-start', type=int, default=5)
    parser.add_argument('--q-step', type=int, default=1)
    parser.add_argument('--q-end', type=int, default=20)
    parser.add_argument('--p-start', type=int, default=5)
    parser.add_argument('--p-step', type=int, default=1)
    parser.add_argument('--p-end', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default="/tmp")
    parser.add_argument('--generator', type=str, default="target/release/slap")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--average', action='store_true')
    args = parser.parse_args()
    asyncio.run(gen_all(args))


if __name__ == "__main__":
    main()

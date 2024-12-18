import argparse
import asyncio
import aiofiles

SRC = """
!memref = memref<[:SIZE:]x[:SIZE:]xf32>
module {
  func.func @stencil_kernel(%A: !memref, %B: !memref) {
    affine.for %i = 1 to [:SIZE-1:] {
      affine.for %j = 1 to [:SIZE-1:] {
        // Load the neighboring values and the center value
        %top    = affine.load %A[%i - 1, %j] : !memref
        %bottom = affine.load %A[%i + 1, %j] : !memref
        %left   = affine.load %A[%i, %j - 1] : !memref
        %right  = affine.load %A[%i, %j + 1] : !memref
        %center = affine.load %A[%i, %j] : !memref

        // Perform the sum of the loaded values
        %sum1 = arith.addf %top, %bottom : f32
        %sum2 = arith.addf %left, %right : f32
        %sum3 = arith.addf %sum1, %sum2 : f32
        %sum4 = arith.addf %sum3, %center : f32

        // Compute the average (sum / 5.0)
        %c5 = arith.constant 5.0 : f32
        %avg = arith.divf %sum4, %c5 : f32

        // Store the result in the output array
        affine.store %avg, %B[%i, %j] : !memref
      }
    } { slap.extract }
    return
  }
}
"""

def gen(SIZE):
    src = SRC.replace("[:SIZE:]", str(SIZE)).replace("[:SIZE-1:]", str(SIZE - 1))
    return src

async def single_task(args, src, SIZE):
    async with aiofiles.tempfile.NamedTemporaryFile("w") as f:
        await f.write(src)
        await f.flush()
        proc_args = [
            args.generator, "vectorize", "-c", "-i", f.name, "-a", 
            f"{args.output_dir}/stencil_{SIZE}.adj.json", "-d", f"{args.output_dir}/stencil_{SIZE}.data.json"
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
    parser = argparse.ArgumentParser(description='Generate a stencil multiplication function')
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

import argparse
import asyncio
import aiofiles

SRC = """
!memref = memref<[:M:]x[:N:]xf32>
module {
  func.func @conv2d_kernel(%input: !memref, %filter: memref<[:X:]x[:Y:]xf32>, %output: memref<[:M-X+1:]x[:N-Y+1:]xf32>) {
    affine.for %i = 0 to [:M-X+1:] {
      affine.for %j = 0 to [:N-Y+1:] {
        // Use affine.parallel to accumulate values into %acc using iter_args
        %zero = arith.constant 0.0 : f32
        %acc = affine.for %fi = 0 to [:X:] iter_args(%acc = %zero) -> (f32) {
          %acc_inner = affine.for %fj = 0 to [:Y:] iter_args(%acc_inner = %acc) -> (f32) {
            // Load filter value
            %filter_val = affine.load %filter[%fi, %fj] : memref<[:X:]x[:Y:]xf32>

            // Load corresponding input value from the input matrix
            %input_val = affine.load %input[%i + %fi, %j + %fj] : !memref

            // Multiply input value with filter value
            %prod = arith.mulf %input_val, %filter_val : f32

            // Add product to the accumulator
            %new_acc = arith.addf %acc_inner, %prod : f32
            affine.yield %new_acc : f32
          }
          affine.yield %acc_inner : f32
        }

        // Store the accumulated result in the output matrix
        affine.store %acc, %output[%i, %j] : memref<[:M-X+1:]x[:N-Y+1:]xf32>
      }
    } { slap.extract }
    return
  }
}

"""

def gen(M, N, X, Y):
    src = SRC.replace("[:M:]", str(M)).replace("[:N:]", str(N)).replace("[:X:]", str(X)).replace("[:Y:]", str(Y))
    src = src.replace("[:M-X+1:]", str(M - X + 1)).replace("[:N-Y+1:]", str(N - Y + 1))
    return src

async def single_task(args, src, M, N, X, Y):
    async with aiofiles.tempfile.NamedTemporaryFile("w") as f:
        await f.write(src)
        await f.flush()
        proc_args = [
            args.generator, "vectorize", "-c", "-i", f.name, "-a", 
            f"{args.output_dir}/convolution_{M}_{N}_{X}_{Y}.adj.json", "-d", f"{args.output_dir}/convolution_{M}_{N}_{X}_{Y}.data.json"
        ]
        if args.average:
            proc_args.append("-A")
        proc = await asyncio.subprocess.create_subprocess_exec(*proc_args)
        await proc.wait()

async def gen_all(args):
    tasks = []
    for m in range(args.m_start, args.m_end + 1, args.m_step):
        for n in range(args.n_start, args.n_end + 1, args.n_step):
            for x in range(args.x_start, args.x_end + 1, args.x_step):
                for y in range(args.y_start, args.y_end + 1, args.y_step):
                    src = gen(m, n, x, y)
                    tasks.append(single_task(args, src, m, n, x, y))
    for i in range(0, len(tasks), args.batch_size):
        lauched = [asyncio.create_task(t) for t in tasks[i:i+args.batch_size]]
        await asyncio.gather(*lauched)
        

def main():
    parser = argparse.ArgumentParser(description='Generate a convolution function')
    parser.add_argument('--m-start', type=int, default=16)
    parser.add_argument('--m-step', type=int, default=1)
    parser.add_argument('--m-end', type=int, default=128)
    parser.add_argument('--n-start', type=int, default=16)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--n-end', type=int, default=128)
    parser.add_argument('--x-start', type=int, default=3)
    parser.add_argument('--x-step', type=int, default=2)
    parser.add_argument('--x-end', type=int, default=15)
    parser.add_argument('--y-start', type=int, default=3)
    parser.add_argument('--y-step', type=int, default=2)
    parser.add_argument('--y-end', type=int, default=15)
    parser.add_argument('--output-dir', type=str, default="/tmp")
    parser.add_argument('--generator', type=str, default="target/release/slap")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--average', action='store_true')
    args = parser.parse_args()
    asyncio.run(gen_all(args))


if __name__ == "__main__":
    main()

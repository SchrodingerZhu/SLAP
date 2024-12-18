import argparse
import asyncio
import aiofiles

SRC = """
module {
  func.func @mvt(%A: memref<[:N:]x[:N:]xf32>, %x1: memref<[:N:]xf32>, %y1: memref<[:N:]xf32>, %y2: memref<[:N:]xf32>) {
    %c0 = arith.constant 0.0 : f32

    // First loop nest
    affine.for %i = 0 to [:N:] {
      affine.for %j = 0 to [:N:] {
        %a = affine.load %x1[%i] : memref<[:N:]xf32>
        %A_ij = affine.load %A[%i, %j] : memref<[:N:]x[:N:]xf32>
        %y_1_j = affine.load %y1[%j] : memref<[:N:]xf32>
        %mul = arith.mulf %A_ij, %y_1_j : f32
        %sum = arith.addf %a, %mul : f32
        affine.store %sum, %x1[%i] : memref<[:N:]xf32>
      }
    }

    // Second loop nest
    affine.for %i = 0 to [:N:] {
      affine.for %j = 0 to [:N:] {
        %a = affine.load %y2[%i] : memref<[:N:]xf32>
        %A_ji = affine.load %A[%j, %i] : memref<[:N:]x[:N:]xf32>
        %y_2_j = affine.load %y1[%j] : memref<[:N:]xf32>
        %mul = arith.mulf %A_ji, %y_2_j : f32
        %sum = arith.addf %a, %mul : f32
        affine.store %sum, %y2[%i] : memref<[:N:]xf32>
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
            f"{args.output_dir}/mvt{n}.adj.json", "-d", f"{args.output_dir}/mvt{n}.data.json"
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
    parser = argparse.ArgumentParser(description='Generate mvt')
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

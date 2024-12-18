import argparse
import asyncio
import aiofiles

SRC = """
module {
    // Constants

    // Function definition
    func.func @bicg(%argA: memref<[:N:]x[:M:]xf32>, %argS: memref<[:M:]xf32>, %argR: memref<[:N:]xf32>, %argQ: memref<[:N:]xf32>, %argP: memref<[:M:]xf32>) {
        %c0 = arith.constant 0.0 : f32

        // Initialize s[i] to 0 for i in [0, M)
        affine.for %i = 0 to [:M:] {
            affine.store %c0, %argS[%i] : memref<[:M:]xf32>
        }

        // Outer loop: for (i = 0; i < N; i++)
        affine.for %i = 0 to [:N:] {
            // Initialize q[i] to 0
            affine.store %c0, %argQ[%i] : memref<[:N:]xf32>

            // Inner loop: for (j = 0; j < M; j++)
            affine.for %j = 0 to [:M:] {
                // s[j] = s[j] + r[i] * A[i][j];
                %s_j = affine.load %argS[%j] : memref<[:M:]xf32>
                %r_i = affine.load %argR[%i] : memref<[:N:]xf32>
                %a_ij = affine.load %argA[%i, %j] : memref<[:N:]x[:M:]xf32>
                %prod_r_a = arith.mulf %r_i, %a_ij : f32
                %new_s_j = arith.addf %s_j, %prod_r_a : f32
                affine.store %new_s_j, %argS[%j] : memref<[:M:]xf32>

                // q[i] = q[i] + A[i][j] * p[j];
                %q_i = affine.load %argQ[%i] : memref<[:N:]xf32>
                %p_j = affine.load %argP[%j] : memref<[:M:]xf32>
                %prod_a_p = arith.mulf %a_ij, %p_j : f32
                %new_q_i = arith.addf %q_i, %prod_a_p : f32
                affine.store %new_q_i, %argQ[%i] : memref<[:N:]xf32>
            }
        } { slap.extract }

    return
    }
}

"""


def gen(n, m):
    src = SRC.replace("[:N:]", str(n)).replace("[:M:]", str(m))
    return src

async def single_task(args, src, n, m):
    async with aiofiles.tempfile.NamedTemporaryFile("w") as f:
        await f.write(src)
        await f.flush()
        proc_args = [
            args.generator, "vectorize", "-c", "-i", f.name, "-a", 
            f"{args.output_dir}/bicg{n}.adj.json", "-d", f"{args.output_dir}/bicg{n}_{m}.data.json"
        ]
        if args.average:
            proc_args.append("-A")
        proc = await asyncio.subprocess.create_subprocess_exec(*proc_args)
        await proc.wait()

async def gen_all(args):
    tasks = []
    for n in range(args.n_start, args.n_end + 1, args.n_step):
        for m in range(args.m_start, args.m_end + 1, args.m_step):
            src = gen(n, m)
            tasks.append(single_task(args, src, n, m))
    for i in range(0, len(tasks), args.batch_size):
        lauched = [asyncio.create_task(t) for t in tasks[i:i+args.batch_size]]
        await asyncio.gather(*lauched)
        

def main():
    parser = argparse.ArgumentParser(description='Generate bicg')
    parser.add_argument('--n-start', type=int, default=5)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--n-end', type=int, default=128)
    parser.add_argument('--m-start', type=int, default=5)
    parser.add_argument('--m-step', type=int, default=1)
    parser.add_argument('--m-end', type=int, default=128)
    parser.add_argument('--output-dir', type=str, default="/tmp")
    parser.add_argument('--generator', type=str, default="target/release/slap")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--average', action='store_true')
    args = parser.parse_args()
    asyncio.run(gen_all(args))


if __name__ == "__main__":
    main()

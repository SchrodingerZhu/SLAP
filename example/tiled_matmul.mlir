module {
  func.func @matmul_tiled(%A: memref<256x256xf32>, %B: memref<256x256xf32>, %C: memref<256x256xf32>) {
    // Outer loops iterate over 16x16 tiles in both row and column of matrix C
    affine.for %ii = 0 to 256 step 16 {
      affine.for %jj = 0 to 256 step 16 {

        // Inner loop processes the product for each tile
        affine.for %kk = 0 to 256 step 16 {
          // Perform the 16x16 matrix multiplication for this block
          affine.for %i = 0 to 16 {
            affine.for %j = 0 to 16 {
              affine.for %k = 0 to 16 {
                // Load the current value of C[%ii + %i, %jj + %j] (accumulation)
                %c_val = affine.load %C[%ii + %i, %jj + %j] : memref<256x256xf32>
                
                // Load values from A and B with the respective offsets
                %a_val = affine.load %A[%ii + %i, %kk + %k] : memref<256x256xf32>
                %b_val = affine.load %B[%kk + %k, %jj + %j] : memref<256x256xf32>

                // Multiply and accumulate
                %prod = arith.mulf %a_val, %b_val : f32
                %c_new = arith.addf %c_val, %prod : f32

                // Store the updated value in C
                affine.store %c_new, %C[%ii + %i, %jj + %j] : memref<256x256xf32>
              }
            }
          }
        }
      }
    } { slap.extract }
    return
  }
}

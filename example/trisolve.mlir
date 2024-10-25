module {
  func.func @kernel(%arg0: memref<512xf32>, %arg1: memref<512xf32>, %arg2: memref<512x512xf32>) {
    %c0 = arith.constant 0.0 : f32

    affine.for %i = 0 to 512 {
      // x[i] = b[i]
      %b_i = affine.load %arg1[%i] : memref<512xf32>
      affine.store %b_i, %arg0[%i] : memref<512xf32>

      // for (j = 0; j < i; j++)
      
      affine.for %j = 0 to %i {
        // x[i] -= L[i][j] * x[j]
        %x_i = affine.load %arg0[%i] : memref<512xf32>
        %L_ij = affine.load %arg2[%i, %j] : memref<512x512xf32>
        %x_j = affine.load %arg0[%j] : memref<512xf32>
        %mul = arith.mulf %L_ij, %x_j : f32
        %sub = arith.subf %x_i, %mul : f32
        affine.store %sub, %arg0[%i] : memref<512xf32>
      }

      // x[i] = x[i] / L[i][i]
      %x_i_final = affine.load %arg0[%i] : memref<512xf32>
      %L_ii = affine.load %arg2[%i, %i] : memref<512x512xf32>
      %div = arith.divf %x_i_final, %L_ii : f32
      affine.store %div, %arg0[%i] : memref<512xf32>
    }
    return
  }
}

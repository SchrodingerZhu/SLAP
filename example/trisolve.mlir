module {
    func.func @trisolve(%b: memref<256xf32>, %L: memref<256x256xf32>, %x: memref<256xf32>) {
        // Allocate a local variable for loop index i and j
        affine.for %i = 0 to 256 {
        // x[i] = b[i]
        %b_i = affine.load %b[%i] : memref<256xf32>
        affine.store %b_i, %x[%i] : memref<256xf32>
        
        affine.for %j = 0 to affine_map<(d0) -> (d0)> (%i) {
            %L_ij = affine.load %L[%i, %j] : memref<256x256xf32>
            %x_j = affine.load %x[%j] : memref<256xf32>
            %prod = arith.mulf %L_ij, %x_j : f32
            %x_i = affine.load %x[%i] : memref<256xf32> 
            %tmp = arith.subf %x_i, %prod : f32
            affine.store %tmp, %x[%i] : memref<256xf32>
        }

        // x[i] = x[i] / L[i][i]
        %L_ii = affine.load %L[%i, %i] : memref<256x256xf32>
        %x_i_final = affine.load %x[%i] : memref<256xf32>
        %final_x_i = arith.divf %x_i_final, %L_ii : f32
        affine.store %final_x_i, %x[%i] : memref<256xf32>
        } { slap.extract }
    return
    }
}

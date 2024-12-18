module {
  func.func @mvt(%A: memref<256x256xf32>, %x1: memref<256xf32>, %y1: memref<256xf32>, %y2: memref<256xf32>) {
    %c0 = arith.constant 0.0 : f32

    // First loop nest
    affine.for %i = 0 to 256 {
      affine.for %j = 0 to 256 {
        %a = affine.load %x1[%i] : memref<256xf32>
        %A_ij = affine.load %A[%i, %j] : memref<256x256xf32>
        %y_1_j = affine.load %y1[%j] : memref<256xf32>
        %mul = arith.mulf %A_ij, %y_1_j : f32
        %sum = arith.addf %a, %mul : f32
        affine.store %sum, %x1[%i] : memref<256xf32>
      }
    }

    // Second loop nest
    affine.for %i = 0 to 256 {
      affine.for %j = 0 to 256 {
        %a = affine.load %y2[%i] : memref<256xf32>
        %A_ji = affine.load %A[%j, %i] : memref<256x256xf32>
        %y_2_j = affine.load %y1[%j] : memref<256xf32>
        %mul = arith.mulf %A_ji, %y_2_j : f32
        %sum = arith.addf %a, %mul : f32
        affine.store %sum, %y2[%i] : memref<256xf32>
      }
    } { slap.extract }
    return
  }
}

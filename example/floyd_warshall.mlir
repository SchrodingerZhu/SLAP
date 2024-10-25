module { 
    func.func @floyd_warshall(%path: memref<256x256xi32>) {
        affine.for %k = 0 to 256 {
            affine.for %i = 0 to 256 {
                affine.for %j = 0 to 256 {
                    %ij = affine.load %path[%i, %j] : memref<256x256xi32>
                    %ik = affine.load %path[%i, %k] : memref<256x256xi32>
                    %kj = affine.load %path[%k, %j] : memref<256x256xi32>
                    
                    %new = arith.addi %ik, %kj : i32
                    
                    %cond = arith.cmpi ule, %new, %ij : i32
                    
                    %final_result = arith.select %cond, %new, %ij : i32
                    
                    affine.store %final_result, %path[%i, %j] : memref<256x256xi32>
                }
            }
        } { slap.extract }
        return
    }
}

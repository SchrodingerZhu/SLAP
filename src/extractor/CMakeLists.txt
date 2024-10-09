
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_mlir_library(SLAPExtractor
  extractor.cpp    
  LINK_LIBS PUBLIC
    MLIRSupport
    MLIRIR
    ${dialect_libs}
    ${extension_libs}
)

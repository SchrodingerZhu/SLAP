use std::{path::PathBuf, str::FromStr};

fn main() {
    let mlir_path = std::env::var("MLIR_DIR")
        .or_else(|_| {
            for i in [20, 19, 18] {
                let hint = format!("/usr/lib/llvm-{}/lib/cmake/mlir/", i);
                if std::path::Path::new(&hint).exists() {
                    return Ok(hint);
                }
            }
            Err(std::env::VarError::NotPresent)
        })
        .unwrap();
    let dst = cmake::Config::new(".")
        .define("MLIR_DIR", &mlir_path)
        .build();
    println!(
        "cargo:rustc-link-search=native={}/build/src/extractor",
        dst.display()
    );
    let pathbuf = PathBuf::from_str(&mlir_path).unwrap();
    let libdir = pathbuf.parent().unwrap().parent().unwrap();
    println!("cargo:rustc-link-search=native={}", libdir.display());
    println!("cargo:rustc-link-lib=static=SLPNExtractor");
    println!("cargo:rustc-link-lib=dylib=MLIR");
    println!("cargo:rustc-link-lib=dylib=LLVM");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}", libdir.display());
}

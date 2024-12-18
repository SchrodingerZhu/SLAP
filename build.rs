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
    let dst = cmake::Config::new("cxx")
        .define("MLIR_DIR", &mlir_path)
        .build();
    println!(
        "cargo:rustc-link-search=native={}/build/src/extractor",
        dst.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/build/src/simulator",
        dst.display()
    );
    let pathbuf = PathBuf::from_str(&mlir_path).unwrap();
    let libdir = pathbuf.parent().unwrap().parent().unwrap();
    println!("cargo:rustc-link-search=native={}", libdir.display());
    println!("cargo:rustc-link-lib=static=SLAPExtractor");
    println!("cargo:rustc-link-lib=static=SLAPSimulator");
    println!("cargo:rustc-link-lib=dylib=MLIR");
    println!("cargo:rustc-link-lib=dylib=LLVM");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo::rustc-link-arg=-Wl,-rpath,{}", libdir.display());

    // find all libMLIRCAPI* in libdir
    for entry in std::fs::read_dir(libdir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            let filename = path.file_name().unwrap().to_str().unwrap();
            if filename.starts_with("libMLIRCAPI") {
                let libname = filename.trim_start_matches("lib").trim_end_matches(".a");
                println!("cargo:rustc-link-lib=static={}", libname);
            }
        }
    }
}

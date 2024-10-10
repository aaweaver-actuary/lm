#![allow(unused_imports)]
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    const FORTRAN_TARGET_DIRECTORY: &str = "target/fortran";

    // Specify the directory where the compiled Fortran library is located
    println!(
        "cargo:rustc-link-search=native={}",
        FORTRAN_TARGET_DIRECTORY
    );

    // Link the Fortran library. The library name should match the prefix of the .a file, e.g., libdqrls.a
    println!("cargo:rustc-link-lib=static=dqrls");
}

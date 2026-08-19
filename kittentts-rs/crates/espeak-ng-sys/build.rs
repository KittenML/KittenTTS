use std::env;
use std::path::PathBuf;

fn main() {
    let dst = cmake::Config::new("espeak-ng")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("USE_LIBPCAUDIO", "OFF")
        .define("USE_KLATT", "OFF")
        .define("USE_MBROLA", "OFF")
        .define("USE_ASYNC", "OFF")
        .define("ENABLE_TESTS", "OFF")
        .build();

    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let cmake_config = if profile == "release" {
        "Release"
    } else {
        "Debug"
    };

    // espeak-ng builds sub-libraries in specific subdirectories
    let mut search_paths = vec![
        dst.join("lib"),
        dst.join("build/src/ucd-tools"),
        dst.join("build/src/speechPlayer"),
    ];

    if cfg!(target_os = "windows") {
        // MSVC adds a config-specific subdirectory
        search_paths.push(dst.join("build/src/ucd-tools").join(cmake_config));
        search_paths.push(dst.join("build/src/speechPlayer").join(cmake_config));
        // Some versions might put espeak-ng.lib in lib/Debug too
        search_paths.push(dst.join("lib").join(cmake_config));
    }

    for path in &search_paths {
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }

    println!("cargo:rustc-link-lib=static=espeak-ng");
    println!("cargo:rustc-link-lib=static=ucd");
    // Piper links speechPlayer too
    println!("cargo:rustc-link-lib=static=speechPlayer");

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=dylib=user32");
        println!("cargo:rustc-link-lib=dylib=shell32");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=dylib=c++");
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-Iespeak-ng/src/include")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

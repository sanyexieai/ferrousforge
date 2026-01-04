//! 构建脚本
//! 
//! 用于配置 llama.cpp 的链接方式。
//! 
//! 支持两种方式：
//! 1. 静态链接：编译 llama.cpp 源码并静态链接
//! 2. 动态链接：链接系统已安装的 llama.cpp 库

use std::env;
use std::path::PathBuf;

fn main() {
    // 告诉 Cargo 如果这些文件改变，需要重新构建
    println!("cargo:rerun-if-changed=build.rs");
    
    // 检查是否启用了 llama-cpp feature
    let llama_cpp_enabled = env::var("CARGO_FEATURE_LLAMA_CPP").is_ok();
    let llama_cpp_dynamic = env::var("CARGO_FEATURE_LLAMA_CPP_DYNAMIC").is_ok();
    
    if llama_cpp_enabled || llama_cpp_dynamic {
        configure_llama_cpp();
    }
}

fn configure_llama_cpp() {
    let llama_cpp_dynamic = env::var("CARGO_FEATURE_LLAMA_CPP_DYNAMIC").is_ok();
    
    if llama_cpp_dynamic {
        // 动态链接模式：链接系统库
        configure_dynamic_linking();
    } else {
        // 静态链接模式：编译 llama.cpp 源码
        // 注意：这需要 llama.cpp 源码在项目中
        // 目前先配置动态链接作为默认
        configure_dynamic_linking();
    }
}

fn configure_dynamic_linking() {
    // 动态链接系统已安装的 llama.cpp 库
    
    // Windows
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=dylib=llama");
        // 如果库在特定路径，可以添加搜索路径
        // let lib_path = PathBuf::from("C:/path/to/llama.cpp/lib");
        // println!("cargo:rustc-link-search=native={}", lib_path.display());
    }
    
    // macOS
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=dylib=llama");
        // 如果使用 Homebrew 安装
        // println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
        // println!("cargo:rustc-link-search=native=/usr/local/lib");
    }
    
    // Linux
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=dylib=llama");
        // 如果库在特定路径
        // println!("cargo:rustc-link-search=native=/usr/local/lib");
    }
}

fn configure_static_linking() {
    // 静态链接模式：编译 llama.cpp 源码
    // 这需要 llama.cpp 源码在项目中（例如在 vendor/llama.cpp）
    
    let llama_cpp_path = PathBuf::from("vendor/llama.cpp");
    
    if !llama_cpp_path.exists() {
        eprintln!("Warning: llama.cpp source not found at {:?}", llama_cpp_path);
        eprintln!("Falling back to dynamic linking");
        configure_dynamic_linking();
        return;
    }
    
    // 使用 cc crate 编译 llama.cpp
    // 注意：这需要添加 cc 依赖到 build-dependencies
    // 这里提供一个框架，实际实现需要根据 llama.cpp 的构建系统调整
    
    println!("cargo:warning=Static linking of llama.cpp is not yet fully implemented");
    println!("cargo:warning=Please use dynamic linking or llama-cpp-rs crate instead");
    configure_dynamic_linking();
}


//! 构建脚本
//! 
//! 用于配置 llama.cpp 的链接方式。
//! 
//! 支持三种方式（类似 Ollama）：
//! 1. 自动下载编译：从 GitHub 下载 llama.cpp 源码并编译（类似 Ollama）
//! 2. Git Submodule：使用 git submodule（如果存在）
//! 3. 动态链接：链接系统已安装的 llama.cpp 库

use std::env;
use std::path::{Path, PathBuf};
use std::fs;
use std::process::Command;

/// 查找 Visual Studio 安装路径
#[cfg(target_os = "windows")]
fn find_visual_studio() -> Option<PathBuf> {
    // 尝试通过 vswhere 查找
    let vswhere_paths = vec![
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"),
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe"),
    ];
    
    for vswhere in vswhere_paths {
        if vswhere.exists() {
            let output = Command::new(&vswhere)
                .args(&["-latest", "-property", "installationPath"])
                .output()
                .ok()?;
            
            if output.status.success() {
                let path = String::from_utf8(output.stdout).ok()?;
                let path = path.trim();
                if !path.is_empty() {
                    return Some(PathBuf::from(path));
                }
            }
        }
    }
    
    // 尝试常见路径
    let common_paths = vec![
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\Community"),
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\Professional"),
        PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise"),
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"),
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"),
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise"),
    ];
    
    for path in common_paths {
        if path.exists() {
            return Some(path);
        }
    }
    
    None
}

#[cfg(not(target_os = "windows"))]
fn find_visual_studio() -> Option<PathBuf> {
    None
}

fn main() {
    // 告诉 Cargo 如果这些文件改变，需要重新构建
    println!("cargo:rerun-if-changed=build.rs");
    
    // 检查是否启用了 llama-cpp feature
    let llama_cpp_enabled = env::var("CARGO_FEATURE_LLAMA_CPP").is_ok();
    let llama_cpp_dynamic = env::var("CARGO_FEATURE_LLAMA_CPP_DYNAMIC").is_ok();
    
    if llama_cpp_enabled || llama_cpp_dynamic {
        configure_llama_cpp();
    } else {
        // 如果没有启用 llama-cpp feature，不进行任何链接配置
        // 代码会使用占位符实现
        println!("cargo:warning=llama-cpp feature not enabled. Using placeholder implementation.");
    }
}

fn configure_llama_cpp() {
    let llama_cpp_dynamic = env::var("CARGO_FEATURE_LLAMA_CPP_DYNAMIC").is_ok();
    
    if llama_cpp_dynamic {
        // 动态链接模式：链接系统库
        configure_dynamic_linking();
    } else {
        // 静态链接模式：尝试自动下载和编译（类似 Ollama）
        if configure_auto_build() {
            // 成功自动构建
            return;
        }
        
        // 如果自动构建失败，检查是否允许使用占位符实现
        // 如果启用了 llama-cpp feature，则必须找到库，否则报错
        // 如果没有启用 feature，可以使用占位符实现
        
        eprintln!("cargo:warning=Failed to auto-build llama.cpp");
        eprintln!("cargo:warning=");
        eprintln!("cargo:warning=Please choose one of the following options:");
        eprintln!("cargo:warning=1. Use git submodule:");
        eprintln!("cargo:warning=   git submodule add https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp");
        eprintln!("cargo:warning=   git submodule update --init --recursive");
        eprintln!("cargo:warning=");
        eprintln!("cargo:warning=2. Install llama.cpp system library and use dynamic linking:");
        eprintln!("cargo:warning=   cargo build --features llama-cpp-dynamic");
        eprintln!("cargo:warning=");
        eprintln!("cargo:warning=3. Ensure CMake and Git are installed for auto-build");
        eprintln!("cargo:warning=");
        eprintln!("cargo:warning=4. Build without llama-cpp feature (uses placeholder):");
        eprintln!("cargo:warning=   cargo build (without --features llama-cpp)");
        eprintln!("cargo:warning=");
        eprintln!("cargo:error=llama.cpp library not found. Cannot continue build with llama-cpp feature.");
        eprintln!("cargo:error=If you want to use placeholder implementation, remove --features llama-cpp");
        std::process::exit(1);
    }
}

/// 自动下载和编译 llama.cpp（类似 Ollama 的方式）
fn configure_auto_build() -> bool {
    let out_dir = env::var("OUT_DIR").unwrap();
    let vendor_dir = PathBuf::from(&out_dir).join("vendor");
    let llama_cpp_dir = vendor_dir.join("llama.cpp");
    
    // 检查是否已经存在
    if llama_cpp_dir.exists() && llama_cpp_dir.join("CMakeLists.txt").exists() {
        println!("cargo:warning=Using existing llama.cpp at {:?}", llama_cpp_dir);
        return build_llama_cpp(&llama_cpp_dir);
    }
    
    // 尝试使用 git submodule（如果存在）
    let submodule_path = PathBuf::from("vendor/llama.cpp");
    if submodule_path.exists() && submodule_path.join("CMakeLists.txt").exists() {
        println!("cargo:warning=Using git submodule llama.cpp at {:?}", submodule_path);
        return build_llama_cpp(&submodule_path);
    }
    
    // 尝试从 GitHub 下载（需要网络）
    println!("cargo:warning=Attempting to download llama.cpp from GitHub...");
    if download_llama_cpp(&llama_cpp_dir) {
        return build_llama_cpp(&llama_cpp_dir);
    }
    
    false
}

/// 从 GitHub 下载 llama.cpp 源码
fn download_llama_cpp(target_dir: &Path) -> bool {
    // 创建目录
    if let Err(e) = fs::create_dir_all(target_dir) {
        eprintln!("Failed to create directory: {}", e);
        return false;
    }
    
    // 使用 git clone（如果 git 可用）
    if Command::new("git").arg("--version").output().is_ok() {
        println!("cargo:warning=Cloning llama.cpp from GitHub...");
        
        // 设置 Git 安全目录（解决 Windows 上的所有权问题）
        let _ = Command::new("git")
            .args(&["config", "--global", "--add", "safe.directory", "*"])
            .output();
        
        let status = Command::new("git")
            .args(&[
                "clone",
                "--depth", "1",
                "--branch", "master",
                "https://github.com/ggerganov/llama.cpp.git",
                target_dir.to_str().unwrap(),
            ])
            .status();
        
        if status.is_ok() && status.unwrap().success() {
            println!("cargo:warning=Successfully cloned llama.cpp");
            return true;
        }
    }
    
    // 如果 git 不可用，提示用户手动安装
    eprintln!("cargo:warning=Git not available. Please install llama.cpp manually:");
    eprintln!("cargo:warning=  git submodule add https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp");
    eprintln!("cargo:warning=Or install llama.cpp system library and use --features llama-cpp-dynamic");
    
    false
}

/// 编译 llama.cpp
fn build_llama_cpp(llama_cpp_dir: &Path) -> bool {
    println!("cargo:warning=Building llama.cpp from source...");
    
    // 检查 CMake 是否可用
    if Command::new("cmake").arg("--version").output().is_err() {
        eprintln!("cargo:warning=CMake not found. Please install CMake to build llama.cpp");
        eprintln!("cargo:warning=Or use --features llama-cpp-dynamic for dynamic linking");
        return false;
    }
    
    // 解决 Git 所有权问题（Windows 上常见）
    // 最简单的方法：删除 .git 目录（我们不需要 Git 历史来构建）
    let git_dir = llama_cpp_dir.join(".git");
    if git_dir.exists() {
        println!("cargo:warning=Removing .git directory to avoid ownership issues...");
        let _ = fs::remove_dir_all(&git_dir);
    }
    
    // Windows 上先检查 MSVC（必须在配置 CMake 之前）
    #[cfg(target_os = "windows")]
    {
        // 检查是否有 MSVC
        let msvc_found = Command::new("cl")
            .arg("/?")
            .output()
            .is_ok();
        
        if !msvc_found {
            // 尝试通过 vswhere 查找 Visual Studio
            let vs_path = find_visual_studio();
            if vs_path.is_none() {
                eprintln!("cargo:error=MSVC compiler not found. llama.cpp requires MSVC on Windows.");
                eprintln!("cargo:error=");
                eprintln!("cargo:error=Please install one of the following:");
                eprintln!("cargo:error=  1. Visual Studio 2022 (with C++ build tools)");
                eprintln!("cargo:error=  2. Visual Studio Build Tools 2022");
                eprintln!("cargo:error=");
                eprintln!("cargo:error=Or use one of these alternatives:");
                eprintln!("cargo:error=  - Use git submodule: git submodule add https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp");
                eprintln!("cargo:error=  - Use dynamic linking: cargo build --features llama-cpp-dynamic");
                eprintln!("cargo:error=  - Build without llama-cpp: cargo build (uses placeholder)");
                return false;
            } else {
                eprintln!("cargo:warning=Visual Studio found but cl.exe not in PATH.");
                eprintln!("cargo:warning=Please run from 'Developer Command Prompt for VS' or set up the environment.");
                return false;
            }
        }
    }
    
    let build_dir = llama_cpp_dir.join("build");
    
    // 如果构建目录已存在且包含 CMakeCache，清理它（避免生成器冲突）
    if build_dir.exists() && build_dir.join("CMakeCache.txt").exists() {
        println!("cargo:warning=Cleaning previous CMake build cache...");
        let _ = fs::remove_dir_all(&build_dir);
    }
    
    // 创建构建目录
    if let Err(e) = fs::create_dir_all(&build_dir) {
        eprintln!("Failed to create build directory: {}", e);
        return false;
    }
    
    // 配置 CMake
    // 禁用不需要的功能以简化构建
    let mut cmake_cmd = Command::new("cmake");
    cmake_cmd
        .current_dir(&build_dir)
        .args(&[
            "..",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=OFF",  // 静态库
            "-DLLAMA_CURL=OFF",  // 禁用 CURL（不需要）
            "-DLLAMA_CUBLAS=OFF",  // 禁用 CUDA（可选）
            "-DLLAMA_METAL=OFF",   // 禁用 Metal（可选）
            "-DLLAMA_OPENBLAS=OFF",  // 禁用 OpenBLAS（可选）
            "-DLLAMA_BLAS=OFF",  // 禁用 BLAS（可选）
        ]);
    
    // Windows 上使用 Visual Studio 生成器
    #[cfg(target_os = "windows")]
    {
        // 尝试不同版本的 Visual Studio 生成器
        let vs_generators = vec![
            ("Visual Studio 17 2022", "x64"),
            ("Visual Studio 16 2019", "x64"),
            ("Visual Studio 15 2017", "x64"),
        ];
        
        let mut generator_found = false;
        for (gen, arch) in vs_generators {
            // 简单测试：尝试运行 cmake -G 命令
            let test_output = Command::new("cmake")
                .args(&["-G", gen, "--help"])
                .output();
            if test_output.is_ok() {
                cmake_cmd.args(&["-G", gen, "-A", arch]);
                generator_found = true;
                println!("cargo:warning=Using CMake generator: {} {}", gen, arch);
                break;
            }
        }
        
        if !generator_found {
            eprintln!("cargo:warning=No suitable Visual Studio generator found, CMake will use default");
        }
    }
    
    // 设置环境变量来抑制 Git 警告
    cmake_cmd.env("GIT_CONFIG_GLOBAL", "/dev/null");
    cmake_cmd.env("GIT_CONFIG_SYSTEM", "/dev/null");
    
    let cmake_status = cmake_cmd.status();
    
    if cmake_status.is_err() || !cmake_status.unwrap().success() {
        eprintln!("cargo:warning=CMake configuration failed");
        eprintln!("cargo:warning=Check the CMake output above for details");
        return false;
    }
    
    // 编译
    println!("cargo:warning=Compiling llama.cpp (this may take several minutes)...");
    let mut build_cmd = Command::new("cmake");
    build_cmd
        .current_dir(&build_dir)
        .args(&["--build", ".", "--config", "Release"]);
    
    // 设置环境变量来抑制 Git 警告
    build_cmd.env("GIT_CONFIG_GLOBAL", "/dev/null");
    build_cmd.env("GIT_CONFIG_SYSTEM", "/dev/null");
    
    let build_output = build_cmd.output();
    
    if build_output.is_err() {
        eprintln!("cargo:warning=Build command failed: {:?}", build_output.err());
        return false;
    }
    
    let build_output = build_output.unwrap();
    if !build_output.status.success() {
        eprintln!("cargo:warning=Build failed with exit code: {:?}", build_output.status.code());
        
        // 输出 stderr（构建错误）
        let stderr = String::from_utf8_lossy(&build_output.stderr);
        if !stderr.is_empty() {
            eprintln!("cargo:warning=Build errors:");
            // 只输出包含 "error" 或 "Error" 或 "failed" 的行，避免输出太多
            let error_lines: Vec<&str> = stderr
                .lines()
                .filter(|line| {
                    line.to_lowercase().contains("error") ||
                    line.to_lowercase().contains("failed") ||
                    line.contains("fatal")
                })
                .take(30)
                .collect();
            
            if !error_lines.is_empty() {
                for line in error_lines {
                    eprintln!("cargo:warning=  {}", line);
                }
            } else {
                // 如果没有明显的错误行，输出最后 20 行
                let lines: Vec<&str> = stderr.lines().rev().take(20).collect();
                for line in lines.iter().rev() {
                    eprintln!("cargo:warning=  {}", line);
                }
            }
        }
        
        // 也检查 stdout（可能有重要信息）
        let stdout = String::from_utf8_lossy(&build_output.stdout);
        if stdout.contains("error") || stdout.contains("Error") || stdout.contains("failed") {
            let error_lines: Vec<&str> = stdout
                .lines()
                .filter(|line| {
                    line.to_lowercase().contains("error") ||
                    line.to_lowercase().contains("failed")
                })
                .take(20)
                .collect();
            if !error_lines.is_empty() {
                eprintln!("cargo:warning=Build output errors:");
                for line in error_lines {
                    eprintln!("cargo:warning=  {}", line);
                }
            }
        }
        
        return false;
    }
    
    // 设置链接路径
    // Windows 上 CMake 构建的库通常在 build/Release/ 或 build/lib/Release/
    // Linux/macOS 上通常在 build/lib/
    
    #[cfg(target_os = "windows")]
    {
        // Windows 上的可能路径
        let lib_paths = vec![
            build_dir.join("Release"),  // build/Release/llama.lib
            build_dir.join("lib").join("Release"),  // build/lib/Release/llama.lib
            build_dir.join("lib"),  // build/lib/llama.lib
        ];
        
        for lib_path in lib_paths {
            if lib_path.exists() {
                // 检查是否有 .lib 文件
                let lib_file = lib_path.join("llama.lib");
                if lib_file.exists() {
                    println!("cargo:rustc-link-search=native={}", lib_path.display());
                    println!("cargo:rustc-link-lib=static=llama");
                    println!("cargo:warning=Successfully built and linked llama.cpp from {:?}", lib_path);
                    return true;
                }
            }
        }
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        // Linux/macOS 上的可能路径
        let lib_paths = vec![
            build_dir.join("lib"),
            build_dir.join("Release").join("lib"),
            build_dir.join("lib").join("Release"),
        ];
        
        for lib_path in lib_paths {
            if lib_path.exists() {
                // 检查是否有 .a 文件（静态库）
                let lib_file = lib_path.join("libllama.a");
                if lib_file.exists() {
                    println!("cargo:rustc-link-search=native={}", lib_path.display());
                    println!("cargo:rustc-link-lib=static=llama");
                    println!("cargo:warning=Successfully built and linked llama.cpp from {:?}", lib_path);
                    return true;
                }
            }
        }
    }
    
    eprintln!("cargo:warning=Library not found after build. Searched in:");
    eprintln!("cargo:warning=  {:?}", build_dir);
    false
}

fn configure_dynamic_linking() {
    // 动态链接系统已安装的 llama.cpp 库
    // 注意：这要求系统已安装 llama.cpp 库
    
    // Windows
    #[cfg(target_os = "windows")]
    {
        // 尝试查找常见的安装路径
        let possible_paths = vec![
            PathBuf::from("C:/Program Files/llama.cpp/lib"),
            PathBuf::from("C:/llama.cpp/lib"),
            PathBuf::from(env::var("LLAMA_CPP_LIB").unwrap_or_default()),
        ];
        
        let mut found = false;
        for path in possible_paths {
            if path.exists() && path.join("llama.lib").exists() {
                println!("cargo:rustc-link-search=native={}", path.display());
                found = true;
                break;
            }
        }
        
        if !found {
            // 检查环境变量
            if let Ok(lib_path) = env::var("LLAMA_CPP_LIB") {
                println!("cargo:rustc-link-search=native={}", lib_path);
            }
        }
        
        println!("cargo:rustc-link-lib=dylib=llama");
        println!("cargo:warning=Using dynamic linking. Ensure llama.dll is in PATH or DLL search path.");
    }
    
    // macOS
    #[cfg(target_os = "macos")]
    {
        // 尝试 Homebrew 路径
        let homebrew_paths = vec![
            PathBuf::from("/opt/homebrew/lib"),
            PathBuf::from("/usr/local/lib"),
        ];
        
        for path in homebrew_paths {
            if path.exists() && path.join("libllama.dylib").exists() {
                println!("cargo:rustc-link-search=native={}", path.display());
                break;
            }
        }
        
        println!("cargo:rustc-link-lib=dylib=llama");
    }
    
    // Linux
    #[cfg(target_os = "linux")]
    {
        // 尝试常见路径
        let possible_paths = vec![
            PathBuf::from("/usr/local/lib"),
            PathBuf::from("/usr/lib"),
        ];
        
        for path in possible_paths {
            if path.exists() && (path.join("libllama.so").exists() || path.join("libllama.a").exists()) {
                println!("cargo:rustc-link-search=native={}", path.display());
                break;
            }
        }
        
        println!("cargo:rustc-link-lib=dylib=llama");
    }
}

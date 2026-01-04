# 故障排除指南

## Windows 链接错误：`cannot open input file 'llama.lib'`

### 问题描述

在 Windows 上编译时遇到：
```
LINK : fatal error LNK1181: cannot open input file 'llama.lib'
```

### 原因

`build.rs` 尝试链接 llama.cpp 库，但找不到库文件。可能的原因：
1. llama.cpp 还没有被编译
2. 库文件路径不正确
3. CMake 构建失败

### 解决方案

#### 方案1: 使用 Git Submodule（推荐）

```powershell
# 添加 submodule
git submodule add https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp
git submodule update --init --recursive

# 编译
cargo build --features llama-cpp
```

#### 方案2: 手动编译 llama.cpp

```powershell
# 克隆 llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp
cd vendor/llama.cpp

# 创建构建目录
mkdir build
cd build

# 配置 CMake（使用 MSVC）
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF

# 编译
cmake --build . --config Release

# 返回项目根目录
cd ../../..

# 编译项目
cargo build --features llama-cpp
```

#### 方案3: 使用动态链接

```powershell
# 安装 llama.cpp 系统库后
cargo build --features llama-cpp-dynamic
```

#### 方案4: 不使用 llama-cpp feature（占位符实现）

```powershell
# 不启用 llama-cpp feature，使用占位符实现
cargo build
```

## Windows 编译错误：编译器问题

### 问题描述

编译 llama.cpp 时遇到：
```
error: unknown type name 'THREAD_POWER_THROTTLING_STATE'
error: '::CreateFile2' has not been declared
```

### 原因

在 Windows 上，llama.cpp **必须使用 MSVC 编译器**，不能使用 GCC/MinGW。

### 解决方案

#### 方案1: 安装 Visual Studio Build Tools

1. 下载并安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
2. 选择 "C++ build tools" 工作负载
3. 重新编译项目

#### 方案2: 安装完整 Visual Studio

1. 下载并安装 [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)
2. 选择 "Desktop development with C++" 工作负载
3. 重新编译项目

#### 方案3: 使用预编译库

如果不想编译，可以使用预编译的 llama.cpp 库：

```powershell
# 下载预编译库并设置路径
$env:LLAMA_CPP_LIB = "C:\path\to\llama.cpp\lib"
cargo build --features llama-cpp-dynamic
```

#### 方案4: 使用 WSL2

在 WSL2 中使用 GCC 编译：

```bash
# 在 WSL2 中
git clone https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp
cd vendor/llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build . --config Release
```

## 其他常见问题

### CMake 未找到

**错误：**
```
CMake not found. Please install CMake to build llama.cpp
```

**解决：**
1. 下载并安装 CMake：https://cmake.org/download/
2. 确保 CMake 在 PATH 中
3. 或使用 `--features llama-cpp-dynamic` 跳过编译

### Git 未找到

**错误：**
```
Git not available
```

**解决：**
1. 安装 Git：https://git-scm.com/download/win
2. 或手动下载 llama.cpp 源码到 `vendor/llama.cpp`

### Git 所有权问题（Windows）

**错误：**
```
fatal: detected dubious ownership in repository
```

**解决：**
```powershell
# 设置 Git 安全目录
git config --global --add safe.directory "*"
```

或者，`build.rs` 会自动删除 `.git` 目录来避免这个问题。

### 库路径问题

如果库文件在非标准位置，可以设置环境变量：

```powershell
# 设置库路径
$env:LLAMA_CPP_LIB = "C:\path\to\llama.cpp\lib"
cargo build --features llama-cpp-dynamic
```

## 调试技巧

### 查看详细构建信息

```powershell
cargo build --features llama-cpp --verbose
```

### 检查库文件是否存在

```powershell
# 检查构建输出目录
Get-ChildItem -Recurse -Filter "llama.lib" target/
```

### 检查 CMake 配置

```powershell
# 在 llama.cpp 构建目录中
cd vendor/llama.cpp/build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF --verbose
```

### 检查编译器

```powershell
# 检查 MSVC
cl

# 检查 GCC（不应该使用）
gcc --version
```

## 获取帮助

如果问题仍然存在：
1. 检查 `build.rs` 的输出信息
2. 查看 `target/debug/build/ferrousforge-*/output` 中的构建日志
3. 确保所有依赖（CMake、Git、Visual Studio Build Tools）都已正确安装
4. 在 Windows 上，**必须使用 MSVC 编译器**，不能使用 GCC/MinGW

# Ollama 自动安装 llama.cpp 机制分析

## Ollama 的实现方式

Ollama **不是**运行时下载 llama.cpp，而是采用以下方式：

### 1. **编译时集成（主要方式）**

Ollama 将 llama.cpp 作为 **git submodule** 包含在项目中：

```bash
# Ollama 的仓库结构
ollama/
  ├── llm/
  │   └── llama.cpp/  # git submodule
  ├── server/
  └── ...
```

在编译时：
- 自动拉取 llama.cpp submodule
- 使用 CMake 或 Make 编译 llama.cpp
- 静态链接到最终二进制文件

### 2. **预编译二进制**

Ollama 发布的安装包已经包含了编译好的 llama.cpp，用户下载的二进制文件已经静态链接了 llama.cpp。

### 3. **构建系统集成**

Ollama 使用 Go 的构建系统，在 `go build` 时：
- 自动处理 submodule
- 调用 CGO 编译 C++ 代码
- 生成单一可执行文件

## FerrousForge 的实现方案

我们实现了类似的自动安装机制：

### 方案1: Git Submodule（推荐，类似 Ollama）

```bash
# 添加 llama.cpp 作为 submodule
git submodule add https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp

# 编译时会自动使用
cargo build --features llama-cpp
```

### 方案2: 自动下载和编译（build.rs）

`build.rs` 会自动：
1. 检查是否存在 `vendor/llama.cpp` submodule
2. 如果不存在，尝试从 GitHub 克隆
3. 使用 CMake 编译
4. 静态链接

### 方案3: 动态链接（备选）

如果自动构建失败，回退到动态链接系统库。

## 使用方式

### 方式1: 使用 Git Submodule（最像 Ollama）

```bash
# 初始化 submodule
git submodule add https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp
git submodule update --init --recursive

# 编译
cargo build --features llama-cpp
```

### 方式2: 自动下载（需要网络和 CMake）

```bash
# build.rs 会自动下载和编译
cargo build --features llama-cpp
```

### 方式3: 系统库（最简单）

```bash
# 安装系统库
# macOS: brew install llama.cpp
# Linux: 从源码编译并安装

# 使用动态链接
cargo build --features llama-cpp-dynamic
```

## 对比 Ollama

| 特性 | Ollama | FerrousForge |
|------|--------|--------------|
| 集成方式 | Git Submodule | Git Submodule + 自动下载 |
| 编译时机 | 构建时 | 构建时 |
| 链接方式 | 静态链接 | 静态/动态可选 |
| 用户操作 | 无需操作 | 可选：submodule 或自动下载 |

## 优势

1. **零配置**：用户无需手动安装 llama.cpp
2. **跨平台**：自动适配不同平台
3. **版本控制**：通过 submodule 锁定版本
4. **灵活性**：支持多种集成方式

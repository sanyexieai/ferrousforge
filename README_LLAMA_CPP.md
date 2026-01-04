# llama.cpp 自动安装说明

## Ollama 是如何"自动安装" llama.cpp 的？

Ollama **并不是**运行时下载 llama.cpp，而是：

### 1. **编译时集成（Git Submodule）**

Ollama 将 llama.cpp 作为 git submodule 包含在项目中：

```
ollama/
  └── llm/
      └── llama.cpp/  # git submodule
```

当用户编译 Ollama 时：
- Git 自动拉取 llama.cpp submodule
- 构建系统编译 llama.cpp
- 静态链接到最终二进制文件

### 2. **预编译二进制**

Ollama 发布的安装包已经包含了编译好的 llama.cpp，所以用户下载后：
- 无需安装 llama.cpp
- 无需配置
- 直接运行即可

## FerrousForge 的实现

我们实现了类似的机制，支持三种方式：

### 方式1: Git Submodule（最像 Ollama，推荐）

```bash
# 添加 submodule
git submodule add https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp
git submodule update --init --recursive

# 编译（会自动使用 submodule）
cargo build --features llama-cpp
```

**优点：**
- 版本锁定（通过 submodule commit）
- 无需网络（编译时）
- 与 Ollama 方式一致

### 方式2: 自动下载和编译（build.rs）

`build.rs` 会自动：
1. 检查是否存在 `vendor/llama.cpp`
2. 如果不存在，尝试从 GitHub 克隆
3. 使用 CMake 编译
4. 静态链接

```bash
# 只需运行（需要网络和 CMake）
cargo build --features llama-cpp
```

**优点：**
- 完全自动化
- 用户无需任何操作

**要求：**
- 需要安装 Git 和 CMake
- 需要网络连接

### 方式3: 系统库动态链接

```bash
# 安装系统库
# macOS: brew install llama.cpp
# Linux: 从源码编译

# 使用动态链接
cargo build --features llama-cpp-dynamic
```

**优点：**
- 最简单
- 不增加二进制大小

## 推荐工作流

### 开发环境

使用 Git Submodule：

```bash
git submodule add https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp
```

### 生产环境

1. **选项A**：使用 submodule（推荐）
   - 版本可控
   - 编译时自动包含

2. **选项B**：预编译二进制
   - 发布时已经包含 llama.cpp
   - 用户无需安装

3. **选项C**：动态链接
   - 用户自行安装系统库
   - 运行时加载

## 与 Ollama 的对比

| 特性 | Ollama | FerrousForge |
|------|--------|--------------|
| 集成方式 | Git Submodule | Git Submodule + 自动下载 |
| 编译时机 | 构建时 | 构建时 |
| 链接方式 | 静态链接 | 静态/动态可选 |
| 用户操作 | 无需操作（预编译） | 可选：submodule 或自动下载 |
| 版本控制 | Submodule commit | Submodule commit |

## 快速开始

### 最简单的方式（类似 Ollama）

```bash
# 1. 添加 submodule
git submodule add https://github.com/ggerganov/llama.cpp.git vendor/llama.cpp

# 2. 编译
cargo build --features llama-cpp

# 完成！llama.cpp 已自动集成
```

### 完全自动（需要网络）

```bash
# 只需编译，build.rs 会自动处理
cargo build --features llama-cpp
```

## 注意事项

1. **首次编译**：如果使用自动下载，需要网络连接和 CMake
2. **编译时间**：编译 llama.cpp 可能需要几分钟
3. **磁盘空间**：llama.cpp 源码约 100MB+
4. **版本锁定**：使用 submodule 可以锁定特定版本


# llama.cpp 快速设置指南

## 快速开始

### 1. 安装 llama.cpp

**Windows:**
```powershell
# 下载预编译的库或从源码编译
# 确保库文件在系统路径中
```

**macOS:**
```bash
brew install llama.cpp
# 或
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make
```

**Linux:**
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make
sudo make install
```

### 2. 编译项目

**静态链接（推荐）:**
```bash
cargo build --features llama-cpp
```

**动态加载:**
```bash
cargo build --features llama-cpp-dynamic
```

### 3. 运行示例

```bash
# 使用真实的 GGUF 模型
cargo run --example llama_cpp_example --features llama-cpp -- \
    --model path/to/model.gguf \
    --prompt "Hello, world!"
```

## 配置 build.rs

`build.rs` 会自动配置链接方式。如果需要自定义路径，可以修改 `build.rs` 中的路径设置。

## 测试

```bash
# 单元测试
cargo test --lib inference::backends::llama_cpp

# 集成测试（需要真实模型）
export TEST_MODEL_PATH=/path/to/model.gguf
cargo test --features llama-cpp integration_llama_cpp
```

## 故障排除

### 链接错误

如果遇到链接错误，检查：
1. llama.cpp 库是否已安装
2. 库路径是否正确
3. 架构是否匹配

### 运行时错误

如果运行时出错：
1. 检查模型文件格式（必须是 GGUF）
2. 确认库文件在系统路径中
3. 检查依赖库（CUDA、Metal 等）

## 下一步

查看 [LLAMA_CPP_INTEGRATION.md](./LLAMA_CPP_INTEGRATION.md) 获取详细文档。


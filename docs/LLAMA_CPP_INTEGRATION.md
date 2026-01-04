# llama.cpp 集成指南

本文档说明如何在 FerrousForge 中集成和使用 llama.cpp 后端。

## 前置要求

### 1. 安装 llama.cpp 库

#### Windows
```powershell
# 选项1: 使用 vcpkg
vcpkg install llama

# 选项2: 从源码编译
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

#### macOS
```bash
# 使用 Homebrew
brew install llama.cpp

# 或从源码编译
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
```

#### Linux
```bash
# 从源码编译
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
```

### 2. 配置库路径

确保 llama.cpp 库在系统路径中，或设置环境变量：

**Windows:**
```powershell
$env:LIB = "C:\path\to\llama.cpp\lib;$env:LIB"
$env:INCLUDE = "C:\path\to\llama.cpp\include;$env:INCLUDE"
```

**macOS/Linux:**
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
```

## 编译选项

### 方式1: 静态链接（推荐）

静态链接 llama.cpp 库：

```bash
cargo build --features llama-cpp
```

这需要：
- llama.cpp 库已安装在系统中
- `build.rs` 已配置正确的链接路径

### 方式2: 动态加载

使用 `libloading` 动态加载库：

```bash
cargo build --features llama-cpp-dynamic
```

这需要：
- llama.cpp 动态库（.dll/.dylib/.so）在系统路径中
- 运行时自动查找并加载库

## 使用示例

### 基本使用

```rust
use ferrousforge::inference::backends::llama_cpp::LlamaCppBackend;
use ferrousforge::inference::backend::{InferenceBackend, BackendConfig};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建后端
    let backend = LlamaCppBackend::new();
    
    // 配置
    let config = BackendConfig {
        num_threads: Some(4),
        context_size: Some(2048),
        batch_size: Some(512),
        ..Default::default()
    };
    
    // 加载模型
    let model_path = Path::new("path/to/model.gguf");
    let handle = backend.load_model(model_path, config, None).await?;
    
    // 执行推理
    let input = ferrousforge::inference::backend::BackendInput::Text(
        "Hello, how are you?".to_string()
    );
    let options = ferrousforge::api::request::InferenceOptions {
        temperature: Some(0.7),
        max_tokens: Some(100),
        ..Default::default()
    };
    
    let output = backend.infer(&handle, input, options).await?;
    
    match output {
        ferrousforge::inference::backend::BackendOutput::Text(text) => {
            println!("Generated: {}", text);
        }
        _ => {}
    }
    
    // 卸载模型
    backend.unload_model(handle).await?;
    
    Ok(())
}
```

## FFI 函数说明

### 已实现的函数

- `llama_model_load_from_file`: 加载模型
- `llama_new_context_with_model`: 创建上下文
- `llama_tokenize`: Tokenize 文本
- `llama_eval`: 执行推理
- `llama_sample_top_p_top_k`: 采样 token
- `llama_token_to_str`: Token 转字符串
- `llama_token_eos`: 获取 EOS token
- `llama_model_free`: 释放模型
- `llama_context_free`: 释放上下文

### 函数签名

所有函数都通过 FFI 调用，支持：
- 静态链接：直接调用 C 函数
- 动态加载：通过 `libloading` 动态调用

## 测试

### 使用真实模型测试

1. 下载一个 GGUF 格式的模型文件
2. 运行测试：

```bash
# 设置模型路径环境变量
export TEST_MODEL_PATH=/path/to/model.gguf

# 运行测试
cargo test --features llama-cpp inference::backends::llama_cpp -- --nocapture
```

### 单元测试

```bash
cargo test --lib inference::backends::llama_cpp
```

## 故障排除

### 问题1: 找不到 llama.cpp 库

**解决方案:**
- 确保库已正确安装
- 检查 `build.rs` 中的链接路径
- 使用动态加载模式：`--features llama-cpp-dynamic`

### 问题2: 链接错误

**解决方案:**
- 检查库文件是否存在
- 确认架构匹配（x86_64 vs ARM）
- 检查依赖库（如 CUDA、Metal 等）

### 问题3: 运行时错误

**解决方案:**
- 检查模型文件格式（必须是 GGUF）
- 确认模型文件完整
- 检查内存是否足够

## 性能优化

1. **使用 GPU 加速:**
   ```rust
   let accelerator = Some(AcceleratorType::Cuda { device_id: 0 });
   let handle = backend.load_model(path, config, accelerator).await?;
   ```

2. **调整上下文大小:**
   ```rust
   let config = BackendConfig {
       context_size: Some(4096),  // 更大的上下文
       ..Default::default()
   };
   ```

3. **批处理:**
   ```rust
   let inputs = vec![
       BackendInput::Text("Prompt 1".to_string()),
       BackendInput::Text("Prompt 2".to_string()),
   ];
   let outputs = backend.infer_batch(&handle, inputs, options).await?;
   ```

## 下一步

- [ ] 实现流式推理
- [ ] 添加更多采样策略
- [ ] 支持多 GPU
- [ ] 性能基准测试
- [ ] 内存优化


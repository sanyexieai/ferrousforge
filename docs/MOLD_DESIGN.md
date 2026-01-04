# Mold 统一模型接口设计

## 概述

"Mold" 是 FerrousForge 的核心抽象，提供统一的模型接口。无论底层使用什么后端（Candle、llama.cpp、ONNX 等），所有模型都通过 Mold 接口暴露，实现真正的统一抽象。

## 设计目标

1. **统一接口**：所有模型类型使用相同的 API
2. **类型安全**：编译时类型检查
3. **易于扩展**：添加新的 Mold 实现简单
4. **运行时选择**：根据模型格式自动选择最佳 Mold
5. **硬件抽象**：自动适配不同硬件加速器

## Mold Trait 定义

```rust
use std::pin::Pin;
use std::future::Future;
use async_trait::async_trait;
use serde::de::DeserializeOwned;

/// 统一模型接口
#[async_trait]
pub trait Mold: Send + Sync {
    /// 输入类型
    type Input: Send + Sync;
    /// 输出类型
    type Output: Send + Sync;
    /// 配置类型
    type Config: DeserializeOwned + Send + Sync;
    
    // ========== 模型标识 ==========
    
    /// 模型名称
    fn name(&self) -> &str;
    
    /// 模型类型（文本、图像、音频等）
    fn model_type(&self) -> ModelType;
    
    /// 后端类型（Candle、LlamaCpp 等）
    fn backend_type(&self) -> BackendType;
    
    /// 当前使用的硬件加速器
    fn hardware_accelerator(&self) -> AcceleratorType;
    
    // ========== 生命周期 ==========
    
    /// 加载模型
    async fn load(&mut self, config: Self::Config) -> Result<()>;
    
    /// 卸载模型
    async fn unload(&mut self) -> Result<()>;
    
    /// 检查模型是否已加载
    fn is_loaded(&self) -> bool;
    
    // ========== 元数据 ==========
    
    /// 获取模型元数据
    fn metadata(&self) -> &ModelMetadata;
    
    // ========== 推理接口 ==========
    
    /// 单次推理
    async fn infer(
        &self,
        input: Self::Input,
        options: InferenceOptions,
    ) -> Result<Self::Output>;
    
    /// 流式推理
    async fn infer_stream(
        &self,
        input: Self::Input,
        options: InferenceOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output>> + Send>>>;
    
    /// 批处理推理
    async fn infer_batch(
        &self,
        inputs: Vec<Self::Input>,
        options: InferenceOptions,
    ) -> Result<Vec<Self::Output>>;
    
    // ========== 资源管理 ==========
    
    /// 获取当前内存使用
    fn memory_usage(&self) -> Result<MemoryUsage>;
    
    /// 预热模型（可选）
    async fn warmup(&self) -> Result<()>;
}
```

## Mold 实现类型

### 1. CandleMold

纯 Rust 实现，使用 Candle 框架。

```rust
pub struct CandleMold {
    name: String,
    model: CandleModel,
    backend: Arc<CandleBackend>,
    accelerator: AcceleratorType,
    metadata: ModelMetadata,
    loaded: bool,
}

#[async_trait]
impl Mold for CandleMold {
    type Input = TextInput;  // 或其他输入类型
    type Output = TextOutput;
    type Config = CandleConfig;
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn backend_type(&self) -> BackendType {
        BackendType::Candle
    }
    
    fn hardware_accelerator(&self) -> AcceleratorType {
        self.accelerator
    }
    
    async fn load(&mut self, config: Self::Config) -> Result<()> {
        // 加载 Candle 模型
        self.model = CandleModel::load(&config.path, self.accelerator)?;
        self.loaded = true;
        Ok(())
    }
    
    async fn infer(
        &self,
        input: Self::Input,
        options: InferenceOptions,
    ) -> Result<Self::Output> {
        // 使用 Candle 进行推理
        let output = self.model.forward(input, options)?;
        Ok(output)
    }
    
    // ... 其他方法实现
}
```

### 2. LlamaCppMold

通过 FFI 调用 llama.cpp（C++）。

```rust
pub struct LlamaCppMold {
    name: String,
    context: LlamaContext,
    backend: Arc<LlamaCppBackend>,
    accelerator: AcceleratorType,
    metadata: ModelMetadata,
    loaded: bool,
}

#[async_trait]
impl Mold for LlamaCppMold {
    type Input = TextInput;
    type Output = TextOutput;
    type Config = LlamaCppConfig;
    
    fn backend_type(&self) -> BackendType {
        BackendType::LlamaCpp
    }
    
    async fn load(&mut self, config: Self::Config) -> Result<()> {
        // 通过 FFI 加载 llama.cpp 模型
        let params = LlamaContextParams {
            n_gpu_layers: config.gpu_layers,
            // ... 其他参数
        };
        self.context = self.backend.load_model(&config.path, params)?;
        self.loaded = true;
        Ok(())
    }
    
    async fn infer(
        &self,
        input: Self::Input,
        options: InferenceOptions,
    ) -> Result<Self::Output> {
        // 通过 FFI 调用 llama.cpp 推理
        let tokens = self.backend.tokenize(&input.text)?;
        let output = self.backend.generate(&self.context, tokens, options)?;
        Ok(TextOutput { text: output })
    }
    
    // ... 其他方法实现
}
```

### 3. ExternalMold

桥接外部框架（PyTorch、TensorFlow 等）。

```rust
pub struct ExternalMold {
    name: String,
    bridge: Box<dyn ExternalBridge>,
    bridge_type: ExternalBridgeType,
    accelerator: AcceleratorType,
    metadata: ModelMetadata,
    loaded: bool,
}

#[async_trait]
impl Mold for ExternalMold {
    type Input = DynamicInput;
    type Output = DynamicOutput;
    type Config = ExternalConfig;
    
    fn backend_type(&self) -> BackendType {
        BackendType::External(self.bridge_type)
    }
    
    async fn load(&mut self, config: Self::Config) -> Result<()> {
        // 通过桥接加载外部模型
        self.bridge = match self.bridge_type {
            ExternalBridgeType::PyTorch => {
                Box::new(PyTorchBridge::new(&config.path)?)
            }
            ExternalBridgeType::TensorFlow => {
                Box::new(TensorFlowBridge::new(&config.path)?)
            }
            ExternalBridgeType::JAX => {
                Box::new(JAXBridge::new(&config.path)?)
            }
            _ => return Err(FerrousForgeError::UnsupportedBridge(self.bridge_type)),
        };
        self.loaded = true;
        Ok(())
    }
    
    async fn infer(
        &self,
        input: Self::Input,
        options: InferenceOptions,
    ) -> Result<Self::Output> {
        // 通过桥接进行推理
        let output = self.bridge.infer(input, options)?;
        Ok(output)
    }
    
    // ... 其他方法实现
}
```

## Mold 工厂

自动选择最佳 Mold 实现：

```rust
pub struct MoldFactory {
    backends: HashMap<BackendType, Arc<dyn InferenceBackend>>,
    available_accelerators: Vec<AcceleratorType>,
}

impl MoldFactory {
    /// 创建 Mold，自动选择最佳实现
    pub async fn create_mold(
        &self,
        model_path: &Path,
        model_metadata: &ModelMetadata,
    ) -> Result<Box<dyn Mold>> {
        // 1. 根据模型格式选择后端
        let backend = self.select_backend(model_metadata.format)?;
        
        // 2. 根据硬件和模型需求选择加速器
        let accelerator = self.select_accelerator(model_metadata)?;
        
        // 3. 创建对应的 Mold
        match backend.backend_type() {
            BackendType::Candle => {
                self.create_candle_mold(backend, accelerator, model_path, model_metadata).await
            }
            BackendType::LlamaCpp => {
                self.create_llama_cpp_mold(backend, accelerator, model_path, model_metadata).await
            }
            BackendType::OnnxRuntime => {
                self.create_onnx_mold(backend, accelerator, model_path, model_metadata).await
            }
            BackendType::Tvm => {
                self.create_tvm_mold(backend, accelerator, model_path, model_metadata).await
            }
            BackendType::External(bridge_type) => {
                self.create_external_mold(backend, accelerator, model_path, model_metadata, bridge_type).await
            }
            _ => Err(FerrousForgeError::UnsupportedBackend(backend.backend_type())),
        }
    }
    
    fn select_backend(&self, format: ModelFormat) -> Result<Arc<dyn InferenceBackend>> {
        match format {
            ModelFormat::Gguf => {
                self.backends.get(&BackendType::LlamaCpp)
                    .ok_or_else(|| FerrousForgeError::BackendNotAvailable(BackendType::LlamaCpp))
            }
            ModelFormat::Onnx => {
                self.backends.get(&BackendType::OnnxRuntime)
                    .ok_or_else(|| FerrousForgeError::BackendNotAvailable(BackendType::OnnxRuntime))
            }
            ModelFormat::Safetensors => {
                self.backends.get(&BackendType::Candle)
                    .ok_or_else(|| FerrousForgeError::BackendNotAvailable(BackendType::Candle))
            }
            ModelFormat::Tvm => {
                self.backends.get(&BackendType::Tvm)
                    .ok_or_else(|| FerrousForgeError::BackendNotAvailable(BackendType::Tvm))
            }
            _ => Err(FerrousForgeError::UnsupportedFormat(format)),
        }
        .map(|b| b.clone())
    }
    
    fn select_accelerator(&self, metadata: &ModelMetadata) -> Result<AcceleratorType> {
        let requirements = &metadata.requirements;
        
        // 如果模型需要 GPU
        if requirements.gpu_required {
            // 优先选择 CUDA
            if let Some(cuda) = self.available_accelerators.iter()
                .find(|a| matches!(a, AcceleratorType::Cuda { .. })) {
                return Ok(*cuda);
            }
            
            // 其次 Metal（Apple Silicon）
            if let Some(metal) = self.available_accelerators.iter()
                .find(|a| matches!(a, AcceleratorType::Metal { .. })) {
                return Ok(*metal);
            }
            
            // 再次 Vulkan
            if let Some(vulkan) = self.available_accelerators.iter()
                .find(|a| matches!(a, AcceleratorType::Vulkan { .. })) {
                return Ok(*vulkan);
            }
            
            return Err(FerrousForgeError::GpuRequired);
        }
        
        // 否则使用 CPU
        Ok(AcceleratorType::Cpu)
    }
}
```

## 使用示例

```rust
// 1. 创建 Mold 工厂
let factory = MoldFactory::new()?;

// 2. 加载模型元数据
let metadata = load_model_metadata("llama2-7b")?;

// 3. 创建 Mold（自动选择最佳实现）
let mut mold = factory.create_mold(&model_path, &metadata).await?;

// 4. 加载模型
mold.load(CandleConfig {
    path: model_path.clone(),
    // ... 其他配置
}).await?;

// 5. 进行推理
let input = TextInput {
    text: "Hello, world!".to_string(),
};
let output = mold.infer(input, InferenceOptions::default()).await?;
println!("Output: {}", output.text);

// 6. 流式推理
let mut stream = mold.infer_stream(input, InferenceOptions::default()).await?;
while let Some(chunk) = stream.next().await {
    print!("{}", chunk?.text);
}

// 7. 批处理
let inputs = vec![
    TextInput { text: "Input 1".to_string() },
    TextInput { text: "Input 2".to_string() },
];
let outputs = mold.infer_batch(inputs, InferenceOptions::default()).await?;
```

## 优势

1. **统一接口**：无论使用什么后端，API 完全一致
2. **类型安全**：编译时检查，避免运行时错误
3. **自动优化**：根据硬件和模型自动选择最佳实现
4. **易于扩展**：添加新的 Mold 实现只需实现 trait
5. **性能优化**：可以针对不同后端进行特定优化

## 扩展性

### 添加新的 Mold 实现

```rust
// 1. 实现 Mold trait
pub struct MyCustomMold { /* ... */ }

#[async_trait]
impl Mold for MyCustomMold {
    // ... 实现所有必需方法
}

// 2. 在工厂中注册
impl MoldFactory {
    fn create_my_custom_mold(...) -> Result<Box<dyn Mold>> {
        Ok(Box::new(MyCustomMold::new(...)?))
    }
}
```

### 添加新的硬件加速器

```rust
// 1. 实现 HardwareAccelerator trait
pub struct MyAccelerator { /* ... */ }

impl HardwareAccelerator for MyAccelerator {
    // ... 实现所有必需方法
}

// 2. 在工厂中注册
impl AcceleratorFactory {
    fn create_my_accelerator() -> Result<Box<dyn HardwareAccelerator>> {
        Ok(Box::new(MyAccelerator::new()?))
    }
}
```


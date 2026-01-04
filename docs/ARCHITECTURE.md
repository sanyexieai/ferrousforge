# FerrousForge 架构设计文档

## 1. 整体架构

### 1.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                FerrousForge 统一平台                         │
├─────────────────────────────────────────────────────────────┤
│   API网关 │ 模型管理 │ 资源调度 │ 插件系统 │ 监控告警           │
├─────────────────────────────────────────────────────────────┤
│                     模型抽象层 (统一接口)                      │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ 统一模型 Trait: Mold                                │  │
│   │  ├── CandleMold: 纯 Rust 实现                       │  │
│   │  ├── LlamaCppMold: C++ 高性能后端                   │  │
│   │  └── ExternalMold: 外部框架桥接                      │  │
│   └─────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     计算后端层                               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│   │ Candle   │ │ llama.cpp│ │ ONNX     │ │ TVM      │    │
│   │ (Rust)   │ │ (C++)    │ │ Runtime  │ │ (C++)    │    │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    硬件加速层                                │
│   CPU │ CUDA │ Metal │ Vulkan │ WebGPU │ WASM              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件交互

```
Client Request
    │
    ├─→ HTTP Server (axum)
    │       │
    │       ├─→ Middleware (auth, logging)
    │       │
    │       └─→ Handler
    │               │
    │               └─→ Core Engine
    │                       │
    │                       ├─→ Registry (查找模型)
    │                       │
    │                       ├─→ Scheduler (任务调度)
    │                       │
    │                       └─→ Model Instance
    │                               │
    │                               └─→ Inference Backend
    │                                       │
    │                                       └─→ Storage (模型文件)
    │
    └─→ Response
```

## 2. 核心设计模式

### 2.1 Trait 抽象层

#### 统一模型接口 "Mold"

"Mold" 是 FerrousForge 的核心抽象，提供统一的模型接口。所有模型类型都通过 Mold 实现，无论底层使用什么后端。

```rust
// 统一模型接口 "Mold"
pub trait Mold: Send + Sync {
    type Input: Send + Sync;
    type Output: Send + Sync;
    type Config: DeserializeOwned + Send + Sync;
    
    // 模型标识
    fn name(&self) -> &str;
    fn model_type(&self) -> ModelType;
    fn backend_type(&self) -> BackendType;
    fn hardware_accelerator(&self) -> HardwareAccelerator;
    
    // 生命周期
    async fn load(&mut self, config: Self::Config) -> Result<()>;
    async fn unload(&mut self) -> Result<()>;
    fn is_loaded(&self) -> bool;
    
    // 元数据
    fn metadata(&self) -> &ModelMetadata;
    
    // 推理
    async fn infer(
        &self,
        input: Self::Input,
        options: InferenceOptions,
    ) -> Result<Self::Output>;
    
    // 流式推理
    async fn infer_stream(
        &self,
        input: Self::Input,
        options: InferenceOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::Output>> + Send>>>;
    
    // 批处理推理
    async fn infer_batch(
        &self,
        inputs: Vec<Self::Input>,
        options: InferenceOptions,
    ) -> Result<Vec<Self::Output>>;
}

// Mold 实现类型
pub enum MoldType {
    CandleMold(Box<dyn CandleMold>),
    LlamaCppMold(Box<dyn LlamaCppMold>),
    ExternalMold(Box<dyn ExternalMold>),
}

// CandleMold: 纯 Rust 实现
pub trait CandleMold: Mold {
    fn candle_backend(&self) -> &CandleBackend;
    fn device(&self) -> Device;  // CPU, CUDA, Metal
}

// LlamaCppMold: C++ 高性能后端
pub trait LlamaCppMold: Mold {
    fn llama_context(&self) -> &LlamaContext;
    fn gpu_layers(&self) -> usize;
}

// ExternalMold: 外部框架桥接
pub trait ExternalMold: Mold {
    fn bridge_type(&self) -> ExternalBridgeType;
    fn bridge_config(&self) -> &BridgeConfig;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalBridgeType {
    PyTorch,
    TensorFlow,
    JAX,
    ONNX,
    Custom(String),
}
```

#### 硬件加速器抽象

```rust
// 硬件加速器 trait
pub trait HardwareAccelerator: Send + Sync {
    fn accelerator_type(&self) -> AcceleratorType;
    fn is_available(&self) -> bool;
    fn memory_info(&self) -> Result<MemoryInfo>;
    fn create_context(&self) -> Result<Box<dyn AcceleratorContext>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AcceleratorType {
    Cpu,
    Cuda { device_id: u32 },
    Metal { device_id: u32 },
    Vulkan { device_id: u32 },
    WebGpu,
    Wasm,
}

// 加速器上下文
pub trait AcceleratorContext: Send + Sync {
    fn allocate_buffer(&mut self, size: usize) -> Result<BufferHandle>;
    fn deallocate_buffer(&mut self, handle: BufferHandle) -> Result<()>;
    fn upload_data(&mut self, handle: BufferHandle, data: &[u8]) -> Result<()>;
    fn download_data(&mut self, handle: BufferHandle) -> Result<Vec<u8>>;
    fn execute_kernel(&mut self, kernel: &Kernel, args: &[KernelArg]) -> Result<()>;
}
```

#### 后端类型枚举

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    Candle,
    LlamaCpp,
    OnnxRuntime,
    TensorRT,
    Tvm,
    Wasm,
    External(ExternalBridgeType),
}
```

#### 模型类型枚举

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    Text(TextModelType),
    Image(ImageModelType),
    Audio(AudioModelType),
    Video(VideoModelType),
    Multimodal(MultimodalModelType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextModelType {
    LanguageModel,      // LLM
    Embedding,          // 嵌入模型
    Classification,     // 分类
    Translation,        // 翻译
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageModelType {
    Generation,         // 生成（Stable Diffusion等）
    Understanding,      // 理解（CLIP等）
    Classification,     // 分类
    Segmentation,       // 分割
}

// ... 其他类型类似
```

### 2.2 推理后端抽象

```rust
pub trait InferenceBackend: Send + Sync {
    type ModelHandle: Send + Sync;
    
    fn name(&self) -> &str;
    fn backend_type(&self) -> BackendType;
    fn supported_formats(&self) -> &[ModelFormat];
    fn supported_accelerators(&self) -> &[AcceleratorType];
    
    async fn load_model(
        &self,
        path: &Path,
        config: BackendConfig,
        accelerator: Option<AcceleratorType>,
    ) -> Result<Self::ModelHandle>;
    
    async fn unload_model(&self, handle: Self::ModelHandle) -> Result<()>;
    
    async fn infer(
        &self,
        handle: &Self::ModelHandle,
        input: BackendInput,
        options: InferenceOptions,
    ) -> Result<BackendOutput>;
    
    // 批处理推理
    async fn infer_batch(
        &self,
        handle: &Self::ModelHandle,
        inputs: Vec<BackendInput>,
        options: InferenceOptions,
    ) -> Result<Vec<BackendOutput>>;
    
    // 获取硬件信息
    fn get_accelerator_info(&self, accelerator: AcceleratorType) -> Result<AcceleratorInfo>;
}
```

### 2.3 硬件加速层设计

硬件加速层提供统一的硬件抽象，支持多种硬件后端：

```rust
// 硬件加速器工厂
pub struct AcceleratorFactory;

impl AcceleratorFactory {
    pub fn create(accelerator_type: AcceleratorType) -> Result<Box<dyn HardwareAccelerator>> {
        match accelerator_type {
            AcceleratorType::Cpu => Ok(Box::new(CpuAccelerator::new()?)),
            AcceleratorType::Cuda { device_id } => {
                Ok(Box::new(CudaAccelerator::new(device_id)?))
            }
            AcceleratorType::Metal { device_id } => {
                Ok(Box::new(MetalAccelerator::new(device_id)?))
            }
            AcceleratorType::Vulkan { device_id } => {
                Ok(Box::new(VulkanAccelerator::new(device_id)?))
            }
            AcceleratorType::WebGpu => Ok(Box::new(WebGpuAccelerator::new()?)),
            AcceleratorType::Wasm => Ok(Box::new(WasmAccelerator::new()?)),
        }
    }
    
    pub fn detect_available() -> Vec<AcceleratorType> {
        let mut available = vec![AcceleratorType::Cpu];
        
        // 检测 CUDA
        if CudaAccelerator::is_available() {
            for i in 0..CudaAccelerator::device_count() {
                available.push(AcceleratorType::Cuda { device_id: i });
            }
        }
        
        // 检测 Metal
        if MetalAccelerator::is_available() {
            for i in 0..MetalAccelerator::device_count() {
                available.push(AcceleratorType::Metal { device_id: i });
            }
        }
        
        // 检测 Vulkan
        if VulkanAccelerator::is_available() {
            for i in 0..VulkanAccelerator::device_count() {
                available.push(AcceleratorType::Vulkan { device_id: i });
            }
        }
        
        // 检测 WebGPU
        if WebGpuAccelerator::is_available() {
            available.push(AcceleratorType::WebGpu);
        }
        
        // WASM 总是可用（在 WASM 环境中）
        #[cfg(target_arch = "wasm32")]
        available.push(AcceleratorType::Wasm);
        
        available
    }
}
```

### 2.4 Mold 工厂模式

根据模型格式和可用硬件自动选择最佳 Mold：

```rust
pub struct MoldFactory {
    backends: HashMap<BackendType, Box<dyn InferenceBackend>>,
    accelerators: Vec<AcceleratorType>,
}

impl MoldFactory {
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
                Ok(Box::new(CandleMoldImpl::new(backend, accelerator, model_path).await?))
            }
            BackendType::LlamaCpp => {
                Ok(Box::new(LlamaCppMoldImpl::new(backend, accelerator, model_path).await?))
            }
            BackendType::OnnxRuntime => {
                Ok(Box::new(OnnxMoldImpl::new(backend, accelerator, model_path).await?))
            }
            BackendType::Tvm => {
                Ok(Box::new(TvmMoldImpl::new(backend, accelerator, model_path).await?))
            }
            BackendType::External(bridge_type) => {
                Ok(Box::new(ExternalMoldImpl::new(
                    backend,
                    accelerator,
                    model_path,
                    bridge_type,
                ).await?))
            }
            _ => Err(FerrousForgeError::UnsupportedBackend(backend.backend_type())),
        }
    }
    
    fn select_backend(&self, format: ModelFormat) -> Result<&dyn InferenceBackend> {
        // 根据格式选择最佳后端
        match format {
            ModelFormat::Gguf => self.backends.get(&BackendType::LlamaCpp),
            ModelFormat::Onnx => self.backends.get(&BackendType::OnnxRuntime),
            ModelFormat::Safetensors => self.backends.get(&BackendType::Candle),
            ModelFormat::Tvm => self.backends.get(&BackendType::Tvm),
            _ => None,
        }
        .ok_or_else(|| FerrousForgeError::UnsupportedFormat(format))
        .map(|b| b.as_ref())
    }
    
    fn select_accelerator(&self, metadata: &ModelMetadata) -> Result<AcceleratorType> {
        // 根据模型需求和可用硬件选择最佳加速器
        let requirements = &metadata.requirements;
        
        // 优先使用 GPU（如果可用且需要）
        if requirements.gpu_required {
            for accel in &self.accelerators {
                match accel {
                    AcceleratorType::Cuda { .. } | 
                    AcceleratorType::Metal { .. } | 
                    AcceleratorType::Vulkan { .. } => {
                        return Ok(*accel);
                    }
                    _ => {}
                }
            }
            return Err(FerrousForgeError::GpuRequired);
        }
        
        // 否则使用 CPU
        Ok(AcceleratorType::Cpu)
    }
}
```

### 2.3 存储抽象

```rust
pub trait Storage: Send + Sync {
    async fn get_model_path(&self, name: &str) -> Result<PathBuf>;
    async fn download_model(&self, name: &str, source: &str) -> Result<()>;
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
    async fn remove_model(&self, name: &str) -> Result<()>;
    async fn model_exists(&self, name: &str) -> bool;
}
```

## 3. 数据模型设计

### 3.1 模型元数据

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
    pub format: ModelFormat,
    pub size: u64,                    // 文件大小（字节）
    pub parameters: Option<u64>,      // 参数量
    pub quantization: Option<QuantizationType>,
    pub architecture: String,
    pub context_size: Option<usize>,
    pub input_shapes: Vec<Shape>,
    pub output_shapes: Vec<Shape>,
    pub requirements: ModelRequirements,
    pub tags: Vec<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub author: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequirements {
    pub min_memory: Option<u64>,      // 最小内存（字节）
    pub min_vram: Option<u64>,        // 最小显存（字节）
    pub cpu_cores: Option<usize>,      // CPU 核心数
    pub gpu_required: bool,
    pub gpu_compute_capability: Option<String>, // CUDA 计算能力
}
```

### 3.2 推理选项

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOptions {
    // 通用选项
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
    
    // 文本特定
    pub stop_sequences: Vec<String>,
    pub repetition_penalty: Option<f32>,
    
    // 图像特定
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    
    // 资源限制
    pub max_memory: Option<u64>,
    pub timeout: Option<Duration>,
}
```

### 3.3 API 请求/响应

```rust
// 生成请求
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub system: Option<String>,
    pub context: Option<Vec<u64>>,    // 上下文 token IDs
    pub stream: Option<bool>,
    pub options: Option<InferenceOptions>,
}

// 生成响应
#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub response: String,
    pub done: bool,
    pub context: Option<Vec<u64>>,
    pub total_duration: Option<Duration>,
    pub load_duration: Option<Duration>,
    pub prompt_eval_count: Option<usize>,
    pub prompt_eval_duration: Option<Duration>,
    pub eval_count: Option<usize>,
    pub eval_duration: Option<Duration>,
}

// 流式响应块
#[derive(Debug, Serialize)]
pub struct GenerateResponseChunk {
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub response: String,
    pub done: bool,
}
```

## 4. 生命周期管理

### 4.1 模型生命周期

```
创建 → 加载 → 就绪 → 推理 → 卸载 → 销毁
  │      │      │      │      │      │
  │      │      │      │      │      └─→ 释放资源
  │      │      │      │      └─→ 从内存卸载
  │      │      │      └─→ 执行推理
  │      │      └─→ 可用状态
  │      └─→ 加载到内存
  └─→ 创建实例
```

### 4.2 资源管理策略

1. **懒加载**：模型按需加载
2. **自动卸载**：空闲一段时间后自动卸载
3. **LRU 缓存**：最近使用的模型保持在内存
4. **内存限制**：总内存使用限制
5. **并发限制**：每个模型的并发推理数限制

## 5. 错误处理

### 5.1 错误类型层次

```rust
#[derive(Debug, thiserror::Error)]
pub enum FerrousForgeError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),
    
    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),
    
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
    
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
    
    #[error("API error: {0}")]
    Api(#[from] ApiError),
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model not found: {0}")]
    NotFound(String),
    
    #[error("Model already loaded: {0}")]
    AlreadyLoaded(String),
    
    #[error("Model not loaded: {0}")]
    NotLoaded(String),
    
    #[error("Unsupported model format: {0}")]
    UnsupportedFormat(String),
    
    #[error("Insufficient memory: required {required}, available {available}")]
    InsufficientMemory { required: u64, available: u64 },
}
```

## 6. 配置系统

### 6.1 配置结构

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub storage: StorageConfig,
    pub inference: InferenceConfig,
    pub logging: LoggingConfig,
    pub metrics: Option<MetricsConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
    pub max_connections: Option<usize>,
    pub timeout: Option<Duration>,
    pub cors: Option<CorsConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StorageConfig {
    pub base_path: PathBuf,
    pub cache_size: Option<u64>,        // 缓存大小限制
    pub auto_download: bool,            // 自动下载缺失模型
    pub download_timeout: Option<Duration>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
    pub default_backend: String,
    pub max_concurrent_requests: usize,
    pub max_memory_per_model: Option<u64>,
    pub auto_unload_timeout: Option<Duration>,
    pub backends: HashMap<String, BackendConfig>,
}
```

### 6.2 配置加载优先级

1. 命令行参数（最高优先级）
2. 环境变量
3. 配置文件
4. 默认值（最低优先级）

## 7. 性能优化策略

### 7.1 内存管理

- **模型量化**：支持 INT8/INT4 量化
- **分片加载**：大模型分片加载
- **内存池**：预分配内存池
- **零拷贝**：尽可能避免数据拷贝

### 7.2 并发处理

- **异步 I/O**：使用 tokio 异步运行时
- **任务队列**：优先级任务队列
- **批处理**：合并相似请求
- **连接池**：后端连接池

### 7.3 硬件加速

- **CPU**：纯 CPU 计算，通用支持
- **CUDA**：NVIDIA GPU 支持，高性能
- **Metal**：Apple Silicon GPU 支持
- **Vulkan**：跨平台 GPU API，支持多种 GPU
- **WebGPU**：现代 Web GPU API，浏览器支持
- **WASM SIMD**：WebAssembly SIMD 加速
- **自动选择**：根据硬件和模型需求自动选择最佳加速器
- **TVM 优化**：使用 Apache TVM 进行模型编译和优化

## 8. 安全设计

### 8.1 输入验证

- **大小限制**：输入大小限制
- **格式验证**：严格验证输入格式
- **内容过滤**：可选的敏感内容过滤

### 8.2 资源限制

- **内存限制**：每个请求的内存限制
- **时间限制**：推理超时
- **并发限制**：最大并发请求数
- **速率限制**：API 速率限制

### 8.3 认证授权

- **API 密钥**：简单的 API 密钥认证
- **JWT**：JWT token 认证（可选）
- **OAuth2**：OAuth2 支持（未来）

## 9. 监控和可观测性

### 9.1 日志

- **结构化日志**：使用 tracing
- **日志级别**：可配置的日志级别
- **日志输出**：文件、控制台、远程

### 9.2 指标

- **Prometheus**：指标导出
- **关键指标**：
  - 请求数、延迟、错误率
  - 模型加载/卸载次数
  - 内存/GPU 使用率
  - 推理吞吐量

### 9.3 追踪

- **分布式追踪**：OpenTelemetry 支持
- **请求追踪**：端到端请求追踪

## 10. 扩展点设计

### 10.1 插件系统

```rust
pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    
    fn on_model_load(&self, model: &dyn Model) -> Result<()>;
    fn on_model_unload(&self, model: &dyn Model) -> Result<()>;
    fn on_inference_start(&self, request: &InferenceRequest) -> Result<()>;
    fn on_inference_end(&self, response: &InferenceResponse) -> Result<()>;
}
```

### 10.2 中间件系统

```rust
pub trait Middleware: Send + Sync {
    async fn handle(
        &self,
        request: Request,
        next: Next<'_>,
    ) -> Result<Response>;
}
```

## 11. 测试策略

### 11.1 单元测试

- 每个模块独立的单元测试
- Mock 依赖进行隔离测试

### 11.2 集成测试

- API 端到端测试
- 模型加载和推理测试
- 多模型并发测试

### 11.3 性能测试

- 基准测试
- 压力测试
- 内存泄漏测试

## 12. 部署考虑

### 12.1 容器化

- Docker 镜像
- 多阶段构建
- 最小化镜像大小

### 12.2 系统服务

- systemd 服务文件
- Windows 服务支持

### 12.3 云原生

- Kubernetes 部署
- Helm charts
- 健康检查端点


# FerrousForge 项目结构设计

## 项目概述

FerrousForge 是一个参考 Ollama 架构的 Rust 实现，支持多种模型类型（文本、图像、音频、视频等）的统一推理服务。

## 整体架构

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

## 核心设计理念

1. **模块化设计**：每个组件独立，便于维护和扩展
2. **类型安全**：充分利用 Rust 的类型系统保证安全性
3. **异步优先**：使用 async/await 处理并发请求
4. **可扩展性**：易于添加新的模型类型和推理后端
5. **性能优化**：利用 Rust 的零成本抽象和内存安全
6. **统一接口**：通过 "Mold" trait 统一所有模型类型的接口
7. **硬件抽象**：支持多种硬件加速后端（CPU、CUDA、Metal、Vulkan、WebGPU、WASM）

## 项目结构

```
ferrousforge/
├── Cargo.toml                    # 项目主配置文件
├── Cargo.lock                    # 依赖锁定文件
├── .gitignore                    # Git 忽略文件
├── README.md                     # 项目说明文档
├── LICENSE                       # 许可证文件
│
├── src/                          # 源代码目录
│   ├── main.rs                   # 程序入口点
│   ├── lib.rs                    # 库入口点（供其他项目使用）
│   │
│   ├── server/                   # 服务器模块
│   │   ├── mod.rs                # 模块导出
│   │   ├── http/                 # HTTP 服务器
│   │   │   ├── mod.rs
│   │   │   ├── server.rs         # HTTP 服务器实现
│   │   │   ├── handlers/         # 请求处理器
│   │   │   │   ├── mod.rs
│   │   │   │   ├── generate.rs   # 生成请求处理
│   │   │   │   ├── chat.rs       # 聊天请求处理
│   │   │   │   ├── embed.rs      # 嵌入请求处理
│   │   │   │   ├── models.rs     # 模型管理请求处理
│   │   │   │   └── health.rs     # 健康检查
│   │   │   ├── middleware.rs     # 中间件（认证、日志等）
│   │   │   └── routes.rs         # 路由定义
│   │   ├── grpc/                 # gRPC 服务器（可选）
│   │   │   ├── mod.rs
│   │   │   ├── server.rs
│   │   │   └── proto/            # Protocol Buffers 定义
│   │   │       └── service.proto
│   │   └── websocket/            # WebSocket 服务器（流式响应）
│   │       ├── mod.rs
│   │       └── server.rs
│   │
│   ├── core/                     # 核心业务逻辑
│   │   ├── mod.rs
│   │   ├── engine.rs             # 推理引擎核心
│   │   ├── registry.rs           # 模型注册表
│   │   ├── scheduler.rs          # 资源调度器
│   │   ├── context.rs            # 请求上下文管理
│   │   └── gateway.rs            # API 网关（路由、限流等）
│   │
│   ├── gateway/                  # API 网关模块
│   │   ├── mod.rs
│   │   ├── router.rs             # 路由管理
│   │   ├── rate_limit.rs         # 速率限制
│   │   ├── load_balancer.rs      # 负载均衡
│   │   └── middleware.rs         # 网关中间件
│   │
│   ├── management/               # 模型管理模块
│   │   ├── mod.rs
│   │   ├── manager.rs            # 模型管理器
│   │   ├── lifecycle.rs          # 生命周期管理
│   │   ├── versioning.rs         # 版本管理
│   │   └── health.rs             # 健康检查
│   │
│   ├── plugins/                  # 插件系统
│   │   ├── mod.rs
│   │   ├── registry.rs           # 插件注册表
│   │   ├── loader.rs             # 插件加载器
│   │   ├── hooks.rs              # 生命周期钩子
│   │   └── traits.rs             # 插件 trait
│   │
│   ├── monitoring/               # 监控告警模块
│   │   ├── mod.rs
│   │   ├── metrics.rs            # 指标收集
│   │   ├── alerts.rs             # 告警系统
│   │   ├── tracing.rs             # 分布式追踪
│   │   └── dashboard.rs          # 监控面板
│   │
│   ├── models/                   # 模型抽象和实现
│   │   ├── mod.rs                # 模型 trait 定义
│   │   ├── mold.rs               # 统一模型接口 "Mold" trait
│   │   ├── traits.rs             # 通用模型 trait
│   │   ├── base.rs               # 基础模型实现
│   │   │
│   │   ├── molds/                # Mold 实现
│   │   │   ├── mod.rs
│   │   │   ├── candle_mold.rs    # CandleMold 实现
│   │   │   ├── llama_cpp_mold.rs # LlamaCppMold 实现
│   │   │   └── external_mold.rs  # ExternalMold 实现（桥接外部框架）
│   │   │
│   │   ├── text/                 # 文本模型
│   │   │   ├── mod.rs
│   │   │   ├── model.rs          # 文本模型实现
│   │   │   ├── llama.rs          # LLaMA 系列
│   │   │   ├── mistral.rs        # Mistral 系列
│   │   │   └── transformers.rs   # Transformers 模型
│   │   │
│   │   ├── image/                # 图像模型
│   │   │   ├── mod.rs
│   │   │   ├── model.rs          # 图像模型实现
│   │   │   ├── diffusion.rs      # 扩散模型（Stable Diffusion等）
│   │   │   ├── vision.rs         # 视觉理解模型（CLIP等）
│   │   │   └── generation.rs     # 图像生成模型
│   │   │
│   │   ├── audio/                # 音频模型
│   │   │   ├── mod.rs
│   │   │   ├── model.rs          # 音频模型实现
│   │   │   ├── speech.rs         # 语音识别/合成（Whisper等）
│   │   │   ├── music.rs          # 音乐生成模型
│   │   │   └── audio_analysis.rs # 音频分析模型
│   │   │
│   │   ├── video/                # 视频模型
│   │   │   ├── mod.rs
│   │   │   ├── model.rs          # 视频模型实现
│   │   │   ├── generation.rs     # 视频生成模型
│   │   │   └── understanding.rs  # 视频理解模型
│   │   │
│   │   └── multimodal/           # 多模态模型
│   │       ├── mod.rs
│   │       ├── model.rs          # 多模态模型实现
│   │       └── llava.rs          # LLaVA 等模型
│   │
│   ├── inference/                # 推理后端抽象
│   │   ├── mod.rs
│   │   ├── backend.rs            # 后端 trait 定义
│   │   ├── backends/             # 具体后端实现
│   │   │   ├── mod.rs
│   │   │   ├── llama_cpp.rs      # llama.cpp 后端
│   │   │   ├── candle.rs         # Candle 后端（纯 Rust）
│   │   │   ├── onnx.rs           # ONNX Runtime 后端
│   │   │   ├── tensorrt.rs       # TensorRT 后端（NVIDIA）
│   │   │   ├── tvm.rs            # TVM 后端（Apache TVM）
│   │   │   └── wasm.rs           # WebAssembly 后端
│   │   ├── executor.rs           # 推理执行器
│   │   └── hardware/             # 硬件加速抽象
│   │       ├── mod.rs
│   │       ├── accelerator.rs    # 加速器 trait
│   │       ├── cpu.rs            # CPU 后端
│   │       ├── cuda.rs           # CUDA 后端（NVIDIA）
│   │       ├── metal.rs          # Metal 后端（Apple）
│   │       ├── vulkan.rs         # Vulkan 后端
│   │       ├── webgpu.rs         # WebGPU 后端
│   │       └── wasm.rs           # WASM 后端
│   │
│   ├── storage/                  # 存储管理
│   │   ├── mod.rs
│   │   ├── manager.rs            # 存储管理器
│   │   ├── registry.rs           # 模型注册表存储
│   │   ├── cache.rs              # 缓存管理
│   │   └── download.rs           # 模型下载
│   │
│   ├── config/                   # 配置管理
│   │   ├── mod.rs
│   │   ├── settings.rs           # 配置结构定义
│   │   ├── loader.rs             # 配置加载器
│   │   └── defaults.rs           # 默认配置
│   │
│   ├── api/                      # API 类型定义
│   │   ├── mod.rs
│   │   ├── request.rs            # 请求类型
│   │   ├── response.rs           # 响应类型
│   │   ├── error.rs              # 错误类型
│   │   └── streaming.rs          # 流式响应类型
│   │
│   ├── utils/                    # 工具函数
│   │   ├── mod.rs
│   │   ├── logging.rs            # 日志工具
│   │   ├── metrics.rs            # 指标收集
│   │   ├── validation.rs         # 数据验证
│   │   └── encoding.rs           # 编码/解码工具
│   │
│   └── cli/                      # 命令行接口
│       ├── mod.rs
│       ├── commands/             # 命令实现
│       │   ├── mod.rs
│       │   ├── serve.rs          # 启动服务器
│       │   ├── pull.rs           # 下载模型
│       │   ├── list.rs           # 列出模型
│       │   ├── run.rs            # 运行模型
│       │   └── remove.rs         # 删除模型
│       └── args.rs               # 参数解析
│
├── tests/                        # 测试目录
│   ├── integration/              # 集成测试
│   │   ├── mod.rs
│   │   ├── server_test.rs        # 服务器测试
│   │   ├── model_test.rs         # 模型测试
│   │   └── api_test.rs           # API 测试
│   ├── unit/                     # 单元测试
│   │   └── mod.rs
│   └── fixtures/                 # 测试数据
│       └── models/               # 测试模型文件
│
├── examples/                     # 示例代码
│   ├── basic_usage.rs            # 基础使用示例
│   ├── text_generation.rs        # 文本生成示例
│   ├── image_generation.rs       # 图像生成示例
│   └── streaming.rs              # 流式响应示例
│
├── docs/                         # 文档目录
│   ├── architecture.md           # 架构文档
│   ├── api.md                    # API 文档
│   ├── models.md                 # 模型支持文档
│   └── development.md             # 开发指南
│
├── scripts/                      # 辅助脚本
│   ├── build.sh                  # 构建脚本
│   ├── test.sh                   # 测试脚本
│   └── release.sh                # 发布脚本
│
└── resources/                    # 资源文件
    ├── config.toml.example       # 配置示例
    └── models/                   # 模型存储目录（运行时创建）
```

## 核心模块详细说明

### 1. `src/core/` - 核心引擎

**职责**：提供统一的推理引擎接口和任务调度

- **engine.rs**: 
  - 推理引擎主入口
  - 管理模型生命周期
  - 处理推理请求队列
  - 资源分配和释放

- **registry.rs**:
  - 模型注册表
  - 模型元数据管理
  - 模型查找和加载

- **scheduler.rs**:
  - 资源调度算法
  - 优先级队列
  - 并发控制
  - 资源配额管理
  - 负载均衡

- **context.rs**:
  - 请求上下文
  - 会话管理
  - 状态跟踪

- **gateway.rs**:
  - API 网关核心
  - 请求路由
  - 限流和熔断
  - 请求聚合

### 2. `src/models/` - 模型抽象层

**职责**：定义统一的模型接口，实现各种模型类型

**核心设计：统一模型接口 "Mold"**

"Mold" 是 FerrousForge 的核心抽象，提供统一的模型接口，所有模型类型都通过 Mold 实现。

**Mold Trait 设计**：
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
}

// Mold 实现类型
pub enum MoldType {
    CandleMold(Box<dyn CandleMold>),
    LlamaCppMold(Box<dyn LlamaCppMold>),
    ExternalMold(Box<dyn ExternalMold>),
}

// CandleMold: 纯 Rust 实现
pub trait CandleMold: Mold {
    // Candle 特定方法
}

// LlamaCppMold: C++ 高性能后端
pub trait LlamaCppMold: Mold {
    // llama.cpp 特定方法
}

// ExternalMold: 外部框架桥接
pub trait ExternalMold: Mold {
    // 外部框架桥接方法
    fn bridge_type(&self) -> ExternalBridgeType;
}
```

**Mold 的优势**：
- 统一的接口，所有模型类型使用相同的 API
- 类型安全，编译时检查
- 易于扩展，添加新的 Mold 实现
- 运行时选择，根据模型格式自动选择 Mold

**各模型类型特点**：
- **text/**: 文本生成、对话、嵌入
- **image/**: 图像生成、理解、编辑
- **audio/**: 语音识别、合成、音乐生成
- **video/**: 视频生成、理解
- **multimodal/**: 跨模态理解

### 3. `src/inference/` - 推理后端

**职责**：抽象不同的推理后端，提供统一接口

**计算后端类型**：
- **candle**: 纯 Rust 实现（推荐用于快速开发）
- **llama_cpp**: 通过 FFI 调用 llama.cpp（C++，高性能）
- **onnx**: ONNX Runtime 后端（跨平台）
- **tvm**: Apache TVM 后端（深度学习编译器）
- **tensorrt**: NVIDIA TensorRT（NVIDIA GPU 优化）
- **wasm**: WebAssembly 后端（浏览器支持）

**硬件加速层**：
- **CPU**: 纯 CPU 计算
- **CUDA**: NVIDIA GPU 加速
- **Metal**: Apple Silicon GPU 加速
- **Vulkan**: 跨平台 GPU API
- **WebGPU**: 现代 Web GPU API
- **WASM**: WebAssembly SIMD 加速

**设计模式**：
- 策略模式：运行时选择后端
- 适配器模式：硬件加速层适配不同后端
- 工厂模式：根据模型格式和硬件自动选择最佳后端

### 4. `src/server/` - 服务器层

**职责**：提供多种协议支持

- **HTTP**: RESTful API，兼容 Ollama API
- **gRPC**: 高性能 RPC（可选）
- **WebSocket**: 流式响应支持

**API 端点设计**（参考 Ollama）：
- `POST /api/generate` - 文本生成
- `POST /api/chat` - 对话
- `POST /api/embed` - 嵌入
- `GET /api/tags` - 列出模型
- `POST /api/pull` - 下载模型
- `DELETE /api/delete` - 删除模型

### 5. `src/storage/` - 存储管理

**职责**：模型文件管理、缓存、下载

- **manager.rs**: 统一存储接口
- **registry.rs**: 模型清单管理（类似 Ollama 的 Modelfile）
- **cache.rs**: 模型缓存策略
- **download.rs**: 从远程仓库下载模型

### 6. `src/config/` - 配置管理

**职责**：配置加载、验证、默认值

支持：
- 配置文件（TOML/YAML）
- 环境变量
- 命令行参数
- 运行时配置

### 7. `src/api/` - API 类型

**职责**：定义请求/响应类型，确保类型安全

- 使用 Serde 进行序列化
- 统一的错误处理
- 流式响应类型

### 8. `src/cli/` - 命令行工具

**职责**：提供命令行界面

命令：
- `ferrousforge serve` - 启动服务器
- `ferrousforge pull <model>` - 下载模型
- `ferrousforge list` - 列出已安装模型
- `ferrousforge run <model> <prompt>` - 运行模型
- `ferrousforge remove <model>` - 删除模型

### 9. `src/gateway/` - API 网关

**职责**：提供统一的 API 入口和高级功能

- **router.rs**: 路由管理，支持动态路由
- **rate_limit.rs**: 速率限制，防止滥用
- **load_balancer.rs**: 负载均衡，分发请求
- **middleware.rs**: 网关中间件（认证、日志、监控等）

### 10. `src/management/` - 模型管理

**职责**：模型生命周期和版本管理

- **manager.rs**: 模型管理器，统一管理所有模型
- **lifecycle.rs**: 生命周期管理（加载、卸载、更新）
- **versioning.rs**: 版本管理，支持多版本共存
- **health.rs**: 健康检查，监控模型状态

### 11. `src/plugins/` - 插件系统

**职责**：提供可扩展的插件机制

- **registry.rs**: 插件注册表
- **loader.rs**: 动态加载插件（支持 WASM 插件）
- **hooks.rs**: 生命周期钩子（模型加载/卸载、推理前后等）
- **traits.rs**: 插件接口定义

### 12. `src/monitoring/` - 监控告警

**职责**：系统监控和告警

- **metrics.rs**: 指标收集（Prometheus 集成）
- **alerts.rs**: 告警系统（阈值告警、异常检测）
- **tracing.rs**: 分布式追踪（OpenTelemetry）
- **dashboard.rs**: 监控面板数据提供

## 数据流设计

```
客户端请求
    ↓
HTTP/gRPC/WebSocket 服务器
    ↓
请求处理器 (handlers/)
    ↓
核心引擎 (core/engine.rs)
    ↓
模型注册表 (core/registry.rs)
    ↓
具体模型实现 (models/*/)
    ↓
推理后端 (inference/backends/*)
    ↓
模型文件 (storage/)
    ↓
响应返回
```

## 关键技术选型建议

### 异步运行时
- **tokio**: 推荐，生态成熟

### Web 框架
- **axum**: 现代、类型安全、性能好
- **warp**: 轻量级备选

### 序列化
- **serde**: Rust 标准序列化库

### 配置
- **config**: 配置管理
- **toml**: TOML 解析

### 日志
- **tracing**: 结构化日志和追踪

### 指标
- **prometheus**: 监控指标

### HTTP 客户端（下载）
- **reqwest**: 异步 HTTP 客户端

### 文件系统
- **tokio::fs**: 异步文件操作

## 扩展性考虑

1. **插件系统**：支持动态加载模型后端和功能插件
   - WASM 插件支持
   - 生命周期钩子
   - 自定义中间件
2. **中间件系统**：请求处理链
   - 网关中间件
   - 模型中间件
   - 响应中间件
3. **钩子系统**：生命周期钩子
   - 模型加载/卸载
   - 推理前后
   - 错误处理
4. **事件系统**：模型加载/卸载事件
   - 事件总线
   - 事件订阅
   - 事件过滤
5. **外部框架桥接**：通过 ExternalMold 桥接外部框架
   - PyTorch 桥接
   - TensorFlow 桥接
   - JAX 桥接

## 性能优化方向

1. **模型缓存**：内存中保持热模型
2. **批处理**：合并多个请求
3. **量化支持**：支持 INT8/INT4 量化
4. **硬件加速**：多后端支持
   - CUDA（NVIDIA GPU）
   - Metal（Apple Silicon）
   - Vulkan（跨平台 GPU）
   - WebGPU（现代 Web GPU）
   - WASM SIMD（WebAssembly 加速）
5. **并发推理**：多模型并行
6. **智能后端选择**：根据硬件和模型自动选择最佳后端
7. **TVM 优化**：使用 TVM 进行模型编译优化

## 安全考虑

1. **输入验证**：防止恶意输入
2. **资源限制**：内存、CPU、GPU 限制
3. **认证授权**：API 密钥、JWT
4. **沙箱隔离**：模型执行隔离

## 下一步讨论点

1. **模型格式**：支持哪些格式？（GGUF, ONNX, Safetensors等）
2. **后端优先级**：优先实现哪个后端？
3. **API 兼容性**：是否需要完全兼容 Ollama API？
4. **部署方式**：Docker、系统服务、云原生？
5. **多租户**：是否需要支持多用户隔离？


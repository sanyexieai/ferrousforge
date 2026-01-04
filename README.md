# FerrousForge

一个参考 Ollama 架构的 Rust 实现，支持多种模型类型（文本、图像、音频、视频等）的统一推理服务。

[![CI](https://github.com/your-org/ferrousforge/workflows/CI/badge.svg)](https://github.com/your-org/ferrousforge/actions)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)

## 📋 项目状态

**当前阶段**: 开发阶段 🚀

项目已初始化，基础架构和 CI/CD 已配置完成。

## 🎯 项目目标

- **多模型支持**: 统一接口支持文本、图像、音频、视频等多种模型类型
- **高性能**: 利用 Rust 的性能优势，提供高效的推理服务
- **易用性**: 提供简洁的 API 和 CLI 工具
- **可扩展**: 模块化设计，易于添加新的模型类型和后端
- **生产就绪**: 完善的错误处理、日志、监控等功能

## 🏗️ 架构特点

### 整体架构

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

### 核心设计

- **统一模型接口 "Mold"**: 所有模型类型通过 Mold trait 统一抽象
- **分层架构**: API网关 → 模型管理 → Mold抽象 → 计算后端 → 硬件加速
- **异步优先**: 基于 tokio 的异步运行时
- **类型安全**: 充分利用 Rust 的类型系统
- **硬件抽象**: 自动适配多种硬件加速器

### 支持的模型类型

- ✅ **文本模型**: LLM、嵌入模型、分类等
- ✅ **图像模型**: 生成（Stable Diffusion）、理解（CLIP）等
- ✅ **音频模型**: 语音识别（Whisper）、合成、音乐生成等
- ✅ **视频模型**: 视频生成、理解等
- ✅ **多模态模型**: LLaVA 等跨模态模型

### 计算后端

- **Candle**: 纯 Rust 实现（推荐用于快速开发）
- **llama.cpp**: 通过 FFI 调用（高性能）
- **ONNX Runtime**: 跨平台支持
- **TVM**: Apache TVM（深度学习编译器）
- **TensorRT**: NVIDIA GPU 加速（未来）
- **WebAssembly**: 浏览器支持

### 硬件加速

- **CPU**: 通用 CPU 计算
- **CUDA**: NVIDIA GPU 加速
- **Metal**: Apple Silicon GPU 加速
- **Vulkan**: 跨平台 GPU API
- **WebGPU**: 现代 Web GPU API
- **WASM**: WebAssembly SIMD 加速

## 📁 项目结构

```
ferrousforge/
├── src/
│   ├── core/              # 核心引擎
│   ├── models/            # 模型抽象和实现
│   │   ├── text/         # 文本模型
│   │   ├── image/        # 图像模型
│   │   ├── audio/        # 音频模型
│   │   ├── video/        # 视频模型
│   │   └── multimodal/   # 多模态模型
│   ├── inference/        # 推理后端
│   ├── server/           # 服务器（HTTP/gRPC/WebSocket）
│   ├── storage/          # 存储管理
│   ├── config/           # 配置管理
│   ├── api/              # API 类型定义
│   ├── utils/            # 工具函数
│   └── cli/              # 命令行工具
├── docs/                 # 文档
│   ├── ARCHITECTURE.md   # 架构设计
│   └── IMPLEMENTATION_ROADMAP.md  # 实现路线图
└── PROJECT_STRUCTURE.md  # 项目结构说明
```

详细的项目结构说明请参考 [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)

## 📚 文档

- [项目结构设计](./PROJECT_STRUCTURE.md) - 详细的目录和文件说明
- [架构设计文档](./docs/ARCHITECTURE.md) - 核心架构和设计决策
- [Mold 设计文档](./docs/MOLD_DESIGN.md) - 统一模型接口设计
- [实现路线图](./docs/IMPLEMENTATION_ROADMAP.md) - 开发阶段和优先级
- [CI/CD 配置](./docs/CI_CD.md) - 持续集成和部署说明

## 🚀 计划中的功能

### MVP（最小可行产品）

- [ ] 文本模型加载和推理
- [ ] HTTP API 服务器
- [ ] 基础 CLI 工具
- [ ] 模型下载和管理
- [ ] 一个推理后端（Candle 或 llama.cpp）

### 后续功能

- [ ] 图像模型支持
- [ ] 音频模型支持
- [ ] 视频模型支持
- [ ] TVM 后端支持
- [ ] 多个推理后端
- [ ] WebSocket 流式响应
- [ ] gRPC 支持
- [ ] API 网关功能
- [ ] 插件系统
- [ ] 监控告警系统
- [ ] 认证授权
- [ ] 模型量化支持
- [ ] 多硬件加速器支持（CUDA、Metal、Vulkan、WebGPU）

## 🛠️ 技术栈（计划）

- **异步运行时**: tokio
- **Web 框架**: axum
- **序列化**: serde
- **日志**: tracing
- **配置**: config
- **HTTP 客户端**: reqwest
- **CLI**: clap

## 💡 设计理念

1. **模块化**: 每个组件独立，便于维护和扩展
2. **类型安全**: 充分利用 Rust 的类型系统
3. **性能优先**: 零成本抽象和内存安全
4. **易用性**: 简洁的 API 和清晰的文档
5. **可扩展**: 易于添加新的模型类型和后端

## 🤝 贡献

欢迎贡献！请查看 [贡献指南](.github/CONTRIBUTING.md) 了解详细信息。

### 开发流程

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 代码规范

- 使用 `cargo fmt` 格式化代码
- 使用 `cargo clippy` 检查代码质量
- 确保所有测试通过
- 遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范

## 📝 许可证

待定

## 🔗 参考

- [Ollama](https://github.com/ollama/ollama) - 参考架构
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 推理后端参考
- [Candle](https://github.com/huggingface/candle) - Rust ML 框架

---

**注意**: 本项目目前处于设计阶段，代码实现尚未开始。欢迎参与讨论和设计！


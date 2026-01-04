//! llama.cpp 后端实现
//! 
//! 通过 FFI 调用 llama.cpp（C++）进行模型推理。
//! 
//! ## 集成说明
//! 
//! 当前实现提供了一个完整的框架，包含：
//! - 模型加载和卸载逻辑框架
//! - 推理执行逻辑框架
//! - 资源管理框架
//! 
//! 要集成实际的 llama.cpp 绑定，需要：
//! 
//! 1. **添加依赖**：在 `Cargo.toml` 中添加 llama.cpp 的 Rust 绑定
//! 2. **更新类型定义**：将 `LlamaCppModelHandle` 中的占位符类型替换为实际的模型和上下文类型
//! 3. **实现实际 API 调用**：取消注释 `load_model`、`unload_model` 和 `do_infer_text` 方法中的实现代码
//! 4. **测试**：使用真实的 GGUF 模型文件进行测试

use crate::Result;
use crate::api::request::InferenceOptions;
use crate::inference::backend::{
    InferenceBackend, BackendConfig, BackendInput, BackendOutput,
    AcceleratorInfo,
};
use crate::models::types::*;
use crate::models::metadata::ModelMetadata;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use futures::Stream;

// 引入 FFI 绑定（如果可用）
#[cfg(feature = "llama-cpp")]
use crate::inference::backends::llama_cpp_ffi::{LlamaCppFFI, LlamaModelPtr, LlamaContextPtr};

/// llama.cpp 模型句柄
/// 
/// 封装已加载的模型实例和相关状态。
/// 
/// 注意：实际的模型和上下文对象应该存储在内部，这里使用 `Option` 来管理生命周期。
pub struct LlamaCppModelHandle {
    /// 模型路径
    pub model_path: PathBuf,
    /// 上下文参数
    pub context_params: LlamaCppContextParams,
    /// 是否已加载
    pub loaded: bool,
    /// 模型元数据
    pub metadata: Option<ModelMetadata>,
    /// 模型实例（FFI 指针）
    #[cfg(feature = "llama-cpp")]
    model_ptr: Option<LlamaModelPtr>,
    /// 上下文实例（FFI 指针）
    #[cfg(feature = "llama-cpp")]
    context_ptr: Option<LlamaContextPtr>,
    /// FFI 包装器
    #[cfg(feature = "llama-cpp")]
    ffi: Option<LlamaCppFFI>,
    /// 模型实例（占位符，用于非 llama-cpp feature）
    #[cfg(not(feature = "llama-cpp"))]
    model_instance: Option<Box<dyn std::any::Any + Send + Sync>>,
    /// 上下文实例（占位符，用于非 llama-cpp feature）
    #[cfg(not(feature = "llama-cpp"))]
    context_instance: Option<Box<dyn std::any::Any + Send + Sync>>,
}

impl std::fmt::Debug for LlamaCppModelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("LlamaCppModelHandle");
        debug.field("model_path", &self.model_path)
            .field("context_params", &self.context_params)
            .field("loaded", &self.loaded)
            .field("metadata", &self.metadata);
        
        #[cfg(feature = "llama-cpp")]
        {
            debug.field("model_ptr", &self.model_ptr.is_some())
                .field("context_ptr", &self.context_ptr.is_some());
        }
        
        #[cfg(not(feature = "llama-cpp"))]
        {
            debug.field("model_instance", &self.model_instance.is_some())
                .field("context_instance", &self.context_instance.is_some());
        }
        
        debug.finish()
    }
}

impl Clone for LlamaCppModelHandle {
    fn clone(&self) -> Self {
        // 注意：克隆时不应该克隆实际的模型实例，只克隆元数据
        Self {
            model_path: self.model_path.clone(),
            context_params: self.context_params.clone(),
            loaded: false,  // 克隆的句柄标记为未加载
            metadata: self.metadata.clone(),
            #[cfg(feature = "llama-cpp")]
            model_ptr: None,
            #[cfg(feature = "llama-cpp")]
            context_ptr: None,
            #[cfg(feature = "llama-cpp")]
            ffi: None,
            #[cfg(not(feature = "llama-cpp"))]
            model_instance: None,
            #[cfg(not(feature = "llama-cpp"))]
            context_instance: None,
        }
    }
}

/// llama.cpp 上下文参数
#[derive(Debug, Clone)]
pub struct LlamaCppContextParams {
    /// GPU 层数（0 表示仅使用 CPU）
    pub n_gpu_layers: i32,
    /// 上下文大小
    pub n_ctx: usize,
    /// 批处理大小
    pub n_batch: usize,
    /// 线程数
    pub n_threads: usize,
    /// 是否使用 mmap
    pub use_mmap: bool,
    /// 是否使用 mlock
    pub use_mlock: bool,
}

impl Default for LlamaCppContextParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            n_ctx: 2048,
            n_batch: 512,
            n_threads: num_cpus::get(),
            use_mmap: true,
            use_mlock: false,
        }
    }
}

/// llama.cpp 后端实现
/// 
/// 提供对 llama.cpp 的封装，实现 `InferenceBackend` trait。
pub struct LlamaCppBackend {
    /// 后端名称
    name: String,
    /// 支持的格式
    supported_formats: Vec<ModelFormat>,
    /// 支持的加速器
    supported_accelerators: Vec<AcceleratorType>,
}

impl LlamaCppBackend {
    /// 创建新的 llama.cpp 后端
    pub fn new() -> Self {
        Self {
            name: "llama.cpp".to_string(),
            supported_formats: vec![ModelFormat::Gguf],
            supported_accelerators: vec![
                AcceleratorType::Cpu,
                AcceleratorType::Cuda { device_id: 0 },
                AcceleratorType::Metal { device_id: 0 },
            ],
        }
    }
    
    /// 从配置创建后端
    pub fn with_config(config: &BackendConfig) -> Self {
        let mut backend = Self::new();
        // 可以根据配置调整支持的加速器
        if config.device_id.is_some() {
            backend.supported_accelerators.push(
                AcceleratorType::Cuda { device_id: config.device_id.unwrap() }
            );
        }
        backend
    }
    
    /// 将 BackendConfig 转换为 LlamaCppContextParams
    fn config_to_params(&self, config: &BackendConfig, accelerator: Option<AcceleratorType>) -> LlamaCppContextParams {
        let n_gpu_layers = match accelerator {
            Some(AcceleratorType::Cuda { device_id: _ }) => {
                // 如果使用 CUDA，设置 GPU 层数
                // 这里可以根据配置或硬件能力动态设置
                35  // 默认值，实际应该根据模型大小调整
            }
            Some(AcceleratorType::Metal { device_id: _ }) => {
                // Metal 支持
                35
            }
            _ => 0,  // CPU only
        };
        
        LlamaCppContextParams {
            n_gpu_layers,
            n_ctx: config.context_size.unwrap_or(2048),
            n_batch: config.batch_size.unwrap_or(512),
            n_threads: config.num_threads.unwrap_or_else(|| num_cpus::get()),
            use_mmap: true,
            use_mlock: false,
        }
    }
    
    /// 加载模型元数据
    /// 
    /// 从模型文件中提取元数据信息。
    async fn load_metadata(&self, path: &Path) -> Result<ModelMetadata> {
        // TODO: 实际实现需要调用 llama.cpp 的 API 来读取模型元数据
        // 这里提供一个占位符实现
        
        let file_name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        Ok(ModelMetadata {
            name: file_name.clone(),
            version: "1.0.0".to_string(),
            model_type: ModelType::Text(TextModelType::LanguageModel),
            format: ModelFormat::Gguf,
            size: std::fs::metadata(path)
                .map(|m| m.len())
                .unwrap_or(0),
            parameters: None,
            quantization: None,
            architecture: "llama".to_string(),
            context_size: Some(2048),
            input_shapes: vec![],
            output_shapes: vec![],
            requirements: crate::models::metadata::ModelRequirements::default(),
            tags: vec!["llama.cpp".to_string()],
            description: Some(format!("llama.cpp model: {}", file_name)),
            license: None,
            author: None,
            created_at: Some(chrono::Utc::now()),
            updated_at: Some(chrono::Utc::now()),
        })
    }
    
    /// 执行文本推理
    /// 
    /// 实际的推理逻辑。
    async fn do_infer_text(
        &self,
        handle: &LlamaCppModelHandle,
        prompt: &str,
        options: &InferenceOptions,
    ) -> Result<String> {
        // 检查模型是否已加载
        #[cfg(feature = "llama-cpp")]
        {
            if !handle.loaded || handle.context_ptr.is_none() || handle.ffi.is_none() {
                return Err(crate::api::error::ModelError::NotLoaded(
                    "Model is not loaded or context is not available".to_string()
                ).into());
            }
        }
        
        #[cfg(not(feature = "llama-cpp"))]
        {
            if !handle.loaded || handle.context_instance.is_none() {
                return Err(crate::api::error::ModelError::NotLoaded(
                    "Model is not loaded or context is not available".to_string()
                ).into());
            }
        }
        
        // 实际的推理实现框架
        // 步骤 1: Tokenize 输入文本
        // let tokens = {
        //     let context = handle.context_instance.as_ref()
        //         .and_then(|ctx| ctx.downcast_ref::<llama_cpp::Context>())
        //         .ok_or_else(|| crate::api::error::InferenceError::Failed(
        //             "Context not available".to_string()
        //         ))?;
        //     
        //     context.tokenize(prompt, true)
        //         .map_err(|e| crate::api::error::InferenceError::Failed(
        //             format!("Tokenization failed: {}", e)
        //         ))?
        // };
        
        // 步骤 2: 设置生成参数
        // let gen_params = llama_cpp::GenerateParams::default()
        //     .with_temperature(options.temperature.unwrap_or(0.7))
        //     .with_top_p(options.top_p.unwrap_or(0.9))
        //     .with_top_k(options.top_k.map(|k| k as i32))
        //     .with_n_predict(options.max_tokens.map(|t| t as i32).unwrap_or(256))
        //     .with_repeat_penalty(options.repetition_penalty.unwrap_or(1.1))
        //     .with_seed(options.seed.map(|s| s as i32));
        
        // 步骤 3: 执行生成
        // let mut output = String::new();
        // {
        //     let context = handle.context_instance.as_ref()
        //         .and_then(|ctx| ctx.downcast_ref::<llama_cpp::Context>())
        //         .ok_or_else(|| crate::api::error::InferenceError::Failed(
        //             "Context not available".to_string()
        //         ))?;
        //     
        //     // 设置输入 tokens
        //     context.eval(tokens, 0, handle.context_params.n_threads)
        //         .map_err(|e| crate::api::error::InferenceError::Failed(
        //             format!("Evaluation failed: {}", e)
        //         ))?;
        //     
        //     // 生成 tokens
        //     loop {
        //         let token = context.sample(gen_params)
        //             .map_err(|e| crate::api::error::InferenceError::Failed(
        //                 format!("Sampling failed: {}", e)
        //             ))?;
        //         
        //         if token == llama_cpp::EOS_TOKEN {
        //             break;
        //         }
        //         
        //         let text = context.token_to_str(token)
        //             .map_err(|e| crate::api::error::InferenceError::Failed(
        //                 format!("Token to string conversion failed: {}", e)
        //             ))?;
        //         
        //         output.push_str(&text);
        //         
        //         // 检查停止序列
        //         if let Some(ref stop_seqs) = options.stop_sequences {
        //             if stop_seqs.iter().any(|seq| output.ends_with(seq)) {
        //                 break;
        //             }
        //         }
        //         
        //         // 检查最大长度
        //         if let Some(max_tokens) = options.max_tokens {
        //             if output.len() >= max_tokens {
        //                 break;
        //             }
        //         }
        //         
        //         // 继续生成
        //         context.eval(vec![token], 0, handle.context_params.n_threads)
        //             .map_err(|e| crate::api::error::InferenceError::Failed(
        //                 format!("Evaluation failed: {}", e)
        //             ))?;
        //     }
        // }
        
        #[cfg(feature = "llama-cpp")]
        {
            // 使用 FFI 进行推理
            let ffi = handle.ffi.as_ref().unwrap();
            let ctx_ptr = handle.context_ptr.unwrap();
            
            // 步骤 1: Tokenize
            let tokens = ffi.tokenize(ctx_ptr, prompt, true)
                .map_err(|e| crate::api::error::InferenceError::Failed(
                    format!("Tokenization failed: {}", e)
                ))?;
            
            if tokens.is_empty() {
                return Err(crate::api::error::InferenceError::Failed(
                    "Tokenization returned empty tokens".to_string()
                ).into());
            }
            
            // 步骤 2: Eval 输入 tokens
            ffi.eval(ctx_ptr, &tokens, 0, handle.context_params.n_threads as i32)
                .map_err(|e| crate::api::error::InferenceError::Failed(
                    format!("Evaluation failed: {}", e)
                ))?;
            
            // 步骤 3: 生成循环
            let mut output = String::new();
            let max_tokens = options.max_tokens.unwrap_or(256);
            let temperature = options.temperature.unwrap_or(0.7);
            let top_p = options.top_p.unwrap_or(0.9);
            let top_k = options.top_k.map(|k| k as i32).unwrap_or(40);
            
            for _ in 0..max_tokens {
                // 采样下一个 token
                let token = ffi.sample(ctx_ptr, temperature, top_p, top_k)
                    .map_err(|e| crate::api::error::InferenceError::Failed(
                        format!("Sampling failed: {}", e)
                    ))?;
                
                // EOS token
                let eos_token = ffi.token_eos();
                if token == eos_token {
                    break;
                }
                
                // 转换为文本
                let text = ffi.token_to_str(ctx_ptr, token)
                    .map_err(|e| crate::api::error::InferenceError::Failed(
                        format!("Token to string conversion failed: {}", e)
                    ))?;
                
                output.push_str(&text);
                
                // 检查停止序列
                if let Some(ref stop_seqs) = options.stop_sequences {
                    if stop_seqs.iter().any(|seq| output.ends_with(seq)) {
                        break;
                    }
                }
                
                // 继续生成（eval 单个 token）
                ffi.eval(ctx_ptr, &[token], 1, handle.context_params.n_threads as i32)
                    .map_err(|e| crate::api::error::InferenceError::Failed(
                        format!("Evaluation failed: {}", e)
                    ))?;
            }
            
            Ok(output)
        }
        
        #[cfg(not(feature = "llama-cpp"))]
        {
            // 占位符实现
            tracing::debug!(
                "Inferencing with llama.cpp: prompt_len={}, temperature={:?}, max_tokens={:?}",
                prompt.len(),
                options.temperature,
                options.max_tokens
            );
            
            Ok(format!("[Placeholder] Generated from prompt ({} chars)", prompt.len()))
        }
    }
}

#[async_trait]
impl InferenceBackend for LlamaCppBackend {
    type ModelHandle = LlamaCppModelHandle;
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn backend_type(&self) -> BackendType {
        BackendType::LlamaCpp
    }
    
    fn supported_formats(&self) -> &[ModelFormat] {
        &self.supported_formats
    }
    
    fn supported_accelerators(&self) -> &[AcceleratorType] {
        &self.supported_accelerators
    }
    
    async fn load_model(
        &self,
        path: &Path,
        config: BackendConfig,
        accelerator: Option<AcceleratorType>,
    ) -> Result<Self::ModelHandle> {
        // 检查文件是否存在
        if !path.exists() {
            return Err(crate::api::error::ModelError::NotFound(
                format!("Model file not found: {}", path.display())
            ).into());
        }
        
        // 检查格式是否支持
        // TODO: 实际实现需要检查文件格式
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        
        if extension != "gguf" {
            return Err(crate::api::error::ModelError::UnsupportedFormat(
                format!("Unsupported format: {}. Only GGUF is supported.", extension)
            ).into());
        }
        
        // 转换配置
        let context_params = self.config_to_params(&config, accelerator);
        
        // 加载元数据
        let metadata = self.load_metadata(path).await?;
        
        tracing::info!(
            "Loading llama.cpp model from: {} (GPU layers: {}, Context: {})",
            path.display(),
            context_params.n_gpu_layers,
            context_params.n_ctx
        );
        
        // 实际的模型加载逻辑
        #[cfg(feature = "llama-cpp")]
        {
            // 使用 FFI 绑定加载模型
            let ffi = LlamaCppFFI::new();
            
            // 步骤 1: 加载模型
            let model_ptr = ffi.load_model(
                path,
                context_params.n_ctx as u32,
                context_params.n_gpu_layers,
            ).map_err(|e| crate::api::error::ModelError::NotFound(
                format!("Failed to load model: {}", e)
            ))?;
            
            // 步骤 2: 创建上下文参数
            let ctx_params = crate::inference::backends::llama_cpp_ffi::LlamaContextParams {
                n_ctx: context_params.n_ctx as u32,
                n_batch: context_params.n_batch as u32,
                n_threads: context_params.n_threads as u32,
                n_gpu_layers: context_params.n_gpu_layers,
                seed: 0xFFFFFFFF,
                f16_kv: true,
                logits_all: false,
                embedding: false,
                use_mmap: context_params.use_mmap,
                use_mlock: context_params.use_mlock,
            };
            
            // 步骤 3: 创建上下文
            let context_ptr = ffi.new_context(model_ptr, ctx_params)
                .map_err(|e| crate::api::error::ModelError::NotFound(
                    format!("Failed to create context: {}", e)
                ))?;
            
            tracing::info!("Model loaded successfully: {}", path.display());
            
            Ok(LlamaCppModelHandle {
                model_path: path.to_path_buf(),
                context_params,
                loaded: true,
                metadata: Some(metadata),
                model_ptr: Some(model_ptr),
                context_ptr: Some(context_ptr),
                ffi: Some(ffi),
            })
        }
        
        #[cfg(not(feature = "llama-cpp"))]
        {
            // 占位符实现（不使用 FFI）
            tracing::warn!("llama-cpp feature not enabled, using placeholder implementation");
            
            Ok(LlamaCppModelHandle {
                model_path: path.to_path_buf(),
                context_params,
                loaded: true,
                metadata: Some(metadata),
                model_instance: None,
                context_instance: None,
            })
        }
    }
    
    async fn unload_model(&self, handle: Self::ModelHandle) -> Result<()> {
        tracing::info!("Unloading llama.cpp model: {}", handle.model_path.display());
        
        #[cfg(feature = "llama-cpp")]
        {
            // 使用 FFI 释放资源
            if let Some(ffi) = handle.ffi {
                if let Some(ctx_ptr) = handle.context_ptr {
                    ffi.free_context(ctx_ptr);
                }
                if let Some(model_ptr) = handle.model_ptr {
                    ffi.free_model(model_ptr);
                }
            }
        }
        
        #[cfg(not(feature = "llama-cpp"))]
        {
            // 占位符：drop 会自动处理
            drop(handle.context_instance);
            drop(handle.model_instance);
        }
        
        tracing::debug!("Model unloaded successfully");
        Ok(())
    }
    
    async fn infer(
        &self,
        handle: &Self::ModelHandle,
        input: BackendInput,
        options: InferenceOptions,
    ) -> Result<BackendOutput> {
        if !handle.loaded {
            return Err(crate::api::error::ModelError::NotLoaded(
                "Model is not loaded".to_string()
            ).into());
        }
        
        match input {
            BackendInput::Text(prompt) => {
                let output = self.do_infer_text(handle, &prompt, &options).await?;
                Ok(BackendOutput::Text(output))
            }
            BackendInput::TextBatch(prompts) => {
                // 批处理推理
                let mut outputs = Vec::new();
                for prompt in prompts {
                    let output = self.do_infer_text(handle, &prompt, &options).await?;
                    outputs.push(output);
                }
                Ok(BackendOutput::TextBatch(outputs))
            }
            _ => {
                Err(crate::api::error::InferenceError::InvalidInput(
                    "llama.cpp backend only supports text input".to_string()
                ).into())
            }
        }
    }
    
    async fn infer_batch(
        &self,
        handle: &Self::ModelHandle,
        inputs: Vec<BackendInput>,
        options: InferenceOptions,
    ) -> Result<Vec<BackendOutput>> {
        let mut outputs = Vec::new();
        for input in inputs {
            let output = self.infer(handle, input, options.clone()).await?;
            outputs.push(output);
        }
        Ok(outputs)
    }
    
    async fn infer_stream(
        &self,
        handle: &Self::ModelHandle,
        input: BackendInput,
        options: InferenceOptions,
    ) -> Result<Option<Pin<Box<dyn Stream<Item = Result<BackendOutput>> + Send>>>> {
        // TODO: 实现流式推理
        // llama.cpp 支持流式输出，需要调用相应的 API
        
        // 目前返回 None，表示不支持流式推理
        let _ = (handle, input, options);
        Ok(None)
    }
    
    fn get_accelerator_info(&self, accelerator: AcceleratorType) -> Result<AcceleratorInfo> {
        match accelerator {
            AcceleratorType::Cpu => {
                Ok(AcceleratorInfo {
                    accelerator_type: accelerator,
                    device_name: "CPU".to_string(),
                    total_memory: None,
                    available_memory: None,
                    compute_capability: None,
                    available: true,
                    metadata: HashMap::new(),
                })
            }
            AcceleratorType::Cuda { device_id } => {
                // TODO: 实际实现需要查询 CUDA 设备信息
                Ok(AcceleratorInfo {
                    accelerator_type: accelerator,
                    device_name: format!("CUDA Device {}", device_id),
                    total_memory: None,
                    available_memory: None,
                    compute_capability: None,
                    available: true,  // 需要实际检测
                    metadata: HashMap::new(),
                })
            }
            AcceleratorType::Metal { device_id } => {
                // TODO: 实际实现需要查询 Metal 设备信息
                Ok(AcceleratorInfo {
                    accelerator_type: accelerator,
                    device_name: format!("Metal Device {}", device_id),
                    total_memory: None,
                    available_memory: None,
                    compute_capability: None,
                    available: true,  // 需要实际检测
                    metadata: HashMap::new(),
                })
            }
            _ => {
                Err(crate::api::error::InferenceError::BackendNotAvailable(
                    format!("Unsupported accelerator: {:?}", accelerator)
                ).into())
            }
        }
    }
    
    async fn get_model_metadata(
        &self,
        handle: &Self::ModelHandle,
    ) -> Result<ModelMetadata> {
        handle.metadata.clone()
            .ok_or_else(|| crate::api::error::ModelError::NotFound(
                "Model metadata not available".to_string()
            ).into())
    }
    
    fn is_model_loaded(&self, handle: &Self::ModelHandle) -> bool {
        #[cfg(feature = "llama-cpp")]
        {
            handle.loaded && handle.model_ptr.is_some() && handle.context_ptr.is_some()
        }
        
        #[cfg(not(feature = "llama-cpp"))]
        {
            handle.loaded
        }
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    
    fn create_test_model_file() -> (TempDir, PathBuf) {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test.gguf");
        // 创建一个假的模型文件（实际测试需要真实的 GGUF 文件）
        fs::write(&model_path, b"fake gguf model data").unwrap();
        (temp_dir, model_path)
    }
    
    #[tokio::test]
    async fn test_llama_cpp_backend_creation() {
        let backend = LlamaCppBackend::new();
        assert_eq!(backend.name(), "llama.cpp");
        assert_eq!(backend.backend_type(), BackendType::LlamaCpp);
    }
    
    #[tokio::test]
    async fn test_supported_formats() {
        let backend = LlamaCppBackend::new();
        assert!(backend.supports_format(ModelFormat::Gguf));
        assert!(!backend.supports_format(ModelFormat::Safetensors));
    }
    
    #[tokio::test]
    async fn test_load_model() {
        let backend = LlamaCppBackend::new();
        let (_temp_dir, model_path) = create_test_model_file();
        
        let config = BackendConfig::default();
        let handle = backend.load_model(&model_path, config, None).await;
        
        // 由于是占位符实现，应该成功加载
        assert!(handle.is_ok());
        let handle = handle.unwrap();
        assert!(handle.loaded);
        assert_eq!(handle.model_path, model_path);
    }
    
    #[tokio::test]
    async fn test_load_nonexistent_model() {
        let backend = LlamaCppBackend::new();
        let config = BackendConfig::default();
        let path = Path::new("/nonexistent/model.gguf");
        
        let result = backend.load_model(path, config, None).await;
        assert!(result.is_err());
    }
}


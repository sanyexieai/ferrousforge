//! llama.cpp FFI 绑定
//! 
//! 提供对 llama.cpp C API 的 Rust 绑定。
//! 
//! 支持两种模式：
//! 1. 静态链接：直接链接 llama.cpp 库（通过 build.rs 配置）
//! 2. 动态加载：使用 libloading 动态加载库（需要 llama-cpp-dynamic feature）

use std::ffi::{CString, CStr};
use std::os::raw::{c_int, c_char, c_void};
use std::path::Path;

/// llama.cpp 模型句柄（C 指针）
pub type LlamaModelPtr = *mut c_void;

/// llama.cpp 上下文句柄（C 指针）
pub type LlamaContextPtr = *mut c_void;

/// llama.cpp 参数结构（简化版，匹配 llama.cpp 的 C API）
#[repr(C)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub mul_mat_q: bool,
    pub f16_kv: bool,
    pub logits_all: bool,
    pub embedding: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub seed: u32,
}

impl Default for LlamaContextParams {
    fn default() -> Self {
        Self {
            n_ctx: 2048,
            n_batch: 512,
            n_threads: num_cpus::get() as u32,
            n_threads_batch: num_cpus::get() as u32,
            n_gpu_layers: 0,
            main_gpu: 0,
            tensor_split: std::ptr::null(),
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
            mul_mat_q: true,
            f16_kv: true,
            logits_all: false,
            embedding: false,
            use_mmap: true,
            use_mlock: false,
            seed: 0xFFFFFFFF,
        }
    }
}

/// llama.cpp 模型参数结构
#[repr(C)]
pub struct LlamaModelParams {
    pub n_gpu_layers: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub use_mmap: bool,
    pub use_mlock: bool,
}

impl Default for LlamaModelParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            main_gpu: 0,
            tensor_split: std::ptr::null(),
            use_mmap: true,
            use_mlock: false,
        }
    }
}

// ========== 静态链接模式：直接声明 C 函数 ==========

#[cfg(all(feature = "llama-cpp", not(feature = "llama-cpp-dynamic")))]
extern "C" {
    // llama_model 相关函数
    fn llama_model_load_from_file(
        path: *const c_char,
        params: LlamaModelParams,
    ) -> LlamaModelPtr;
    
    fn llama_model_free(model: LlamaModelPtr);
    
    // llama_context 相关函数
    fn llama_new_context_with_model(
        model: LlamaModelPtr,
        params: LlamaContextParams,
    ) -> LlamaContextPtr;
    
    fn llama_context_free(ctx: LlamaContextPtr);
    
    // Tokenization
    fn llama_tokenize(
        ctx: LlamaContextPtr,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut i32,
        n_max_tokens: c_int,
        add_bos: bool,
        special: bool,
    ) -> c_int;
    
    // Evaluation
    fn llama_eval(
        ctx: LlamaContextPtr,
        tokens: *const i32,
        n_tokens: c_int,
        n_past: c_int,
        n_threads: c_int,
    ) -> c_int;
    
    // Sampling
    fn llama_sample_top_p_top_k(
        ctx: LlamaContextPtr,
        last_n_tokens: *const i32,
        last_n_tokens_size: c_int,
        top_k: c_int,
        top_p: f32,
        temp: f32,
        repeat_penalty: f32,
    ) -> i32;
    
    // Token to string
    fn llama_token_to_str(ctx: LlamaContextPtr, token: i32) -> *const c_char;
    
    // Special tokens
    fn llama_token_eos() -> i32;
    fn llama_token_bos() -> i32;
}

// ========== 动态加载模式：使用 libloading ==========

#[cfg(feature = "llama-cpp-dynamic")]
use libloading::{Library, Symbol};

#[cfg(feature = "llama-cpp-dynamic")]
struct LlamaCppLib {
    lib: Library,
}

#[cfg(feature = "llama-cpp-dynamic")]
impl LlamaCppLib {
    fn load() -> Result<Self, String> {
        // 尝试加载 llama.cpp 库
        let lib_names = if cfg!(target_os = "windows") {
            vec!["llama.dll", "libllama.dll"]
        } else if cfg!(target_os = "macos") {
            vec!["libllama.dylib", "llama.dylib"]
        } else {
            vec!["libllama.so", "llama.so"]
        };
        
        let mut last_error = None;
        for name in lib_names {
            match Library::new(name) {
                Ok(lib) => {
                    tracing::info!("Loaded llama.cpp library: {}", name);
                    return Ok(Self { lib });
                }
                Err(e) => {
                    last_error = Some(format!("Failed to load {}: {}", name, e));
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| "No llama.cpp library found".to_string()))
    }
    
    unsafe fn get_symbol<T>(&self, name: &[u8]) -> Result<Symbol<T>, String> {
        self.lib.get(name)
            .map_err(|e| format!("Failed to get symbol {}: {}", String::from_utf8_lossy(name), e))
    }
}

// ========== FFI 包装器 ==========

/// llama.cpp FFI 包装器
/// 
/// 这个结构提供了对 llama.cpp C API 的封装。
/// 支持静态链接和动态加载两种模式。
pub struct LlamaCppFFI {
    #[cfg(feature = "llama-cpp-dynamic")]
    lib: Option<LlamaCppLib>,
}

impl LlamaCppFFI {
    /// 创建新的 FFI 包装器
    pub fn new() -> Self {
        #[cfg(feature = "llama-cpp-dynamic")]
        {
            let lib = LlamaCppLib::load().ok();
            Self { lib }
        }
        
        #[cfg(not(feature = "llama-cpp-dynamic"))]
        {
            Self {}
        }
    }
    
    /// 检查库是否可用
    pub fn is_available(&self) -> bool {
        #[cfg(feature = "llama-cpp-dynamic")]
        {
            self.lib.is_some()
        }
        
        #[cfg(all(feature = "llama-cpp", not(feature = "llama-cpp-dynamic")))]
        {
            true  // 静态链接时总是可用
        }
        
        #[cfg(not(feature = "llama-cpp"))]
        {
            false
        }
    }
    
    /// 加载模型
    pub fn load_model(
        &self,
        model_path: &Path,
        n_ctx: u32,
        n_gpu_layers: i32,
    ) -> Result<LlamaModelPtr, String> {
        let path_str = model_path.to_str()
            .ok_or_else(|| "Invalid model path".to_string())?;
        
        let c_path = CString::new(path_str)
            .map_err(|e| format!("Failed to create C string: {}", e))?;
        
        let params = LlamaModelParams {
            n_gpu_layers,
            main_gpu: 0,
            tensor_split: std::ptr::null(),
            use_mmap: true,
            use_mlock: false,
        };
        
        #[cfg(all(feature = "llama-cpp", not(feature = "llama-cpp-dynamic")))]
        {
            // 静态链接：直接调用
            let model = unsafe {
                llama_model_load_from_file(c_path.as_ptr(), params)
            };
            
            if model.is_null() {
                return Err("Failed to load model".to_string());
            }
            
            tracing::info!("Model loaded successfully: {}", path_str);
            Ok(model)
        }
        
        #[cfg(feature = "llama-cpp-dynamic")]
        {
            // 动态加载：通过 libloading 调用
            if let Some(ref lib) = self.lib {
                unsafe {
                    let func: Symbol<unsafe extern "C" fn(*const c_char, LlamaModelParams) -> LlamaModelPtr> =
                        lib.get_symbol(b"llama_model_load_from_file")?;
                    
                    let model = func(c_path.as_ptr(), params);
                    
                    if model.is_null() {
                        return Err("Failed to load model".to_string());
                    }
                    
                    tracing::info!("Model loaded successfully: {}", path_str);
                    Ok(model)
                }
            } else {
                Err("llama.cpp library not loaded".to_string())
            }
        }
        
        #[cfg(not(any(feature = "llama-cpp", feature = "llama-cpp-dynamic")))]
        {
            // 占位符模式
            tracing::warn!("llama-cpp feature not enabled, using placeholder");
            Ok(std::ptr::null_mut())
        }
    }
    
    /// 创建上下文
    pub fn new_context(
        &self,
        model: LlamaModelPtr,
        params: LlamaContextParams,
    ) -> Result<LlamaContextPtr, String> {
        if model.is_null() {
            return Err("Model pointer is null".to_string());
        }
        
        #[cfg(all(feature = "llama-cpp", not(feature = "llama-cpp-dynamic")))]
        {
            // 静态链接
            let ctx = unsafe {
                llama_new_context_with_model(model, params)
            };
            
            if ctx.is_null() {
                return Err("Failed to create context".to_string());
            }
            
            Ok(ctx)
        }
        
        #[cfg(feature = "llama-cpp-dynamic")]
        {
            // 动态加载
            if let Some(ref lib) = self.lib {
                unsafe {
                    let func: Symbol<unsafe extern "C" fn(LlamaModelPtr, LlamaContextParams) -> LlamaContextPtr> =
                        lib.get_symbol(b"llama_new_context_with_model")?;
                    
                    let ctx = func(model, params);
                    
                    if ctx.is_null() {
                        return Err("Failed to create context".to_string());
                    }
                    
                    Ok(ctx)
                }
            } else {
                Err("llama.cpp library not loaded".to_string())
            }
        }
        
        #[cfg(not(any(feature = "llama-cpp", feature = "llama-cpp-dynamic")))]
        {
            // 占位符
            Ok(std::ptr::null_mut())
        }
    }
    
    /// Tokenize 文本
    pub fn tokenize(
        &self,
        ctx: LlamaContextPtr,
        text: &str,
        add_bos: bool,
    ) -> Result<Vec<i32>, String> {
        if ctx.is_null() {
            return Err("Context pointer is null".to_string());
        }
        
        let c_text = CString::new(text)
            .map_err(|e| format!("Failed to create C string: {}", e))?;
        
        let text_len = text.len() as c_int;
        let n_max_tokens = (text_len * 2).max(512);  // 估算最大 tokens
        let mut tokens = vec![0i32; n_max_tokens as usize];
        
        #[cfg(all(feature = "llama-cpp", not(feature = "llama-cpp-dynamic")))]
        {
            // 静态链接
            let n_tokens = unsafe {
                llama_tokenize(
                    ctx,
                    c_text.as_ptr(),
                    text_len,
                    tokens.as_mut_ptr(),
                    n_max_tokens,
                    add_bos,
                    false,  // special
                )
            };
            
            if n_tokens < 0 {
                return Err("Tokenization failed".to_string());
            }
            
            tokens.truncate(n_tokens as usize);
            Ok(tokens)
        }
        
        #[cfg(feature = "llama-cpp-dynamic")]
        {
            // 动态加载
            if let Some(ref lib) = self.lib {
                unsafe {
                    let func: Symbol<unsafe extern "C" fn(
                        LlamaContextPtr,
                        *const c_char,
                        c_int,
                        *mut i32,
                        c_int,
                        bool,
                        bool,
                    ) -> c_int> = lib.get_symbol(b"llama_tokenize")?;
                    
                    let n_tokens = func(
                        ctx,
                        c_text.as_ptr(),
                        text_len,
                        tokens.as_mut_ptr(),
                        n_max_tokens,
                        add_bos,
                        false,
                    );
                    
                    if n_tokens < 0 {
                        return Err("Tokenization failed".to_string());
                    }
                    
                    tokens.truncate(n_tokens as usize);
                    Ok(tokens)
                }
            } else {
                Err("llama.cpp library not loaded".to_string())
            }
        }
        
        #[cfg(not(any(feature = "llama-cpp", feature = "llama-cpp-dynamic")))]
        {
            // 占位符：返回空 tokens
            Ok(vec![])
        }
    }
    
    /// 执行推理（eval）
    pub fn eval(
        &self,
        ctx: LlamaContextPtr,
        tokens: &[i32],
        n_past: i32,
        n_threads: i32,
    ) -> Result<(), String> {
        if ctx.is_null() {
            return Err("Context pointer is null".to_string());
        }
        
        if tokens.is_empty() {
            return Ok(());  // 空 tokens，无需 eval
        }
        
        #[cfg(all(feature = "llama-cpp", not(feature = "llama-cpp-dynamic")))]
        {
            // 静态链接
            let result = unsafe {
                llama_eval(
                    ctx,
                    tokens.as_ptr(),
                    tokens.len() as c_int,
                    n_past,
                    n_threads,
                )
            };
            
            if result != 0 {
                return Err(format!("Evaluation failed with code: {}", result));
            }
            
            Ok(())
        }
        
        #[cfg(feature = "llama-cpp-dynamic")]
        {
            // 动态加载
            if let Some(ref lib) = self.lib {
                unsafe {
                    let func: Symbol<unsafe extern "C" fn(
                        LlamaContextPtr,
                        *const i32,
                        c_int,
                        c_int,
                        c_int,
                    ) -> c_int> = lib.get_symbol(b"llama_eval")?;
                    
                    let result = func(
                        ctx,
                        tokens.as_ptr(),
                        tokens.len() as c_int,
                        n_past,
                        n_threads,
                    );
                    
                    if result != 0 {
                        return Err(format!("Evaluation failed with code: {}", result));
                    }
                    
                    Ok(())
                }
            } else {
                Err("llama.cpp library not loaded".to_string())
            }
        }
        
        #[cfg(not(any(feature = "llama-cpp", feature = "llama-cpp-dynamic")))]
        {
            // 占位符
            Ok(())
        }
    }
    
    /// 采样下一个 token
    pub fn sample(
        &self,
        ctx: LlamaContextPtr,
        temperature: f32,
        top_p: f32,
        top_k: i32,
    ) -> Result<i32, String> {
        if ctx.is_null() {
            return Err("Context pointer is null".to_string());
        }
        
        #[cfg(all(feature = "llama-cpp", not(feature = "llama-cpp-dynamic")))]
        {
            // 静态链接
            // 注意：这里使用简化的采样函数，实际可能需要更复杂的实现
            let last_n_tokens = std::ptr::null();
            let last_n_tokens_size = 0;
            let repeat_penalty = 1.1;
            
            let token = unsafe {
                llama_sample_top_p_top_k(
                    ctx,
                    last_n_tokens,
                    last_n_tokens_size,
                    top_k,
                    top_p,
                    temperature,
                    repeat_penalty,
                )
            };
            
            Ok(token)
        }
        
        #[cfg(feature = "llama-cpp-dynamic")]
        {
            // 动态加载
            if let Some(ref lib) = self.lib {
                unsafe {
                    let func: Symbol<unsafe extern "C" fn(
                        LlamaContextPtr,
                        *const i32,
                        c_int,
                        c_int,
                        f32,
                        f32,
                        f32,
                    ) -> i32> = lib.get_symbol(b"llama_sample_top_p_top_k")?;
                    
                    let last_n_tokens = std::ptr::null();
                    let last_n_tokens_size = 0;
                    let repeat_penalty = 1.1;
                    
                    let token = func(
                        ctx,
                        last_n_tokens,
                        last_n_tokens_size,
                        top_k,
                        top_p,
                        temperature,
                        repeat_penalty,
                    );
                    
                    Ok(token)
                }
            } else {
                Err("llama.cpp library not loaded".to_string())
            }
        }
        
        #[cfg(not(any(feature = "llama-cpp", feature = "llama-cpp-dynamic")))]
        {
            // 占位符
            Ok(0)
        }
    }
    
    /// 将 token 转换为字符串
    pub fn token_to_str(
        &self,
        ctx: LlamaContextPtr,
        token: i32,
    ) -> Result<String, String> {
        if ctx.is_null() {
            return Err("Context pointer is null".to_string());
        }
        
        #[cfg(all(feature = "llama-cpp", not(feature = "llama-cpp-dynamic")))]
        {
            // 静态链接
            let c_str = unsafe {
                llama_token_to_str(ctx, token)
            };
            
            if c_str.is_null() {
                return Err("Token to string conversion failed".to_string());
            }
            
            unsafe {
                Ok(CStr::from_ptr(c_str).to_string_lossy().into_owned())
            }
        }
        
        #[cfg(feature = "llama-cpp-dynamic")]
        {
            // 动态加载
            if let Some(ref lib) = self.lib {
                unsafe {
                    let func: Symbol<unsafe extern "C" fn(LlamaContextPtr, i32) -> *const c_char> =
                        lib.get_symbol(b"llama_token_to_str")?;
                    
                    let c_str = func(ctx, token);
                    
                    if c_str.is_null() {
                        return Err("Token to string conversion failed".to_string());
                    }
                    
                    Ok(CStr::from_ptr(c_str).to_string_lossy().into_owned())
                }
            } else {
                Err("llama.cpp library not loaded".to_string())
            }
        }
        
        #[cfg(not(any(feature = "llama-cpp", feature = "llama-cpp-dynamic")))]
        {
            // 占位符
            Ok(String::new())
        }
    }
    
    /// 获取 EOS token
    pub fn token_eos(&self) -> i32 {
        #[cfg(all(feature = "llama-cpp", not(feature = "llama-cpp-dynamic")))]
        {
            unsafe { llama_token_eos() }
        }
        
        #[cfg(feature = "llama-cpp-dynamic")]
        {
            if let Some(ref lib) = self.lib {
                unsafe {
                    if let Ok(func) = lib.get_symbol::<unsafe extern "C" fn() -> i32>(b"llama_token_eos") {
                        func()
                    } else {
                        2  // 默认 EOS token
                    }
                }
            } else {
                2
            }
        }
        
        #[cfg(not(any(feature = "llama-cpp", feature = "llama-cpp-dynamic")))]
        {
            2  // 默认 EOS token
        }
    }
    
    /// 释放模型
    pub fn free_model(&self, model: LlamaModelPtr) {
        if model.is_null() {
            return;
        }
        
        #[cfg(all(feature = "llama-cpp", not(feature = "llama-cpp-dynamic")))]
        {
            unsafe {
                llama_model_free(model);
            }
        }
        
        #[cfg(feature = "llama-cpp-dynamic")]
        {
            if let Some(ref lib) = self.lib {
                unsafe {
                    if let Ok(func) = lib.get_symbol::<unsafe extern "C" fn(LlamaModelPtr)>(b"llama_model_free") {
                        func(model);
                    }
                }
            }
        }
    }
    
    /// 释放上下文
    pub fn free_context(&self, ctx: LlamaContextPtr) {
        if ctx.is_null() {
            return;
        }
        
        #[cfg(all(feature = "llama-cpp", not(feature = "llama-cpp-dynamic")))]
        {
            unsafe {
                llama_context_free(ctx);
            }
        }
        
        #[cfg(feature = "llama-cpp-dynamic")]
        {
            if let Some(ref lib) = self.lib {
                unsafe {
                    if let Ok(func) = lib.get_symbol::<unsafe extern "C" fn(LlamaContextPtr)>(b"llama_context_free") {
                        func(ctx);
                    }
                }
            }
        }
    }
}

impl Default for LlamaCppFFI {
    fn default() -> Self {
        Self::new()
    }
}

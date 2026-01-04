//! llama.cpp 后端使用示例
//! 
//! 运行示例：
//! ```bash
//! cargo run --example llama_cpp_example --features llama-cpp -- --model path/to/model.gguf
//! ```

use clap::Parser;
use ferrousforge::inference::backends::llama_cpp::LlamaCppBackend;
use ferrousforge::inference::backend::{InferenceBackend, BackendConfig, BackendInput};
use ferrousforge::api::request::InferenceOptions;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// 模型文件路径
    #[arg(short, long)]
    model: PathBuf,
    
    /// 提示文本
    #[arg(short, long, default_value = "Hello, how are you?")]
    prompt: String,
    
    /// 上下文大小
    #[arg(long, default_value_t = 2048)]
    context_size: usize,
    
    /// 最大生成 tokens
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,
    
    /// 温度
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    tracing_subscriber::fmt::init();
    
    let args = Args::parse();
    
    // 检查模型文件
    if !args.model.exists() {
        eprintln!("Error: Model file not found: {}", args.model.display());
        eprintln!("\nPlease provide a valid GGUF model file path.");
        eprintln!("Example: cargo run --example llama_cpp_example --features llama-cpp -- --model model.gguf");
        std::process::exit(1);
    }
    
    println!("Initializing llama.cpp backend...");
    let backend = LlamaCppBackend::new();
    
    // 检查后端是否可用
    #[cfg(feature = "llama-cpp")]
    {
        use ferrousforge::inference::backends::llama_cpp_ffi::LlamaCppFFI;
        let ffi = LlamaCppFFI::new();
        if !ffi.is_available() {
            eprintln!("Warning: llama.cpp library may not be available");
            eprintln!("Make sure llama.cpp is installed and linked correctly");
        }
    }
    
    // 配置后端
    let config = BackendConfig {
        context_size: Some(args.context_size),
        num_threads: Some(num_cpus::get()),
        ..Default::default()
    };
    
    println!("Loading model: {}", args.model.display());
    let handle = match backend.load_model(&args.model, config, None).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            eprintln!("\nTroubleshooting:");
            eprintln!("1. Ensure the model file is a valid GGUF format");
            eprintln!("2. Check that llama.cpp library is installed");
            eprintln!("3. Try using --features llama-cpp-dynamic for dynamic loading");
            std::process::exit(1);
        }
    };
    
    println!("Model loaded successfully!");
    if let Some(ref metadata) = handle.metadata {
        println!("Model: {}", metadata.name);
        println!("Type: {:?}", metadata.model_type);
        if let Some(size) = metadata.parameters {
            println!("Parameters: {}B", size);
        }
    }
    
    // 执行推理
    println!("\nGenerating response...");
    println!("Prompt: {}", args.prompt);
    
    let input = BackendInput::Text(args.prompt);
    let options = InferenceOptions {
        temperature: Some(args.temperature),
        max_tokens: Some(args.max_tokens),
        ..Default::default()
    };
    
    match backend.infer(&handle, input, options).await {
        Ok(output) => {
            match output {
                ferrousforge::inference::backend::BackendOutput::Text(text) => {
                    println!("\nResponse:");
                    println!("{}", text);
                }
                _ => {
                    println!("Unexpected output type");
                }
            }
        }
        Err(e) => {
            eprintln!("Inference failed: {}", e);
            eprintln!("\nNote: If using placeholder implementation, this is expected.");
            eprintln!("Enable llama-cpp feature to use actual llama.cpp library.");
        }
    }
    
    // 清理
    println!("\nUnloading model...");
    backend.unload_model(handle).await?;
    println!("Done!");
    
    Ok(())
}


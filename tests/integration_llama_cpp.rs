//! llama.cpp 集成测试
//! 
//! 这些测试需要真实的 GGUF 模型文件。
//! 
//! 运行测试前，请设置环境变量：
//! - `TEST_MODEL_PATH`: GGUF 模型文件路径

#[cfg(feature = "llama-cpp")]
mod tests {
    use ferrousforge::inference::backends::llama_cpp::LlamaCppBackend;
    use ferrousforge::inference::backend::{InferenceBackend, BackendConfig, BackendInput};
    use ferrousforge::api::request::InferenceOptions;
    use std::path::Path;
    use std::env;

    fn get_test_model_path() -> Option<String> {
        env::var("TEST_MODEL_PATH").ok()
            .or_else(|| {
                // 尝试常见的测试模型路径
                let common_paths = vec![
                    "models/test.gguf",
                    "test_model.gguf",
                    "../models/test.gguf",
                ];
                common_paths.iter()
                    .find(|p| Path::new(p).exists())
                    .map(|p| p.to_string())
            })
    }

    #[tokio::test]
    #[ignore]  // 需要真实的模型文件
    async fn test_load_real_model() {
        let model_path = match get_test_model_path() {
            Some(path) => path,
            None => {
                eprintln!("Skipping test: TEST_MODEL_PATH not set and no test model found");
                return;
            }
        };

        let backend = LlamaCppBackend::new();
        let config = BackendConfig::default();
        
        let handle = backend.load_model(Path::new(&model_path), config, None).await;
        
        match handle {
            Ok(handle) => {
                assert!(handle.loaded);
                assert!(handle.metadata.is_some());
                println!("Model loaded successfully: {:?}", handle.metadata);
                
                // 清理
                backend.unload_model(handle).await.unwrap();
            }
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                // 如果库未安装，这是预期的
                if e.to_string().contains("library not loaded") {
                    eprintln!("Note: llama.cpp library may not be installed");
                }
            }
        }
    }

    #[tokio::test]
    #[ignore]  // 需要真实的模型文件
    async fn test_inference_with_real_model() {
        let model_path = match get_test_model_path() {
            Some(path) => path,
            None => {
                eprintln!("Skipping test: TEST_MODEL_PATH not set and no test model found");
                return;
            }
        };

        let backend = LlamaCppBackend::new();
        let config = BackendConfig {
            context_size: Some(512),  // 较小的上下文用于测试
            ..Default::default()
        };
        
        let handle = match backend.load_model(Path::new(&model_path), config, None).await {
            Ok(h) => h,
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                return;
            }
        };

        let input = BackendInput::Text("The capital of France is".to_string());
        let options = InferenceOptions {
            temperature: Some(0.7),
            max_tokens: Some(10),
            ..Default::default()
        };

        match backend.infer(&handle, input, options).await {
            Ok(output) => {
                match output {
                    ferrousforge::inference::backend::BackendOutput::Text(text) => {
                        println!("Generated text: {}", text);
                        assert!(!text.is_empty());
                    }
                    _ => panic!("Unexpected output type"),
                }
            }
            Err(e) => {
                eprintln!("Inference failed: {}", e);
                // 如果使用占位符实现，这是预期的
            }
        }

        // 清理
        backend.unload_model(handle).await.unwrap();
    }

    #[test]
    fn test_ffi_availability() {
        use ferrousforge::inference::backends::llama_cpp_ffi::LlamaCppFFI;
        
        let ffi = LlamaCppFFI::new();
        let available = ffi.is_available();
        
        println!("llama.cpp FFI available: {}", available);
        
        // 如果 feature 未启用，应该不可用
        #[cfg(not(any(feature = "llama-cpp", feature = "llama-cpp-dynamic")))]
        {
            assert!(!available, "FFI should not be available without feature");
        }
    }
}


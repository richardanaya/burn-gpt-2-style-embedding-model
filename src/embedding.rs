use crate::model::load_model;
use crate::{Gpt2Config, Gpt2Model, Gpt2Tokenizer};
use anyhow::Result;
use burn::prelude::*;
use std::path::PathBuf;

/// Get embedding for a sentence
///
/// This function converts a text sentence into a dense vector representation (embedding)
/// that captures its semantic meaning. The embedding can be used for similarity comparison,
/// clustering, classification, and other downstream NLP tasks.
///
/// ## Process Overview
/// 1. **Model Loading**: Loads trained model or uses random weights for demo
/// 2. **Tokenization**: Converts text to token IDs using GPT-2 tokenizer
/// 3. **Forward Pass**: Processes through 12 transformer layers
/// 4. **Mean Pooling**: Averages token embeddings to get sentence-level representation
/// 5. **Output**: Returns embedding as JSON or raw format
///
/// ## Parameters
/// - `model_path`: Optional path to trained model (.mpk file). If None, uses random weights
/// - `sentence`: Input text to embed
/// - `format`: Output format ("json" for structured output, "raw" for simple display)
/// - Model architecture parameters (n_heads, n_layers, d_model, context_size)
/// - `device`: WebGPU device for computation
///
/// ## Output Formats
/// - **JSON**: Structured output with sentence, embedding vector, and dimensions
/// - **Raw**: Human-readable format showing sentence, dimensions, and vector values
pub async fn embed_sentence<B: Backend<FloatElem = f32>>(
    model_path: Option<&PathBuf>,
    sentence: &str,
    format: &str,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    context_size: usize,
    device: B::Device,
) -> Result<()> {
    let config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: context_size,
        d_model,
        n_heads,
        n_layers,
        dropout: 0.1,
        margin: 1.0,
    };
    let model = if let Some(path) = model_path {
        println!("Loading model from: {}", path.display());
        load_model::<B>(config, path, &device)?
    } else {
        println!("Using randomly initialized model for demo");
        Gpt2Model::new(config, &device)
    };

    let tokenizer = Gpt2Tokenizer::new_simple(context_size)?;

    // Get embedding - for now using regular method to avoid tensor dimension issues
    // TODO: Switch back to masked version once tensor dimension issues are resolved
    let token_ids = tokenizer.encode(sentence, true)?;
    let input_tensor = Tensor::<B, 1, Int>::from_data(
        TensorData::from(&token_ids.iter().map(|&x| x as i64).collect::<Vec<_>>()[..]),
        &device,
    )
    .unsqueeze_dim(0);

    let embedding = model.get_sentence_embedding(input_tensor);
    // embedding is now [batch_size, d_model] - already the correct shape
    let embedding_data = embedding.into_data().to_vec::<f32>().unwrap();

    match format {
        "json" => {
            let json_output = serde_json::json!({
                "sentence": sentence,
                "embedding": embedding_data,
                "dimensions": embedding_data.len()
            });
            println!("{}", serde_json::to_string_pretty(&json_output)?);
        }
        "raw" => {
            println!("Sentence: \"{}\"", sentence);
            println!("Embedding dimensions: {}", embedding_data.len());
            println!("Vector: {:?}", embedding_data);
        }
        _ => {
            eprintln!("Unknown format: {}. Use 'json' or 'raw'", format);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Generate embeddings for multiple sentences efficiently
///
/// This function is useful for batch processing multiple sentences at once,
/// which can be more efficient than processing them individually.
pub async fn embed_sentences<B: Backend<FloatElem = f32>>(
    model_path: Option<&PathBuf>,
    sentences: &[String],
    format: &str,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    context_size: usize,
    device: B::Device,
) -> Result<Vec<Vec<f32>>> {
    let config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: context_size,
        d_model,
        n_heads,
        n_layers,
        dropout: 0.1,
        margin: 1.0,
    };

    let model = if let Some(path) = model_path {
        load_model::<B>(config, path, &device)?
    } else {
        Gpt2Model::new(config, &device)
    };

    let tokenizer = Gpt2Tokenizer::new_simple(context_size)?;
    let mut embeddings = Vec::new();

    for sentence in sentences {
        // TODO: Switch back to masked version once tensor dimension issues are resolved
        let token_ids = tokenizer.encode(sentence, true)?;
        let input_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::from(&token_ids.iter().map(|&x| x as i64).collect::<Vec<_>>()[..]),
            &device,
        )
        .unsqueeze_dim(0);

        let embedding = model.get_sentence_embedding(input_tensor);
        let embedding_data = embedding.into_data().to_vec::<f32>().unwrap();

        embeddings.push(embedding_data.clone());

        // Print individual results based on format
        match format {
            "json" => {
                let json_output = serde_json::json!({
                    "sentence": sentence,
                    "embedding": embedding_data,
                    "dimensions": embedding_data.len()
                });
                println!("{}", serde_json::to_string_pretty(&json_output)?);
            }
            "raw" => {
                println!("Sentence: \"{}\"", sentence);
                println!("Embedding dimensions: {}", embedding_data.len());
                println!("Vector: {:?}", embedding_data);
                println!("---");
            }
            _ => {
                eprintln!("Unknown format: {}. Use 'json' or 'raw'", format);
                std::process::exit(1);
            }
        }
    }

    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::Wgpu;

    type TestBackend = Wgpu;

    #[tokio::test]
    async fn test_embed_sentence_with_random_model() {
        let device = Default::default();
        let sentence = "Hello world";

        // Test with random model (no model path)
        let result = embed_sentence::<TestBackend>(
            None, sentence, "raw", 12,   // n_heads
            12,   // n_layers
            768,  // d_model
            1024, // context_size
            device,
        )
        .await;

        assert!(
            result.is_ok(),
            "embed_sentence should succeed with random model"
        );
    }

    #[tokio::test]
    async fn test_embed_multiple_sentences() {
        let device = Default::default();
        let sentences = vec!["Hello world".to_string(), "How are you?".to_string()];

        let result = embed_sentences::<TestBackend>(
            None, &sentences, "raw", 12,   // n_heads
            12,   // n_layers
            768,  // d_model
            1024, // context_size
            device,
        )
        .await;

        assert!(result.is_ok(), "embed_sentences should succeed");
        let embeddings = result.unwrap();
        assert_eq!(
            embeddings.len(),
            2,
            "Should return embeddings for both sentences"
        );
        assert_eq!(
            embeddings[0].len(),
            768,
            "Each embedding should have d_model dimensions"
        );
        assert_eq!(
            embeddings[1].len(),
            768,
            "Each embedding should have d_model dimensions"
        );
    }
}

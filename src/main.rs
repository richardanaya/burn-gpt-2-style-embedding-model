use anyhow::Result;
use burn::prelude::*;
use burn_gpt2_embedding_model::{
    create_demo_tokenizer, load_model, Gpt2Config, Gpt2Model,
    SimilarityCalculator,
};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

type Backend = burn::backend::wgpu::Wgpu;

/// GPT-2 Embedding Model CLI
/// 
/// A command-line interface for training and using a GPT-2 style embedding model
/// built with the Burn deep learning framework and WebGPU backend.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Get vector embedding for a sentence
    Embed {
        /// Path to the trained model file (optional - will use random weights if not provided)
        #[arg(short, long)]
        model: Option<PathBuf>,
        
        /// Sentence to get embedding for
        #[arg(short, long)]
        sentence: String,
        
        /// Output format: json or raw
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    
    /// Calculate similarity between two sentences
    Similarity {
        /// Path to the trained model file (optional - will use random weights if not provided)
        #[arg(short, long)]
        model: Option<PathBuf>,
        
        /// First sentence
        #[arg(long)]
        sentence1: String,
        
        /// Second sentence
        #[arg(long)]
        sentence2: String,
        
        /// Show all similarity metrics (not just cosine)
        #[arg(long)]
        all_metrics: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Initialize WebGPU device
    let device = burn::backend::wgpu::WgpuDevice::default();
    
    match &cli.command {
        Commands::Embed {
            model,
            sentence,
            format,
        } => embed_sentence(model.as_ref(), sentence, format, device).await,
        
        Commands::Similarity {
            model,
            sentence1,
            sentence2,
            all_metrics,
        } => calculate_similarity(model.as_ref(), sentence1, sentence2, *all_metrics, device).await,
    }
}

/// Get embedding for a sentence
async fn embed_sentence(
    model_path: Option<&PathBuf>,
    sentence: &str,
    format: &str,
    device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    let config = Gpt2Config::default();
    let model = if let Some(path) = model_path {
        println!("Loading model from: {}", path.display());
        load_model::<Backend>(config, path, &device)?
    } else {
        println!("Using randomly initialized model for demo");
        Gpt2Model::new(config, &device)
    };
    
    let tokenizer = create_demo_tokenizer()?;
    
    // Get embedding
    let token_ids = tokenizer.encode(sentence, true)?;
    let input_tensor = Tensor::<Backend, 1, Int>::from_data(
        TensorData::from(&token_ids.iter().map(|&x| x as i64).collect::<Vec<_>>()[..]),
        &device,
    ).unsqueeze_dim(0);
    
    let embedding = model.get_sentence_embedding(input_tensor);
    let embedding: Tensor<Backend, 2> = embedding.squeeze_dims(&[1]); // Remove middle dimension to get [batch_size, d_model]
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

/// Calculate similarity between two sentences
async fn calculate_similarity(
    model_path: Option<&PathBuf>,
    sentence1: &str,
    sentence2: &str,
    all_metrics: bool,
    device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    let config = Gpt2Config::default();
    let model = if let Some(path) = model_path {
        println!("Loading model from: {}", path.display());
        load_model::<Backend>(config, path, &device)?
    } else {
        println!("Using randomly initialized model for demo");
        Gpt2Model::new(config, &device)
    };
    
    let tokenizer = create_demo_tokenizer()?;
    
    // Create similarity calculator
    let calculator = SimilarityCalculator::new(model, tokenizer);
    
    if all_metrics {
        // Calculate all similarity metrics
        let metrics = calculator.calculate_all_metrics(sentence1, sentence2)?;
        metrics.print_formatted(sentence1, sentence2);
    } else {
        // Calculate just cosine similarity
        let similarity = calculator.calculate_similarity(sentence1, sentence2)?;
        println!("Sentences:");
        println!("  1: \"{}\"", sentence1);
        println!("  2: \"{}\"", sentence2);
        println!("Cosine Similarity: {:.4} (0=different, 1=identical)", similarity);
    }
    
    Ok(())
}
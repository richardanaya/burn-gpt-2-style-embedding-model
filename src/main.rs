use anyhow::Result;
use burn::prelude::*;
use burn_gpt_n_embedding_model::{
    create_demo_tokenizer, load_model, train_model as train_model_real, Dataset, Gpt2Config,
    Gpt2Model, LossFunction, LearningRateScheduler, SimilarityCalculator, TrainingConfig,
};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

type Backend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu>;

/// Parse learning rate scheduler from string and automatically choose initial learning rate
fn parse_learning_rate_config(scheduler_str: &str, initial_lr: Option<f64>) -> (LearningRateScheduler, f64) {
    let (scheduler, auto_lr) = match scheduler_str.to_lowercase().as_str() {
        "fixed" => (LearningRateScheduler::Fixed, 0.001),
        "linear-decay" => (LearningRateScheduler::LinearDecay { final_lr: 0.00001 }, 0.01),
        "exponential-decay" => (LearningRateScheduler::ExponentialDecay { decay_rate: 0.95 }, 0.005),
        "step-decay" => (LearningRateScheduler::StepDecay { step_size: 3, gamma: 0.5 }, 0.01),
        "cosine-annealing" => (LearningRateScheduler::CosineAnnealing { min_lr: 0.00001 }, 0.01),
        _ => {
            eprintln!("Unknown learning rate scheduler: {}. Using cosine-annealing.", scheduler_str);
            (LearningRateScheduler::CosineAnnealing { min_lr: 0.00001 }, 0.01)
        }
    };
    
    let final_lr = initial_lr.unwrap_or(auto_lr);
    (scheduler, final_lr)
}

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
    /// Train the model on TSV data
    Train {
        /// Path to training TSV file
        #[arg(short, long)]
        train_data: PathBuf,

        /// Path to validation TSV file (optional)
        #[arg(short, long)]
        validation_data: Option<PathBuf>,

        /// Path to save model checkpoints (default: checkpoints/)
        #[arg(short, long, default_value = "checkpoints")]
        output_dir: PathBuf,

        /// Number of training epochs
        #[arg(short, long, default_value = "10")]
        epochs: usize,

        /// Batch size for training
        #[arg(short, long, default_value = "2")]
        batch_size: usize,

        /// Learning rate scheduler: fixed, linear-decay, exponential-decay, step-decay, cosine-annealing
        #[arg(long, default_value = "cosine-annealing")]
        lr_scheduler: String,
        
        /// Initial learning rate (default: adaptive based on scheduler)
        #[arg(long)]
        initial_lr: Option<f64>,

        /// Loss function: contrastive, cosine, or mse
        #[arg(long, default_value = "contrastive")]
        loss: String,

        /// Checkpoint frequency (save every N epochs)
        #[arg(long, default_value = "1")]
        checkpoint_every: usize,

        /// Load pre-trained model to continue training
        #[arg(long)]
        resume_from: Option<PathBuf>,
        
        /// Number of attention heads (default: 12)
        #[arg(long, default_value = "12")]
        n_heads: usize,
        
        /// Number of transformer layers (default: 12)
        #[arg(long, default_value = "12")]
        n_layers: usize,
        
        /// Embedding dimension size (default: 768)
        #[arg(long, default_value = "768")]
        d_model: usize,
    },

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
        
        /// Number of attention heads (default: 4)
        #[arg(long, default_value = "4")]
        n_heads: usize,
        
        /// Number of transformer layers (default: 4)
        #[arg(long, default_value = "4")]
        n_layers: usize,
        
        /// Embedding dimension size (default: 768)
        #[arg(long, default_value = "768")]
        d_model: usize,
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
        
        /// Number of attention heads (default: 4)
        #[arg(long, default_value = "4")]
        n_heads: usize,
        
        /// Number of transformer layers (default: 4)
        #[arg(long, default_value = "4")]
        n_layers: usize,
        
        /// Embedding dimension size (default: 768)
        #[arg(long, default_value = "768")]
        d_model: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize WebGPU device
    let device = burn::backend::wgpu::WgpuDevice::default();

    match &cli.command {
        Commands::Train {
            train_data,
            validation_data,
            output_dir,
            epochs,
            batch_size,
            lr_scheduler,
            initial_lr,
            loss,
            checkpoint_every,
            resume_from,
            n_heads,
            n_layers,
            d_model,
        } => {
            train_model(
                train_data,
                validation_data.as_ref(),
                output_dir,
                *epochs,
                *batch_size,
                lr_scheduler,
                *initial_lr,
                loss,
                *checkpoint_every,
                resume_from.as_ref(),
                *n_heads,
                *n_layers,
                *d_model,
                device,
            )
            .await
        }

        Commands::Embed {
            model,
            sentence,
            format,
            n_heads,
            n_layers,
            d_model,
        } => embed_sentence(model.as_ref(), sentence, format, *n_heads, *n_layers, *d_model, device).await,

        Commands::Similarity {
            model,
            sentence1,
            sentence2,
            all_metrics,
            n_heads,
            n_layers,
            d_model,
        } => calculate_similarity(model.as_ref(), sentence1, sentence2, *all_metrics, *n_heads, *n_layers, *d_model, device).await,
    }
}

/// Get embedding for a sentence
async fn embed_sentence(
    model_path: Option<&PathBuf>,
    sentence: &str,
    format: &str,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    let config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: 1024,
        d_model,
        n_heads,
        n_layers,
        dropout: 0.1,
    };
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

/// Calculate similarity between two sentences
async fn calculate_similarity(
    model_path: Option<&PathBuf>,
    sentence1: &str,
    sentence2: &str,
    all_metrics: bool,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    let config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: 1024,
        d_model,
        n_heads,
        n_layers,
        dropout: 0.1,
    };
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
        println!(
            "Cosine Similarity: {:.4} (0=different, 1=identical)",
            similarity
        );
    }

    Ok(())
}

/// Train the model on TSV data
async fn train_model(
    train_data_path: &PathBuf,
    validation_data_path: Option<&PathBuf>,
    output_dir: &PathBuf,
    epochs: usize,
    batch_size: usize,
    lr_scheduler: &str,
    initial_lr: Option<f64>,
    loss_function: &str,
    _checkpoint_every: usize,
    resume_from: Option<&PathBuf>,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    println!("🚀 Starting GPT-2 Embedding Model Training");
    println!("==========================================");

    // Load training dataset
    println!("Loading training data from: {}", train_data_path.display());
    let train_dataset = Dataset::from_tsv(train_data_path)?;
    train_dataset.statistics().print();
    println!();

    // Load validation dataset if provided
    let validation_dataset = if let Some(val_path) = validation_data_path {
        println!("Loading validation data from: {}", val_path.display());
        let val_dataset = Dataset::from_tsv(val_path)?;
        val_dataset.statistics().print();
        println!();
        Some(val_dataset)
    } else {
        println!("No validation data provided");
        None
    };

    // Parse learning rate scheduler and automatically choose initial learning rate
    let (lr_scheduler, initial_learning_rate) = parse_learning_rate_config(lr_scheduler, initial_lr);
    
    // Parse loss function
    let loss_fn = match loss_function.to_lowercase().as_str() {
        "contrastive" => LossFunction::Contrastive,
        "cosine" => LossFunction::CosineEmbedding,
        "mse" => LossFunction::MseSimilarity,
        _ => {
            eprintln!(
                "Unknown loss function: {}. Using contrastive loss.",
                loss_function
            );
            LossFunction::Contrastive
        }
    };

    // Create training configuration
    let config = TrainingConfig {
        initial_learning_rate,
        lr_scheduler,
        epochs,
        batch_size,
        margin: 1.0, // Default margin for contrastive loss
        checkpoint_dir: output_dir.to_string_lossy().to_string(),
        early_stopping: true,
        loss_function: loss_fn,
    };

    // Create or load model
    let model_config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: 1024,
        d_model,
        n_heads,
        n_layers,
        dropout: 0.1,
    };
    let model = if let Some(model_path) = resume_from {
        println!("Resuming training from: {}", model_path.display());
        load_model::<Backend>(model_config, model_path, &device)?
    } else {
        println!("Initializing new model with random weights");
        Gpt2Model::new(model_config, &device)
    };

    // Create tokenizer
    let tokenizer = create_demo_tokenizer()?;

    // Start training with new approach
    train_model_real(
        model,
        tokenizer,
        config,
        train_dataset,
        validation_dataset,
        device,
    )
    .await?;

    println!("\n🎉 Training completed!");

    Ok(())
}

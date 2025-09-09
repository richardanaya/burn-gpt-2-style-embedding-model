use anyhow::Result;
use burn_gpt_n_embedding_model::{
    calculate_similarity, embed_sentence, train_model, validate_model, Dataset,
};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

// For our case, we need a backend where InnerBackend = B (i.e., not wrapped by Autodiff)
// This means we should use the inner backend directly for training
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
    /// Train the model on TSV data
    Train {
        /// Path to training TSV file
        #[arg(short, long)]
        train_data: PathBuf,

        /// Path to validation TSV file (optional)
        #[arg(short, long)]
        validation_data: Option<PathBuf>,

        /// Path to save model checkpoints (default: training_output/)
        #[arg(short, long, default_value = "training_output")]
        output_dir: PathBuf,

        /// Number of training epochs
        #[arg(short, long, default_value = "10")]
        epochs: usize,

        /// Batch size for training
        #[arg(short, long, default_value = "8")]
        batch_size: usize,

        /// Initial learning rate
        #[arg(long, default_value = "1e-5")]
        initial_lr: f64,

        /// Number of attention heads (default: 12)
        #[arg(long, default_value = "12")]
        n_heads: usize,

        /// Number of transformer layers (default: 12)
        #[arg(long, default_value = "12")]
        n_layers: usize,

        /// Embedding dimension size (default: 768)
        #[arg(long, default_value = "768")]
        d_model: usize,

        /// Maximum sequence length / context size (default: 1024)
        #[arg(long, default_value = "1024")]
        context_size: usize,

        /// Limit training examples for testing (0 = no limit)
        #[arg(long, default_value = "0")]
        limit_train: usize,

        /// Limit validation examples for testing (0 = no limit)
        #[arg(long, default_value = "0")]
        limit_validation: usize,

        /// Margin parameter for contrastive loss (default: 1.0)
        #[arg(long, default_value = "1.0")]
        margin: f32,
    },

    /// Get vector embedding for a sentence
    Embed {
        /// Path to the trained model file (optional - will use random weights if not provided)
        #[arg(short, long, default_value = "./training_output/model.mpk")]
        model: Option<PathBuf>,

        /// Sentence to get embedding for
        #[arg(short, long)]
        sentence: String,

        /// Output format: json or raw
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Number of attention heads (default: 12)
        #[arg(long, default_value = "12")]
        n_heads: usize,

        /// Number of transformer layers (default: 12)
        #[arg(long, default_value = "12")]
        n_layers: usize,

        /// Embedding dimension size (default: 768)
        #[arg(long, default_value = "768")]
        d_model: usize,

        /// Maximum sequence length / context size (default: 1024)
        #[arg(long, default_value = "1024")]
        context_size: usize,
    },

    /// Validate model accuracy on a dataset
    Validate {
        /// Path to the trained model file
        #[arg(short, long)]
        model: PathBuf,

        /// Path to validation TSV file
        #[arg(short, long)]
        validation_data: PathBuf,

        /// Batch size for validation (default: 4)
        #[arg(short, long, default_value = "8")]
        batch_size: usize,

        /// Number of attention heads (default: 12)
        #[arg(long, default_value = "12")]
        n_heads: usize,

        /// Number of transformer layers (default: 12)
        #[arg(long, default_value = "12")]
        n_layers: usize,

        /// Embedding dimension size (default: 768)
        #[arg(long, default_value = "768")]
        d_model: usize,

        /// Maximum sequence length / context size (default: 1024)
        #[arg(long, default_value = "1024")]
        context_size: usize,

        /// Limit validation samples (for testing)
        #[arg(long)]
        limit: Option<usize>,
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
        #[arg(long, default_value = "12")]
        n_heads: usize,

        /// Number of transformer layers (default: 4)
        #[arg(long, default_value = "12")]
        n_layers: usize,

        /// Embedding dimension size (default: 768)
        #[arg(long, default_value = "768")]
        d_model: usize,

        /// Maximum sequence length / context size (default: 1024)
        #[arg(long, default_value = "1024")]
        context_size: usize,
    },
}

fn load_datasets(
    train_data_path: &PathBuf,
    validation_data_path: Option<&PathBuf>,
    limit_train: usize,
    limit_validation: usize,
) -> Result<(Dataset, Option<Dataset>)> {
    println!("Loading training data from: {}", train_data_path.display());
    let mut train_dataset = Dataset::from_tsv(train_data_path)?;

    if limit_train > 0 {
        println!(
            "🔬 Limiting training data to {} examples for testing",
            limit_train
        );
        train_dataset.limit(limit_train);
    }

    train_dataset.statistics().print();
    println!();

    let validation_dataset = if let Some(val_path) = validation_data_path {
        println!("Loading validation data from: {}", val_path.display());
        let mut val_dataset = Dataset::from_tsv(val_path)?;

        if limit_validation > 0 {
            println!(
                "🔬 Limiting validation data to {} examples for testing (before: {})",
                limit_validation,
                val_dataset.examples.len()
            );
            val_dataset.limit(limit_validation);
            println!("🔬 After limiting: {} examples", val_dataset.examples.len());
        } else {
            println!(
                "🔬 No validation limit specified (limit_validation = {})",
                limit_validation
            );
        }

        val_dataset.statistics().print();
        println!();
        Some(val_dataset)
    } else {
        println!("No validation data provided");
        None
    };

    Ok((train_dataset, validation_dataset))
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
            initial_lr,
            n_heads,
            n_layers,
            d_model,
            context_size,
            limit_train,
            limit_validation,
            margin,
        } => {
            // Load datasets
            let (train_dataset, validation_dataset) = load_datasets(
                train_data,
                validation_data.as_ref(),
                *limit_train,
                *limit_validation,
            )?;

            // Display training configuration
            println!("\n🔥 Starting Training with Burn Framework");
            println!("==========================================");
            println!("📊 Training Examples: {}", train_dataset.len());
            if let Some(ref val_dataset) = validation_dataset {
                println!("📊 Validation Examples: {}", val_dataset.len());
            }
            println!("🔄 Epochs: {}", epochs);
            println!("📦 Batch Size: {}", batch_size);
            println!("🎯 Learning Rate: {:.2e}", initial_lr);
            println!("💾 Output dir: {}", output_dir.display());
            println!();

            train_model(
                train_dataset,
                validation_dataset,
                output_dir,
                *epochs,
                *batch_size,
                *initial_lr,
                *n_heads,
                *n_layers,
                *d_model,
                *context_size,
                *margin,
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
            context_size,
        } => {
            embed_sentence::<Backend>(
                model.as_ref(),
                sentence,
                format,
                *n_heads,
                *n_layers,
                *d_model,
                *context_size,
                device,
            )
            .await
        }

        Commands::Validate {
            model,
            validation_data,
            batch_size,
            n_heads,
            n_layers,
            d_model,
            context_size,
            limit,
        } => {
            validate_model::<Backend>(
                model,
                validation_data,
                *batch_size,
                *n_heads,
                *n_layers,
                *d_model,
                *context_size,
                *limit,
                device,
            )
            .await
        }

        Commands::Similarity {
            model,
            sentence1,
            sentence2,
            all_metrics,
            n_heads,
            n_layers,
            d_model,
            context_size,
        } => {
            calculate_similarity::<Backend>(
                model.as_ref(),
                sentence1,
                sentence2,
                *all_metrics,
                *n_heads,
                *n_layers,
                *d_model,
                *context_size,
                device,
            )
            .await
        }
    }
}

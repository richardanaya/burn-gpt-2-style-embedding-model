use anyhow::Result;
use burn_gpt_n_embedding_model::{
      train_model, Dataset,
};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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

        /// Path to validation TSV file
        #[arg(short, long)]
        validation_data: PathBuf,

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
        #[arg(long, default_value = "1e-3")]
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

        /// Disable TUI and use simple console output for training
        #[arg(long)]
        no_tui: bool,
    },
}

fn load_datasets(
    train_data_path: &PathBuf,
    validation_data_path: &PathBuf,
    limit_train: usize,
    limit_validation: usize,
) -> Result<(Dataset, Dataset)> {
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

    println!("Loading validation data from: {}", validation_data_path.display());
    let mut validation_dataset = Dataset::from_tsv(validation_data_path)?;

    if limit_validation > 0 {
        println!(
            "🔬 Limiting validation data to {} examples for testing (before: {})",
            limit_validation,
            validation_dataset.examples.len()
        );
        validation_dataset.limit(limit_validation);
        println!("🔬 After limiting: {} examples", validation_dataset.examples.len());
    } else {
        println!(
            "🔬 No validation limit specified (limit_validation = {})",
            limit_validation
        );
    }

    validation_dataset.statistics().print();
    println!();

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
            no_tui,
        } => {
            // Load datasets
            let (train_dataset, validation_dataset) = load_datasets(
                train_data,
                validation_data,
                *limit_train,
                *limit_validation,
            )?;

            // Display training configuration
            println!("\n🔥 Starting Training with Burn Framework");
            println!("==========================================");
            println!("📊 Training Examples: {}", train_dataset.len());
            println!("📊 Validation Examples: {}", validation_dataset.len());
            println!("🔄 Epochs: {}", epochs);
            println!("📦 Batch Size: {}", batch_size);
            println!("🎯 Learning Rate: {:.2e}", initial_lr);
            println!("💾 Output dir: {}", output_dir.display());
            println!();

            train_model(
                train_dataset,
                Some(validation_dataset),
                output_dir,
                *epochs,
                *batch_size,
                *initial_lr,
                *n_heads,
                *n_layers,
                *d_model,
                *context_size,
                *no_tui,
                device,
            )
            .await
        }
    }
}

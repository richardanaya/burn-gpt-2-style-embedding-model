use anyhow::Result;
use burn_gpt_n_embedding_model::{
    calculate_similarity, embed_sentence, train_model, validate_model, LearningRateScheduler,
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

        /// Learning rate scheduler: fixed, linear-decay, exponential-decay, step-decay, cosine-annealing
        #[arg(long, default_value = "cosine-annealing")]
        lr_scheduler: String,

        /// Initial learning rate (default: adaptive based on scheduler)
        #[arg(long)]
        initial_lr: Option<f64>,

        /// Loss function: contrastive, cosine, or mse
        #[arg(long, default_value = "contrastive")]
        loss: String,

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

/// Parse learning rate scheduler from string and automatically choose initial learning rate
pub fn parse_learning_rate_config(
    scheduler_str: &str,
    initial_lr: Option<f64>,
) -> (LearningRateScheduler, f64) {
    let (scheduler, auto_lr) = match scheduler_str.to_lowercase().as_str() {
        "fixed" => (LearningRateScheduler::Fixed, 1e-5),
        "linear-decay" => (LearningRateScheduler::LinearDecay { final_lr: 1e-6 }, 2e-5),
        "exponential-decay" => (
            LearningRateScheduler::ExponentialDecay { decay_rate: 0.95 },
            1.5e-5,
        ),
        "step-decay" => (
            LearningRateScheduler::StepDecay {
                step_size: 3,
                gamma: 0.5,
            },
            2e-5,
        ),
        "cosine-annealing" => (
            LearningRateScheduler::CosineAnnealing { min_lr: 1e-6 },
            2e-5,
        ),
        _ => {
            eprintln!(
                "Unknown learning rate scheduler: {}. Using cosine-annealing.",
                scheduler_str
            );
            (
                LearningRateScheduler::CosineAnnealing { min_lr: 1e-6 },
                2e-5,
            )
        }
    };

    let final_lr = initial_lr.unwrap_or(auto_lr);
    (scheduler, final_lr)
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
            n_heads,
            n_layers,
            d_model,
            context_size,
            limit_train,
            limit_validation,
            margin,
        } => {
            // Parse learning rate scheduler and automatically choose initial learning rate
            let (lr_scheduler_parsed, initial_learning_rate) =
                parse_learning_rate_config(lr_scheduler, *initial_lr);

            train_model(
                train_data,
                validation_data.as_ref(),
                output_dir,
                *epochs,
                *batch_size,
                &lr_scheduler_parsed,
                initial_learning_rate,
                loss,
                *n_heads,
                *n_layers,
                *d_model,
                *context_size,
                *limit_train,
                *limit_validation,
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

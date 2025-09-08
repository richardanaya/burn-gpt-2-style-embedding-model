use anyhow::Result;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn_gpt_n_embedding_model::{
    load_model, train_with_learner, BurnTrainingDataset, Dataset, Gpt2Config, Gpt2Model,
    Gpt2Tokenizer, LearningRateScheduler, LossFunction, SimilarityCalculator, TrainingConfig,
};
use indicatif::{ProgressBar, ProgressStyle};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

// For our case, we need a backend where InnerBackend = B (i.e., not wrapped by Autodiff)
// This means we should use the inner backend directly for training
type Backend = burn::backend::wgpu::Wgpu;
type AutodiffBackend = burn::backend::Autodiff<Backend>;

/// Parse learning rate scheduler from string and automatically choose initial learning rate
fn parse_learning_rate_config(
    scheduler_str: &str,
    initial_lr: Option<f64>,
) -> (LearningRateScheduler, f64) {
    let (scheduler, auto_lr) = match scheduler_str.to_lowercase().as_str() {
        "fixed" => (LearningRateScheduler::Fixed, 0.001),
        "linear-decay" => (
            LearningRateScheduler::LinearDecay { final_lr: 0.00001 },
            0.01,
        ),
        "exponential-decay" => (
            LearningRateScheduler::ExponentialDecay { decay_rate: 0.95 },
            0.005,
        ),
        "step-decay" => (
            LearningRateScheduler::StepDecay {
                step_size: 3,
                gamma: 0.5,
            },
            0.01,
        ),
        "cosine-annealing" => (
            LearningRateScheduler::CosineAnnealing { min_lr: 0.00001 },
            0.01,
        ),
        _ => {
            eprintln!(
                "Unknown learning rate scheduler: {}. Using cosine-annealing.",
                scheduler_str
            );
            (
                LearningRateScheduler::CosineAnnealing { min_lr: 0.00001 },
                0.01,
            )
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

        /// Path to save model checkpoints (default: training_output/)
        #[arg(short, long, default_value = "training_output")]
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

        /// Maximum sequence length / context size (default: 1024)
        #[arg(long, default_value = "1024")]
        context_size: usize,

        /// Limit training examples for testing (0 = no limit)
        #[arg(long, default_value = "0")]
        limit_train: usize,

        /// Limit validation examples for testing (0 = no limit)
        #[arg(long, default_value = "0")]
        limit_validation: usize,
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
        #[arg(short, long, default_value = "4")]
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
            context_size,
            limit_train,
            limit_validation,
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
                *context_size,
                *limit_train,
                *limit_validation,
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
            embed_sentence(
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
            validate_model(
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
            calculate_similarity(
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

/// Get embedding for a sentence
async fn embed_sentence(
    model_path: Option<&PathBuf>,
    sentence: &str,
    format: &str,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    context_size: usize,
    device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    let config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: context_size,
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

    let tokenizer = Gpt2Tokenizer::new_simple(context_size)?;

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
    context_size: usize,
    device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    let config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: context_size,
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

    let tokenizer = Gpt2Tokenizer::new_simple(context_size)?;

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
    _resume_from: Option<&PathBuf>,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    context_size: usize,
    limit_train: usize,
    limit_validation: usize,
    device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    println!("🚀 Starting GPT-2 Embedding Model Training");
    println!("==========================================");

    // Load training dataset
    println!("Loading training data from: {}", train_data_path.display());
    let mut train_dataset = Dataset::from_tsv(train_data_path)?;

    // Apply training limit if specified
    if limit_train > 0 {
        println!(
            "🔬 Limiting training data to {} examples for testing",
            limit_train
        );
        train_dataset.limit(limit_train);
    }

    train_dataset.statistics().print();
    println!();

    // Load validation dataset if provided
    let validation_dataset = if let Some(val_path) = validation_data_path {
        println!("Loading validation data from: {}", val_path.display());
        let mut val_dataset = Dataset::from_tsv(val_path)?;

        // Apply validation limit if specified
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

    // Parse learning rate scheduler and automatically choose initial learning rate
    let (_lr_scheduler, initial_learning_rate) =
        parse_learning_rate_config(lr_scheduler, initial_lr);

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

    // Create model configuration first
    let model_config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: context_size,
        d_model,
        n_heads,
        n_layers,
        dropout: 0.1,
    };

    // Create training configuration using the new Config pattern
    let config = TrainingConfig {
        model: model_config,
        optimizer: AdamConfig::new(),
        num_epochs: epochs,
        batch_size,
        num_workers: 1,
        seed: 42,
        learning_rate: initial_learning_rate,
        margin: 1.0,
        loss_function: loss_fn,
    };
    // Create tokenizer
    let tokenizer = Gpt2Tokenizer::new_simple(context_size)?;

    // Convert datasets to Burn format
    let burn_train_dataset = BurnTrainingDataset::from_dataset(&train_dataset);
    let burn_validation_dataset = validation_dataset
        .as_ref()
        .map(|ds| BurnTrainingDataset::from_dataset(ds));

    // Always provide a validation dataset - either the provided one or a limited version of training data
    let burn_validation_dataset = if let Some(val_dataset) = burn_validation_dataset {
        val_dataset
    } else {
        // No validation dataset provided, use a limited version of training data
        let mut limited_train = train_dataset.clone();
        if limit_validation > 0 {
            limited_train.limit(limit_validation);
            println!(
                "🔬 Using {} examples from training data for validation",
                limit_validation
            );
        }
        BurnTrainingDataset::from_dataset(&limited_train)
    };

    // Start training with new Learner-based approach
    train_with_learner::<AutodiffBackend>(
        &output_dir.to_string_lossy(),
        config,
        device,
        burn_train_dataset,
        burn_validation_dataset,
        tokenizer,
    );

    println!("\n🎉 Training completed!");

    Ok(())
}

/// Validate model accuracy on a dataset
async fn validate_model(
    model_path: &PathBuf,
    validation_data: &PathBuf,
    batch_size: usize,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    context_size: usize,
    limit: Option<usize>,
    device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    println!("🔍 Starting validation...");
    println!("Model: {}", model_path.display());
    println!("Validation data: {}", validation_data.display());

    // Load model configuration and weights
    let config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: context_size,
        d_model,
        n_heads,
        n_layers,
        dropout: 0.1,
    };

    println!("📦 Loading model...");
    let model = load_model::<Backend>(config, model_path, &device)?;
    let tokenizer = Gpt2Tokenizer::new_simple(context_size)?;

    // Load validation dataset
    println!("📊 Loading validation dataset...");
    let mut validation_dataset = Dataset::from_tsv(validation_data)?;
    
    if let Some(limit_val) = limit {
        println!("⚠️  Limiting validation to {} samples for testing", limit_val);
        validation_dataset.limit(limit_val);
    }

    let total_samples = validation_dataset.len();
    println!("📈 Dataset loaded: {} samples", total_samples);

    // Create similarity calculator
    let calculator = SimilarityCalculator::new(model, tokenizer);

    // Create progress bar
    let progress_bar = ProgressBar::new(total_samples as u64);
    progress_bar.set_style(
        ProgressStyle::with_template(
            "🔍 Validating [{bar:40.cyan/blue}] {pos}/{len} samples | Accuracy: {msg}"
        )
        .unwrap()
        .progress_chars("#>-")
    );

    // Process validation samples in batches
    let mut correct_predictions = 0;
    let mut total_processed = 0;

    println!("\n🚀 Starting validation...");
    
    for batch_start in (0..total_samples).step_by(batch_size) {
        let batch_end = std::cmp::min(batch_start + batch_size, total_samples);
        let batch_samples = &validation_dataset.examples[batch_start..batch_end];
        
        for sample in batch_samples {
            // Calculate cosine similarity between the two sentences
            let similarity = calculator.calculate_similarity(&sample.sentence1, &sample.sentence2)?;
            
            // Determine prediction based on similarity threshold
            // Typically, similarity > 0.5 means similar (label 1), otherwise different (label 0)
            let predicted_label = if similarity > 0.5 { 1 } else { 0 };
            
            if predicted_label == sample.label {
                correct_predictions += 1;
            }
            
            total_processed += 1;
            
            // Update progress bar with running accuracy
            let running_accuracy = if total_processed > 0 {
                (correct_predictions as f32 / total_processed as f32) * 100.0
            } else {
                0.0
            };
            
            progress_bar.set_position(total_processed as u64);
            progress_bar.set_message(format!("{:.1}%", running_accuracy));
        }
    }
    
    progress_bar.finish_with_message(format!("{:.1}%", 
        (correct_predictions as f32 / total_processed as f32) * 100.0));

    // Final results
    let final_accuracy = (correct_predictions as f32 / total_processed as f32) * 100.0;
    
    println!("\n\n✅ Validation Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Validation Results:");
    println!("   • Total samples processed: {}", total_processed);
    println!("   • Correct predictions: {}", correct_predictions);
    println!("   • Incorrect predictions: {}", total_processed - correct_predictions);
    println!("   • Accuracy: {:.2}% ({}/{} correct)", 
             final_accuracy, correct_predictions, total_processed);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Provide interpretation
    println!("\n🎯 Interpretation:");
    if final_accuracy >= 85.0 {
        println!("   • 🌟 Excellent performance! Model generalizes very well.");
    } else if final_accuracy >= 75.0 {
        println!("   • ✅ Good performance! Model is learning meaningful patterns.");
    } else if final_accuracy >= 65.0 {
        println!("   • ⚠️  Moderate performance. Consider more training or tuning.");
    } else if final_accuracy >= 55.0 {
        println!("   • 📈 Better than random guessing but needs improvement.");
    } else {
        println!("   • ❌ Poor performance. Model needs significant improvement.");
        println!("   • 💡 Try: More training epochs, different loss function, or data review.");
    }
    
    // Note about similarity threshold
    println!("\n💡 Note: This validation uses a 0.5 similarity threshold.");
    println!("   You may want to experiment with different thresholds for optimal results.");

    Ok(())
}

use crate::batcher::{TrainingBatch, TrainingBatcher};
use crate::summary::print_educational_metrics_explanation;
use crate::{
    data::{BurnTrainingDataset, Dataset},
    Gpt2Config, Gpt2Model, Gpt2Tokenizer,
};
use anyhow::Result;
use burn::data::dataloader::{DataLoaderBuilder, Dataset as BurnDataset};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{LearningRateMetric, LossMetric};
use burn::train::renderer::tui::TuiMetricsRenderer;
use burn::train::{LearnerBuilder, LearnerSummary, TrainOutput, TrainStep, ValidStep};
use burn::train::{RegressionOutput, TrainingInterrupter};
use std::path::PathBuf;

// Type aliases for main.rs compatibility
type WgpuBackend = burn::backend::wgpu::Wgpu;
type WgpuAutodiffBackend = burn::backend::Autodiff<WgpuBackend>;

/// Training configuration using official Burn Config pattern
#[derive(Config)]
pub struct TrainingConfig {
    /// Model configuration
    pub model: Gpt2Config,
    /// Adam optimizer configuration
    pub optimizer: AdamConfig,
    /// Number of training epochs
    #[config(default = 10)]
    pub num_epochs: usize,
    /// Batch size for training
    #[config(default = 8)]
    pub batch_size: usize,
    /// Number of worker threads for data loading
    #[config(default = 1)]
    pub num_workers: usize,
    /// Random seed for reproducibility
    #[config(default = 42)]
    pub seed: u64,
    /// Initial learning rate
    #[config(default = 1.0e-5)]
    pub learning_rate: f64,
    /// Loss function to use
    #[config(default = "LossFunction::Contrastive")]
    pub loss_function: LossFunction,
}

/// Available loss functions for training
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LossFunction {
    Contrastive,
    CosineEmbedding,
    MseSimilarity,
}

/// Learning rate scheduling strategies
#[derive(Debug, Clone)]
pub enum LearningRateScheduler {
    /// Fixed learning rate (no scheduling)
    Fixed,
    /// Linear decay from initial to final over all epochs
    LinearDecay { final_lr: f64 },
    /// Exponential decay: lr = initial_lr * decay_rate^epoch
    ExponentialDecay { decay_rate: f64 },
    /// Step decay: reduce by factor every N epochs
    StepDecay { step_size: usize, gamma: f64 },
    /// Cosine annealing with warm restarts
    CosineAnnealing { min_lr: f64 },
}

/// Implement TrainStep for our Gpt2Model
impl<B: AutodiffBackend> TrainStep<TrainingBatch<B>, RegressionOutput<B>> for Gpt2Model<B> {
    fn step(&self, batch: TrainingBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        // Forward pass - get sentence embeddings
        let emb1 = self.get_sentence_embedding(batch.sentence1.clone());
        let emb2 = self.get_sentence_embedding(batch.sentence2.clone());

        // Calculate squared Euclidean distance between embeddings
        let diff = emb1.clone() - emb2.clone();
        let sq_dist = diff.powf_scalar(2.0).mean_dim(1).squeeze_dims(&[1]); // [batch] - mean over hidden dim only

        // Labels are already f32, ensure they're on the same device
        let y = batch.labels.clone(); // [batch]

        // Calculate distance for margin
        let dist = sq_dist.clone().sqrt();

        // Contrastive loss: ½ * y * d² + ½ * (1−y) * max(0, margin − d)²
        let pos_loss = y.clone() * sq_dist.clone();
        let neg_loss = (Tensor::<B, 1>::ones_like(&y) - y.clone())
            * (self.margin - dist).clamp_min(0.0).powf_scalar(2.0);
        let loss_tensor: Tensor<B, 1> = 0.5 * (pos_loss + neg_loss).mean();

        // Calculate cosine similarity as prediction for metrics (map to 0-1 range)
        let dot_product = (emb1.clone() * emb2.clone()).sum_dim(1);
        let norm1 = emb1.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let norm2 = emb2.powf_scalar(2.0).sum_dim(1).sqrt();
        let cosine_sim = dot_product / (norm1 * norm2 + 1e-8);
        // Map from [-1,1] to [0,1] range to match label scale
        let predictions = (cosine_sim + 1.0) * 0.5;

        let output = RegressionOutput::new(
            batch.labels,
            predictions.detach(),
            loss_tensor.clone().unsqueeze(),
        );
        let grads = loss_tensor.backward();
        TrainOutput::new(self, grads, output)
    }
}

/// Implement ValidStep for validation
impl<B: Backend> ValidStep<TrainingBatch<B>, RegressionOutput<B>> for Gpt2Model<B> {
    fn step(&self, batch: TrainingBatch<B>) -> RegressionOutput<B> {
        // Forward pass without gradients for validation
        let embeddings1 = self.get_sentence_embedding(batch.sentence1).detach();
        let embeddings2 = self.get_sentence_embedding(batch.sentence2).detach();

        // Calculate cosine similarity as predictions (map to 0-1 scale)
        let dot_product = (embeddings1.clone() * embeddings2.clone()).sum_dim(1);
        let norm1 = embeddings1.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let norm2 = embeddings2.powf_scalar(2.0).sum_dim(1).sqrt();
        let cosine_sim = dot_product / (norm1 * norm2 + 1e-8);
        // Map from [-1,1] to [0,1] range to match label scale
        let predictions = (cosine_sim + 1.0) * 0.5;

        // For validation, we don't have a separate loss tensor, so we use predictions as both
        RegressionOutput::new(batch.labels, predictions.clone(), predictions.unsqueeze())
    }
}

/// Create artifact directory for training
fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts to get a clean start
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Official Burn training function using the Learner pattern with TUI metrics
pub fn train_with_learner<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    train_dataset: BurnTrainingDataset,
    validation_dataset: BurnTrainingDataset,
    tokenizer: Gpt2Tokenizer,
    _lr_scheduler: LearningRateScheduler,
) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher = TrainingBatcher::new(tokenizer.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset.clone());

    let validation_examples_count = validation_dataset.len();
    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(validation_dataset.clone());

    println!("\n🔥 Starting Training with Burn Framework");
    println!("==========================================");
    println!("📊 Training Examples: {}", train_dataset.len());
    println!("📊 Validation Examples: {}", validation_examples_count);
    println!("🔄 Epochs: {}", config.num_epochs);
    println!("📦 Batch Size: {}", config.batch_size);
    println!("🎯 Learning Rate: {}", config.learning_rate);
    println!("💾 Output dir: {}", artifact_dir);
    println!();

    // Create training interrupter for graceful shutdown
    let interrupter = TrainingInterrupter::new();

    // Setup Ctrl+C handler for graceful shutdown
    let interrupt_clone = interrupter.clone();
    ctrlc::set_handler(move || {
        println!("\n🛑 Graceful shutdown requested (Ctrl+C)...");
        interrupt_clone.stop();
    })
    .expect("Error setting Ctrl-C handler");

    // Setup TUI metrics renderer
    println!("📊 Initializing TUI metrics renderer...");
    let renderer = TuiMetricsRenderer::new(interrupter.clone(), None);

    // Note: Burn 0.18 may not have direct LR scheduler support in LearnerBuilder
    // The scheduler logic is implemented in our custom get_learning_rate method
    // and can be integrated with manual training loop if needed
    let learner = LearnerBuilder::new(artifact_dir)
        .with_file_checkpointer(CompactRecorder::new())
        .renderer(renderer)
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .metric_train_numeric(LearningRateMetric::new())
        .metric_valid_numeric(LearningRateMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    println!("\n✅ Training Complete!");

    // Load and display educational metric explanations
    if let Ok(summary) = LearnerSummary::new(&artifact_dir, &["Loss"]) {
        print_educational_metrics_explanation(&summary);
    } else {
        println!("📊 Unable to load training metrics summary for educational analysis.");
    }

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    println!("💾 Model saved to: {}/model", artifact_dir);
}

impl LearningRateScheduler {
    /// Calculate the current learning rate for the given epoch
    pub fn get_learning_rate(&self, epoch: usize, initial_lr: f64, total_epochs: usize) -> f64 {
        match self {
            LearningRateScheduler::Fixed => initial_lr,
            LearningRateScheduler::LinearDecay { final_lr } => {
                let progress = epoch as f64 / (total_epochs - 1) as f64;
                initial_lr + (final_lr - initial_lr) * progress
            }
            LearningRateScheduler::ExponentialDecay { decay_rate } => {
                initial_lr * decay_rate.powf(epoch as f64)
            }
            LearningRateScheduler::StepDecay { step_size, gamma } => {
                initial_lr * gamma.powf((epoch / step_size) as f64)
            }
            LearningRateScheduler::CosineAnnealing { min_lr } => {
                let progress = epoch as f64 / (total_epochs - 1) as f64;
                min_lr
                    + (initial_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0
            }
        }
    }
}

/// Train the model on TSV data
pub async fn train_model(
    train_data_path: &PathBuf,
    validation_data_path: Option<&PathBuf>,
    output_dir: &PathBuf,
    epochs: usize,
    batch_size: usize,
    lr_scheduler: &LearningRateScheduler,
    initial_lr: f64,
    loss_function: &str,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    context_size: usize,
    limit_train: usize,
    limit_validation: usize,
    margin: f32,
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

    // Learning rate scheduler and initial learning rate are now passed as parameters

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
        margin,
    };

    // Create training configuration using the new Config pattern
    let config = TrainingConfig {
        model: model_config,
        optimizer: AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig { penalty: 0.01 }))
            .with_grad_clipping(Some(GradientClippingConfig::Value(1.0))),
        num_epochs: epochs,
        batch_size,
        num_workers: 1,
        seed: 42,
        learning_rate: initial_lr,
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

    // Start training with TUI metrics renderer
    train_with_learner::<WgpuAutodiffBackend>(
        &output_dir.to_string_lossy(),
        config,
        device,
        burn_train_dataset,
        burn_validation_dataset,
        tokenizer,
        lr_scheduler.clone(),
    );

    println!("\n🎉 Training completed!");

    Ok(())
}

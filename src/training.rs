use crate::{data::Dataset, Gpt2Config, Gpt2Model, Gpt2Tokenizer};
use anyhow::Result;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoaderBuilder, Dataset as BurnDataset};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{Adaptor, Metric, MetricEntry, MetricMetadata};
use burn::train::{LearnerBuilder, LearnerSummary, TrainOutput, TrainStep, ValidStep};
use burn_train::metric::{ LearningRateMetric, LossMetric};
use burn_train::renderer::tui::TuiMetricsRenderer;
use burn_train::TrainingInterrupter;
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
    #[config(default = 2)]
    pub batch_size: usize,
    /// Number of worker threads for data loading
    #[config(default = 1)]
    pub num_workers: usize,
    /// Random seed for reproducibility
    #[config(default = 42)]
    pub seed: u64,
    /// Initial learning rate
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
    /// Margin for contrastive loss
    #[config(default = 1.0)]
    pub margin: f32,
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

/// Batch item for the Burn training system
#[derive(Clone, Debug)]
pub struct TrainingBatch<B: Backend> {
    pub sentence1: Tensor<B, 2, Int>,
    pub sentence2: Tensor<B, 2, Int>,
    pub labels: Tensor<B, 1>,
}

impl<B: Backend> TrainingBatch<B> {
    pub fn new(
        sentence1: Tensor<B, 2, Int>,
        sentence2: Tensor<B, 2, Int>,
        labels: Tensor<B, 1>,
    ) -> Self {
        Self {
            sentence1,
            sentence2,
            labels,
        }
    }
}

/// Training example from our dataset
#[derive(Clone, Debug)]
pub struct TrainingItem {
    pub sentence1: String,
    pub sentence2: String,
    pub label: f32,
}

impl TrainingItem {
    pub fn new(sentence1: String, sentence2: String, label: f32) -> Self {
        Self {
            sentence1,
            sentence2,
            label,
        }
    }
}

/// Wrapper for our dataset to implement Burn's Dataset trait
#[derive(Clone, Debug)]
pub struct BurnTrainingDataset {
    pub items: Vec<TrainingItem>,
}

impl BurnTrainingDataset {
    pub fn from_dataset(dataset: &Dataset) -> Self {
        let items = dataset
            .examples
            .iter()
            .map(|example| {
                TrainingItem::new(
                    example.sentence1.clone(),
                    example.sentence2.clone(),
                    example.label as f32,
                )
            })
            .collect();

        Self { items }
    }
}

impl BurnDataset<TrainingItem> for BurnTrainingDataset {
    fn get(&self, index: usize) -> Option<TrainingItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// Batcher to convert training items to batched tensors
#[derive(Clone)]
pub struct TrainingBatcher {
    tokenizer: Gpt2Tokenizer,
}

impl TrainingBatcher {
    pub fn new(tokenizer: Gpt2Tokenizer) -> Self {
        Self { tokenizer }
    }
}

impl<B: Backend> Batcher<B, TrainingItem, TrainingBatch<B>> for TrainingBatcher {
    fn batch(&self, items: Vec<TrainingItem>, device: &B::Device) -> TrainingBatch<B> {
        let mut sentence1_ids = Vec::new();
        let mut sentence2_ids = Vec::new();
        let mut labels = Vec::new();

        // Tokenize all sentences
        for item in items {
            if let (Ok(tokens1), Ok(tokens2)) = (
                self.tokenizer.encode(&item.sentence1, true),
                self.tokenizer.encode(&item.sentence2, true),
            ) {
                sentence1_ids.push(tokens1);
                sentence2_ids.push(tokens2);
                labels.push(item.label);
            }
        }

        if sentence1_ids.is_empty() {
            // Return empty batch if all tokenization failed
            let empty_tensor_1 = Tensor::<B, 2, Int>::zeros([0, 0], device);
            let empty_tensor_2 = Tensor::<B, 2, Int>::zeros([0, 0], device);
            let empty_labels = Tensor::<B, 1>::zeros([0], device);
            return TrainingBatch::new(empty_tensor_1, empty_tensor_2, empty_labels);
        }

        // Pad sequences
        let max_len1 = sentence1_ids.iter().map(|s| s.len()).max().unwrap_or(0);
        let max_len2 = sentence2_ids.iter().map(|s| s.len()).max().unwrap_or(0);

        let batch_size = sentence1_ids.len();
        let mut padded_sentence1 = Vec::with_capacity(batch_size * max_len1);
        let mut padded_sentence2 = Vec::with_capacity(batch_size * max_len2);

        for seq in sentence1_ids.iter() {
            let mut padded = seq.clone();
            padded.resize(max_len1, 0);
            padded_sentence1.extend(padded.iter().map(|&x| x as i64));
        }

        for seq in sentence2_ids.iter() {
            let mut padded = seq.clone();
            padded.resize(max_len2, 0);
            padded_sentence2.extend(padded.iter().map(|&x| x as i64));
        }

        let sentence1_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::from(&padded_sentence1[..]), device)
                .reshape([batch_size, max_len1]);

        let sentence2_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::from(&padded_sentence2[..]), device)
                .reshape([batch_size, max_len2]);

        let labels_tensor = Tensor::<B, 1>::from_data(TensorData::from(&labels[..]), device);

        TrainingBatch::new(sentence1_tensor, sentence2_tensor, labels_tensor)
    }
}

/// Training output containing loss - using RegressionOutput for compatibility
pub type SimilarityLoss<B> = burn::train::RegressionOutput<B>;

/// Input type for similarity accuracy metric
pub struct SimilarityAccuracyInput<B: Backend> {
    pub predictions: Tensor<B, 1>, // Model predictions (similarity scores)
    pub targets: Tensor<B, 1>,     // True labels (0 or 1)
}

/// Custom accuracy metric for similarity prediction following official Burn pattern
pub struct SimilarityAccuracyMetric<B: Backend> {
    state: NumericMetricState,
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for SimilarityAccuracyMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> SimilarityAccuracyMetric<B> {
    pub fn new() -> Self {
        Self {
            state: NumericMetricState::default(),
            _b: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Metric for SimilarityAccuracyMetric<B> {
    type Input = SimilarityAccuracyInput<B>;

    fn name(&self) -> String {
        "Similarity Accuracy".to_string()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size] = item.predictions.dims();

        // Convert predictions to binary (threshold at 0.5)
        let predicted_labels = item.predictions.clone().greater_elem(0.5);
        let true_labels = item.targets.clone().greater_elem(0.5);

        // Count correct predictions
        let correct = predicted_labels.equal(true_labels).int().sum();
        let accuracy_value = correct.clone().into_scalar().elem::<f32>() as f64 / batch_size as f64;

        self.state.update(
            accuracy_value * 100.0, // Convert to percentage
            batch_size,
            FormatOptions::new(self.name()).precision(2).unit("%"),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> burn::train::metric::Numeric for SimilarityAccuracyMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

/// Adaptor to convert from RegressionOutput to SimilarityAccuracyInput
impl<B: Backend> Adaptor<SimilarityAccuracyInput<B>> for SimilarityLoss<B> {
    fn adapt(&self) -> SimilarityAccuracyInput<B> {
        // Extract predictions and targets from the regression output
        // For similarity task: output contains predictions, targets contain true labels
        let predictions = self.output.clone().flatten::<1>(0, 1); // Flatten to 1D
        let targets = self.targets.clone().flatten::<1>(0, 1); // Flatten to 1D

        SimilarityAccuracyInput {
            predictions,
            targets,
        }
    }
}

/// Implement TrainStep for our Gpt2Model
impl<B: AutodiffBackend> TrainStep<TrainingBatch<B>, SimilarityLoss<B>> for Gpt2Model<B> {
    fn step(&self, batch: TrainingBatch<B>) -> TrainOutput<SimilarityLoss<B>> {
        // Forward pass - get embeddings
        let embeddings1 = self.get_sentence_embedding(batch.sentence1);
        let embeddings2 = self.get_sentence_embedding(batch.sentence2);

        // Calculate contrastive loss (simple MSE for now)
        let diff = embeddings1.clone() - embeddings2;
        let loss_tensor = diff.powf_scalar(2.0).mean_dim(1).mean();

        let output =
            SimilarityLoss::new(batch.labels, embeddings1, loss_tensor.clone().unsqueeze());
        let grads = loss_tensor.backward();
        TrainOutput::new(self, grads, output)
    }
}

/// Implement ValidStep for validation
impl<B: Backend> ValidStep<TrainingBatch<B>, SimilarityLoss<B>> for Gpt2Model<B> {
    fn step(&self, batch: TrainingBatch<B>) -> SimilarityLoss<B> {
        // Forward pass without gradients for validation
        let embeddings1 = self.get_sentence_embedding(batch.sentence1).detach();
        let embeddings2 = self.get_sentence_embedding(batch.sentence2).detach();

        // Calculate predictions (similarity scores)
        let diff = embeddings1.clone() - embeddings2;
        let predictions = diff.powf_scalar(2.0).mean_dim(1);

        SimilarityLoss::new(batch.labels, embeddings1, predictions.unsqueeze())
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
    }).expect("Error setting Ctrl-C handler");
    
    // Setup TUI metrics renderer
    println!("📊 Initializing TUI metrics renderer...");
    let renderer = TuiMetricsRenderer::new(interrupter.clone(), None);
    
    let learner = LearnerBuilder::new(artifact_dir)
        .with_file_checkpointer(CompactRecorder::new())
        .renderer(renderer)
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .metric_train_numeric(LearningRateMetric::new())
        .metric_valid_numeric(LearningRateMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(SimilarityAccuracyMetric::new())
        .metric_valid_numeric(SimilarityAccuracyMetric::new())
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

/// Pre-tokenized validation data (CPU-only to save GPU memory)
#[derive(Debug)]
#[allow(dead_code)]
struct TokenizedValidationBatch {
    padded1: Vec<i64>,
    padded2: Vec<i64>,
    max_len1: usize,
    max_len2: usize,
    labels: Vec<f32>,
}

/// Preprocess validation dataset once - keep on CPU to avoid GPU OOM
#[allow(dead_code)]
fn preprocess_validation_data_cpu(
    tokenizer: &Gpt2Tokenizer,
    val_dataset: &Dataset,
    batch_size: usize,
) -> Vec<TokenizedValidationBatch> {
    let val_batches = val_dataset.batches(batch_size);
    let mut preprocessed_batches = Vec::new();

    for batch_examples in val_batches.iter() {
        let mut sentence1_ids = Vec::new();
        let mut sentence2_ids = Vec::new();
        let mut labels = Vec::new();

        for example in batch_examples.iter() {
            if let (Ok(tokens1), Ok(tokens2)) = (
                tokenizer.encode(&example.sentence1, true),
                tokenizer.encode(&example.sentence2, true),
            ) {
                sentence1_ids.push(tokens1);
                sentence2_ids.push(tokens2);
                labels.push(example.label as f32);
            }
        }

        if sentence1_ids.is_empty() {
            continue;
        }

        // Pad sequences
        let max_len1 = sentence1_ids.iter().map(|s| s.len()).max().unwrap_or(0);
        let max_len2 = sentence2_ids.iter().map(|s| s.len()).max().unwrap_or(0);

        let mut padded1 = Vec::new();
        let mut padded2 = Vec::new();

        for tokens in sentence1_ids {
            let mut padded = tokens.clone();
            padded.resize(max_len1, 0);
            padded1.extend(padded.iter().map(|&x| x as i64));
        }

        for tokens in sentence2_ids {
            let mut padded = tokens.clone();
            padded.resize(max_len2, 0);
            padded2.extend(padded.iter().map(|&x| x as i64));
        }

        preprocessed_batches.push(TokenizedValidationBatch {
            padded1,
            padded2,
            max_len1,
            max_len2,
            labels,
        });
    }

    preprocessed_batches
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

/// Print educational explanations of training metrics based on actual values
fn print_educational_metrics_explanation(summary: &LearnerSummary) {
    println!("\n📊 Training Results Explanation:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Find training and validation loss metrics
    let train_loss = summary
        .metrics
        .train
        .iter()
        .find(|m| m.name == "Loss")
        .and_then(|m| m.entries.last())
        .map(|e| e.value);

    let valid_loss = summary
        .metrics
        .valid
        .iter()
        .find(|m| m.name == "Loss")
        .and_then(|m| m.entries.last())
        .map(|e| e.value);

    println!("📈 Loss Values:");
    if let Some(train_val) = train_loss {
        println!(
            "   • Training Loss: {:.3} - Measures how well the model fits training data",
            train_val
        );
    }
    if let Some(valid_val) = valid_loss {
        println!(
            "   • Validation Loss: {:.3} - Measures performance on unseen data",
            valid_val
        );
    }

    println!();
    println!("🔍 Key Observations:");

    // Provide conditional analysis based on actual values
    match (train_loss, valid_loss) {
        (Some(train_val), Some(valid_val)) => {
            if valid_val < train_val {
                let diff = train_val - valid_val;
                println!("   • ✅ Good Generalization: Validation loss ({:.3}) is {:.3} lower than training loss", 
                         valid_val, diff);
                println!("     This suggests the model isn't overfitting and may generalize well");
            } else if valid_val > train_val {
                let diff = valid_val - train_val;
                if diff > 0.1 {
                    println!("   • ⚠️  Possible Overfitting: Validation loss ({:.3}) is {:.3} higher than training loss", 
                             valid_val, diff);
                    println!(
                        "     Consider reducing model complexity or increasing regularization"
                    );
                } else {
                    println!("   • ✅ Normal Pattern: Validation loss ({:.3}) slightly higher than training loss ({:.3})", 
                             valid_val, train_val);
                    println!("     This is typical and suggests healthy learning");
                }
            } else {
                println!(
                    "   • ✅ Perfect Balance: Training and validation losses are nearly equal"
                );
                println!("     This indicates good generalization without overfitting");
            }

            // Loss magnitude analysis
            if train_val > 1.0 || valid_val > 1.0 {
                println!("   • 📈 High Loss Values: Consider running more epochs or adjusting learning rate");
            } else if train_val < 0.1 && valid_val < 0.1 {
                println!("   • 🎯 Very Low Loss: Model has learned the patterns well");
            } else {
                println!("   • 📊 Moderate Loss: Values in reasonable range for early training");
            }
        }
        (Some(train_val), None) => {
            println!(
                "   • Training Loss: {:.3} (validation data not available)",
                train_val
            );
        }
        (None, Some(valid_val)) => {
            println!(
                "   • Validation Loss: {:.3} (training data not available)",
                valid_val
            );
        }
        (None, None) => {
            println!("   • No loss metrics available for analysis");
        }
    }

    println!();
    println!("🚀 Next Steps:");
    if summary.epochs <= 2 {
        println!(
            "   • Consider running more epochs - {} epoch(s) is typically insufficient",
            summary.epochs
        );
        println!("     for a 117M parameter GPT-2 model to learn meaningful patterns");
    } else if summary.epochs < 10 {
        println!(
            "   • Good start with {} epochs - monitor loss trends over more epochs",
            summary.epochs
        );
    } else {
        println!(
            "   • With {} epochs completed, evaluate if further training improves results",
            summary.epochs
        );
    }

    if let (Some(train_val), Some(valid_val)) = (train_loss, valid_loss) {
        if (train_val - valid_val).abs() > 0.2 {
            println!("   • Large gap between train/validation suggests reviewing data split or regularization");
        }
    }

    println!("   • Monitor that validation loss follows training loss without diverging");
    println!("   • Use embeddings for downstream tasks to evaluate real-world performance");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

/// Parse learning rate scheduler from string and automatically choose initial learning rate
pub fn parse_learning_rate_config(
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

/// Train the model on TSV data
pub async fn train_model(
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

    // Start training with TUI metrics renderer
    train_with_learner::<WgpuAutodiffBackend>(
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

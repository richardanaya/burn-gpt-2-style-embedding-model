use crate::batcher::{TrainingBatch, TrainingBatcher};
use crate::summary::print_educational_metrics_explanation;
use crate::{
    data::{BurnTrainingDataset, Dataset},
    Gpt2Config, Gpt2Model, Gpt2Tokenizer,
};
use anyhow::Result;
use burn::data::dataloader::DataLoaderBuilder;
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

type WgpuBackend = burn::backend::wgpu::Wgpu;
type WgpuAutodiffBackend = burn::backend::Autodiff<WgpuBackend>;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: Gpt2Config,
    pub optimizer: AdamConfig,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub num_workers: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub loss_function: LossFunction,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LossFunction {
    Contrastive,
    CosineEmbedding,
    MseSimilarity,
}

#[derive(Debug, Clone)]
pub enum LearningRateScheduler {
    Fixed,
    LinearDecay { final_lr: f64 },
    ExponentialDecay { decay_rate: f64 },
    StepDecay { step_size: usize, gamma: f64 },
    CosineAnnealing { min_lr: f64 },
}

impl<B: AutodiffBackend> TrainStep<TrainingBatch<B>, RegressionOutput<B>> for Gpt2Model<B> {
    fn step(&self, batch: TrainingBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let emb1 = self.get_sentence_embedding(batch.sentence1.clone());
        let emb2 = self.get_sentence_embedding(batch.sentence2.clone());

        let diff = emb1.clone() - emb2.clone();
        let sq_dist = diff.powf_scalar(2.0).mean_dim(1).squeeze_dims(&[1]);

        let y = batch.labels.clone();

        let dist = sq_dist.clone().sqrt();

        let pos_loss = y.clone() * sq_dist.clone();
        let neg_loss = (Tensor::<B, 1>::ones_like(&y) - y.clone())
            * (self.margin - dist).clamp_min(0.0).powf_scalar(2.0);
        let loss_tensor: Tensor<B, 1> = 0.5 * (pos_loss + neg_loss).mean();

        let dot_product = (emb1.clone() * emb2.clone()).sum_dim(1);
        let norm1 = emb1.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let norm2 = emb2.powf_scalar(2.0).sum_dim(1).sqrt();
        let cosine_sim = dot_product / (norm1 * norm2 + 1e-8);
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

impl<B: Backend> ValidStep<TrainingBatch<B>, RegressionOutput<B>> for Gpt2Model<B> {
    fn step(&self, batch: TrainingBatch<B>) -> RegressionOutput<B> {
        let embeddings1 = self.get_sentence_embedding(batch.sentence1).detach();
        let embeddings2 = self.get_sentence_embedding(batch.sentence2).detach();

        let dot_product = (embeddings1.clone() * embeddings2.clone()).sum_dim(1);
        let norm1 = embeddings1.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let norm2 = embeddings2.powf_scalar(2.0).sum_dim(1).sqrt();
        let cosine_sim = dot_product / (norm1 * norm2 + 1e-8);
        let predictions = (cosine_sim + 1.0) * 0.5;

        RegressionOutput::new(batch.labels, predictions.clone(), predictions.unsqueeze())
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

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

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(validation_dataset.clone());


    println!("📊 Initializing TUI metrics renderer...");
    let renderer = TuiMetricsRenderer::new(TrainingInterrupter::new(), None);

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

pub async fn train_model(
    train_dataset: Dataset,
    validation_dataset: Option<Dataset>,
    output_dir: &PathBuf,
    epochs: usize,
    batch_size: usize,
    lr_scheduler: &LearningRateScheduler,
    initial_lr: f64,
    loss_function: LossFunction,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    context_size: usize,
    margin: f32,
    device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {

    let model_config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: context_size,
        d_model,
        n_heads,
        n_layers,
        dropout: 0.1,
        margin,
    };

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
        loss_function,
    };
    let tokenizer = Gpt2Tokenizer::new_simple(context_size)?;

    let burn_train_dataset = BurnTrainingDataset::from_dataset(&train_dataset);
    let burn_validation_dataset = if let Some(val_dataset) = validation_dataset {
        BurnTrainingDataset::from_dataset(&val_dataset)
    } else {
        BurnTrainingDataset::from_dataset(&train_dataset)
    };

    train_with_learner::<WgpuAutodiffBackend>(
        &output_dir.to_string_lossy(),
        config,
        device,
        burn_train_dataset,
        burn_validation_dataset,
        tokenizer,
        lr_scheduler.clone(),
    );

    Ok(())
}

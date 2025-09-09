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

        let y = batch.labels.clone();

        // Calculate contrastive loss (same formula as training, but without backprop)
        let diff = embeddings1.clone() - embeddings2.clone();
        let sq_dist = diff.powf_scalar(2.0).mean_dim(1).squeeze_dims(&[1]);
        let dist = sq_dist.clone().sqrt();

        let pos_loss = y.clone() * sq_dist.clone();
        let neg_loss = (Tensor::<B, 1>::ones_like(&y) - y.clone())
            * (self.margin - dist).clamp_min(0.0).powf_scalar(2.0);
        let valid_loss = 0.5 * (pos_loss + neg_loss).mean().unsqueeze();

        // Calculate predictions using cosine similarity for metrics
        let dot_product = (embeddings1.clone() * embeddings2.clone()).sum_dim(1);
        let norm1 = embeddings1.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let norm2 = embeddings2.powf_scalar(2.0).sum_dim(1).sqrt();
        let cosine_sim = dot_product / (norm1 * norm2 + 1e-8);
        let predictions = (cosine_sim + 1.0) * 0.5;

        RegressionOutput::new(y, predictions, valid_loss)
    }
}

pub async fn train_model(
    train_dataset: Dataset,
    validation_dataset: Option<Dataset>,
    output_dir: &PathBuf,
    epochs: usize,
    batch_size: usize,
    initial_lr: f64,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    context_size: usize,
    no_tui: bool,
    device: burn::backend::wgpu::WgpuDevice,
) -> Result<()> {
    let model_config = Gpt2Config {
        vocab_size: 50257,
        max_seq_len: context_size,
        d_model,
        n_heads,
        n_layers,
        dropout: 0.1,
        margin: 1.0, // Good default for contrastive loss
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
    };
    let tokenizer = Gpt2Tokenizer::new_simple(context_size)?;

    let burn_train_dataset = BurnTrainingDataset::from_dataset(&train_dataset);
    let burn_validation_dataset = if let Some(val_dataset) = validation_dataset {
        BurnTrainingDataset::from_dataset(&val_dataset)
    } else {
        BurnTrainingDataset::from_dataset(&train_dataset)
    };

    std::fs::remove_dir_all(output_dir).ok();
    std::fs::create_dir_all(output_dir).ok();
    config
        .save(format!("{}/config.json", output_dir.display()))
        .expect("Config should be saved successfully");

    WgpuAutodiffBackend::seed(config.seed);

    let batcher = TrainingBatcher::new(tokenizer.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(burn_train_dataset.clone());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(burn_validation_dataset.clone());

    let mut learner_builder =
        LearnerBuilder::new(output_dir).with_file_checkpointer(CompactRecorder::new());

    if !no_tui {
        println!("📊 Initializing TUI metrics renderer...");
        let renderer = TuiMetricsRenderer::new(TrainingInterrupter::new(), None);
        learner_builder = learner_builder.renderer(renderer);
    } else {
        println!("📊 Running in headless mode (no TUI)...");
    }

    let learner = learner_builder
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .metric_train_numeric(LearningRateMetric::new())
        .metric_valid_numeric(LearningRateMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .summary()
        .build(
            config.model.init::<WgpuAutodiffBackend>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    println!("\n✅ Training Complete!");

    if let Ok(summary) = LearnerSummary::new(output_dir, &["Loss"]) {
        print_educational_metrics_explanation(&summary);
    } else {
        println!("📊 Unable to load training metrics summary for educational analysis.");
    }

    model_trained
        .save_file(
            format!("{}/model", output_dir.display()),
            &CompactRecorder::new(),
        )
        .expect("Trained model should be saved successfully");

    println!("💾 Model saved to: {}/model", output_dir.display());

    Ok(())
}

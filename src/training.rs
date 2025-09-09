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

/* -------------------------------------------------------------------------- */
/*                         Shared contrastive loss logic                      */
/* -------------------------------------------------------------------------- */

/// Contrastive-loss + cosine-similarity metrics used by *both* training and
/// validation steps.
///
/// Returns the unreduced scalar loss (needed for `backward()` during training)
/// together with the ready-to-use `RegressionOutput`.
fn build_regression_output<B: Backend>(
    emb1: Tensor<B, 2>,
    emb2: Tensor<B, 2>,
    labels: Tensor<B, 1>,
    margin: f32,
) -> (Tensor<B, 1>, RegressionOutput<B>) {
    // -------- contrastive loss (pairwise distance) -------------------------
    let diff = emb1.clone() - emb2.clone();
    let sq_dist = diff.powf_scalar(2.0).sum_dim(1).squeeze_dims(&[1]); // [batch]
    let dist = sq_dist.clone().sqrt();

    let pos_loss = labels.clone() * sq_dist.clone();
    let neg_loss = (Tensor::<B, 1>::ones_like(&labels) - labels.clone())
        * (margin - dist).clamp_min(0.0).powf_scalar(2.0);

    // Mean over batch → scalar (shape [1])
    let loss_scalar: Tensor<B, 1> = 0.5 * (pos_loss + neg_loss).mean();

    // -------- cosine similarity predictions --------------------------------
    let dot_product = (emb1.clone() * emb2.clone()).sum_dim(1);
    let norm1 = emb1.powf_scalar(2.0).sum_dim(1).sqrt();
    let norm2 = emb2.powf_scalar(2.0).sum_dim(1).sqrt();
    let cosine_sim = dot_product / (norm1 * norm2 + 1e-8);
    let predictions = (cosine_sim + 1.0) * 0.5; // range [0,1]

    // `RegressionOutput` expects 2-D tensors for predictions / targets
    let output = RegressionOutput::new(
        loss_scalar.clone().unsqueeze(), // [1]
        predictions.unsqueeze(),         // [batch,1]
        labels.unsqueeze(),              // [batch,1]
    );

    (loss_scalar, output)
}

/* -------------------------------------------------------------------------- */
/*                       Train & Valid step implementations                   */
/* -------------------------------------------------------------------------- */

impl<B: AutodiffBackend> TrainStep<TrainingBatch<B>, RegressionOutput<B>> for Gpt2Model<B> {
    fn step(&self, batch: TrainingBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        // ----- sentence embeddings (track grads) ---------------------------
        let emb1 =
            self.get_sentence_embedding_masked(batch.sentence1.clone(), &batch.sentence1_lengths);
        let emb2 =
            self.get_sentence_embedding_masked(batch.sentence2.clone(), &batch.sentence2_lengths);

        let labels = batch.labels.clone();

        // Shared logic
        let (loss_tensor, output) = build_regression_output(emb1, emb2, labels, self.margin);

        // Back-prop
        let gradients = loss_tensor.backward();
        TrainOutput::new(self, gradients, output)
    }
}

impl<B: Backend> ValidStep<TrainingBatch<B>, RegressionOutput<B>> for Gpt2Model<B> {
    fn step(&self, batch: TrainingBatch<B>) -> RegressionOutput<B> {
        // Detach embeddings to avoid autograd graph creation
        let emb1 = self
            .get_sentence_embedding_masked(batch.sentence1, &batch.sentence1_lengths)
            .detach();
        let emb2 = self
            .get_sentence_embedding_masked(batch.sentence2, &batch.sentence2_lengths)
            .detach();

        let labels = batch.labels.clone();

        // We only need the `RegressionOutput`
        let (_, output) = build_regression_output(emb1, emb2, labels, self.margin);
        output
    }
}

/* -------------------------------------------------------------------------- */
/*                              Training driver                               */
/* -------------------------------------------------------------------------- */

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
        margin: 1.0, // default for contrastive loss
    };

    let config = TrainingConfig {
        model: model_config,
        optimizer: AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig { penalty: 0.01 }))
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
        num_epochs: epochs,
        batch_size,
        num_workers: 1,
        seed: 42,
        learning_rate: initial_lr,
    };

    let tokenizer = Gpt2Tokenizer::new_simple(context_size)?;

    // Burn datasets ---------------------------------------------------------
    let burn_train_dataset = BurnTrainingDataset::from_dataset(&train_dataset);
    let burn_validation_dataset = validation_dataset
        .as_ref()
        .map(|d| BurnTrainingDataset::from_dataset(d))
        .unwrap_or_else(|| BurnTrainingDataset::from_dataset(&train_dataset));

    // Output dir ------------------------------------------------------------
    std::fs::remove_dir_all(output_dir).ok();
    std::fs::create_dir_all(output_dir).ok();
    config
        .save(format!("{}/config.json", output_dir.display()))
        .expect("Config should be saved successfully");

    // Seeding
    WgpuAutodiffBackend::seed(config.seed);

    // Dataloaders -----------------------------------------------------------
    let batcher = TrainingBatcher::new(tokenizer.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(burn_train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(burn_validation_dataset);

    // Learner ---------------------------------------------------------------
    let mut learner_builder =
        LearnerBuilder::new(output_dir).with_file_checkpointer(CompactRecorder::new());

    if !no_tui {
        println!("📊 Initializing TUI metrics renderer…");
        let renderer = TuiMetricsRenderer::new(TrainingInterrupter::new(), None);
        learner_builder = learner_builder.renderer(renderer);
    } else {
        println!("📊 Running in headless mode (no TUI) …");
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

    // Train --------------------------------------------------------------
    let model_trained = learner.fit(dataloader_train, dataloader_valid);
    println!("\n✅ Training Complete!");

    // Educational summary ------------------------------------------------
    if let Ok(summary) = LearnerSummary::new(output_dir, &["Loss"]) {
        print_educational_metrics_explanation(&summary);
    } else {
        println!("📊 Unable to load training metrics summary for educational analysis.");
    }

    // Save ---------------------------------------------------------------
    model_trained
        .save_file(
            format!("{}/model", output_dir.display()),
            &CompactRecorder::new(),
        )
        .expect("Trained model should be saved successfully");
    println!("💾 Model saved to: {}/model", output_dir.display());

    Ok(())
}

use crate::{data::Dataset, Gpt2Config, Gpt2Model, Gpt2Tokenizer};
use anyhow::{anyhow, Result};
use burn::prelude::*;
use burn::record::{BinGzFileRecorder, FullPrecisionSettings};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tokio::signal;

/// Save model weights in binary format
pub fn save_model<B: Backend>(
    model: &Gpt2Model<B>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let recorder = BinGzFileRecorder::<FullPrecisionSettings>::default();
    model
        .clone()
        .save_file(path.as_ref().to_path_buf(), &recorder)
        .map_err(|e| anyhow!("Failed to save model: {}", e))?;
    Ok(())
}

/// Load model weights from binary format
pub fn load_model<B: Backend>(
    config: Gpt2Config,
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<Gpt2Model<B>> {
    let mut model = Gpt2Model::new(config, device);
    let recorder = BinGzFileRecorder::<FullPrecisionSettings>::default();
    model = model
        .load_file(path.as_ref().to_path_buf(), &recorder, device)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;
    Ok(model)
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate for the optimizer
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Margin for contrastive loss
    pub margin: f32,
    /// How often to save checkpoints (in epochs)
    pub checkpoint_every: usize,
    /// Path to save model checkpoints
    pub checkpoint_dir: String,
    /// Whether to shuffle data between epochs
    pub shuffle: bool,
    /// Loss function to use
    pub loss_function: LossFunction,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            epochs: 10,
            batch_size: 16,
            margin: 1.0,
            checkpoint_every: 1,
            checkpoint_dir: "checkpoints".to_string(),
            shuffle: true,
            loss_function: LossFunction::Contrastive,
        }
    }
}

/// Available loss functions for training
#[derive(Debug, Clone)]
pub enum LossFunction {
    Contrastive,
    CosineEmbedding,
    MseSimilarity,
}

/// Training statistics for an epoch
#[derive(Debug, Clone)]
pub struct EpochStats {
    pub epoch: usize,
    pub train_loss: f32,
    pub train_accuracy: f32,
    pub validation_loss: Option<f32>,
    pub validation_accuracy: Option<f32>,
    pub duration_secs: f32,
}

impl EpochStats {
    /// Print formatted epoch statistics
    pub fn print(&self) {
        print!(
            "Epoch {}: train_loss={:.4}, train_acc={:.1}%",
            self.epoch, self.train_loss, self.train_accuracy * 100.0
        );
        
        if let (Some(val_loss), Some(val_acc)) = (self.validation_loss, self.validation_accuracy) {
            print!(", val_loss={:.4}, val_acc={:.1}%", val_loss, val_acc * 100.0);
        }
        
        println!(", time={:.1}s", self.duration_secs);
    }
}

/// Simplified trainer for the GPT-2 embedding model
/// This is a demonstration implementation that shows the training workflow
/// In a full implementation, this would include proper gradient computation and backpropagation
pub struct Trainer<B: Backend> {
    model: Gpt2Model<B>,
    tokenizer: Gpt2Tokenizer,
    config: TrainingConfig,
    device: B::Device,
    interrupt_flag: Arc<AtomicBool>,
}

impl<B: Backend<FloatElem = f32>> Trainer<B> {
    /// Create a new trainer
    pub fn new(
        model: Gpt2Model<B>,
        tokenizer: Gpt2Tokenizer,
        config: TrainingConfig,
        device: B::Device,
    ) -> Self {
        Self {
            model,
            tokenizer,
            config,
            device,
            interrupt_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Train the model on the provided dataset
    /// This is a simplified implementation that demonstrates the training workflow
    pub async fn train(
        &mut self,
        train_dataset: Dataset,
        validation_dataset: Option<Dataset>,
    ) -> Result<Vec<EpochStats>> {
        println!("Starting training with configuration:");
        println!("  Learning rate: {}", self.config.learning_rate);
        println!("  Epochs: {}", self.config.epochs);
        println!("  Batch size: {}", self.config.batch_size);
        println!("  Loss function: {:?}", self.config.loss_function);
        println!("  Checkpoint every: {} epochs", self.config.checkpoint_every);
        println!();

        // Set up interrupt handler
        self.setup_interrupt_handler().await?;

        // Create checkpoint directory
        std::fs::create_dir_all(&self.config.checkpoint_dir)?;

        let mut all_stats = Vec::new();
        let mut train_data = train_dataset;

        // Training loop
        for epoch in 1..=self.config.epochs {
            // Check for interruption
            if self.interrupt_flag.load(Ordering::Relaxed) {
                println!("\nTraining interrupted by user. Saving checkpoint...");
                self.save_checkpoint(epoch - 1).await?;
                break;
            }

            let epoch_start = Instant::now();
            
            // Shuffle data if configured
            if self.config.shuffle {
                train_data.shuffle(&mut rand::thread_rng());
            }

            // Train for one epoch
            let (train_loss, train_accuracy) = self.train_epoch(&train_data).await?;

            // Validate if validation dataset provided
            let (validation_loss, validation_accuracy) = if let Some(ref val_data) = validation_dataset {
                let (val_loss, val_acc) = self.validate_epoch(val_data).await?;
                (Some(val_loss), Some(val_acc))
            } else {
                (None, None)
            };

            let duration = epoch_start.elapsed().as_secs_f32();

            let stats = EpochStats {
                epoch,
                train_loss,
                train_accuracy,
                validation_loss,
                validation_accuracy,
                duration_secs: duration,
            };

            stats.print();
            all_stats.push(stats);

            // Save checkpoint if needed
            if epoch % self.config.checkpoint_every == 0 {
                self.save_checkpoint(epoch).await?;
            }
        }

        // Save final model
        println!("\nTraining completed. Saving final model...");
        self.save_final_model().await?;

        Ok(all_stats)
    }

    /// Train for a single epoch (simplified implementation)
    async fn train_epoch(&mut self, dataset: &Dataset) -> Result<(f32, f32)> {
        let batches = dataset.batches(self.config.batch_size);
        let total_batches = batches.len();

        let progress = ProgressBar::new(total_batches as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let mut batch_count = 0;

        for (batch_idx, _batch_examples) in batches.iter().enumerate() {
            // Check for interruption
            if self.interrupt_flag.load(Ordering::Relaxed) {
                break;
            }

            // Simulate processing the batch
            // In a real implementation, this would:
            // 1. Tokenize the batch
            // 2. Run forward pass through the model
            // 3. Calculate loss
            // 4. Compute gradients
            // 5. Update model weights
            
            // For demonstration, we'll just simulate some work and generate mock metrics
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            
            // Mock loss and accuracy (in real implementation, these would be computed)
            let mock_loss = 0.5 + (batch_idx as f32 * 0.01) * rand::random::<f32>();
            let mock_accuracy = 0.7 + 0.3 * rand::random::<f32>();

            total_loss += mock_loss;
            total_accuracy += mock_accuracy;
            batch_count += 1;

            progress.set_message(format!("Processing batch {}/{}", batch_idx + 1, total_batches));
            progress.inc(1);
        }

        progress.finish_with_message("Epoch completed");

        let avg_loss = if batch_count > 0 { total_loss / batch_count as f32 } else { 0.0 };
        let avg_accuracy = if batch_count > 0 { total_accuracy / batch_count as f32 } else { 0.0 };

        Ok((avg_loss, avg_accuracy))
    }

    /// Validate for a single epoch (simplified implementation)
    async fn validate_epoch(&self, dataset: &Dataset) -> Result<(f32, f32)> {
        let batches = dataset.batches(self.config.batch_size);
        
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let mut batch_count = 0;

        for (batch_idx, _batch_examples) in batches.iter().enumerate() {
            // Simulate validation processing
            tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
            
            // Mock validation metrics
            let mock_loss = 0.6 + (batch_idx as f32 * 0.005) * rand::random::<f32>();
            let mock_accuracy = 0.65 + 0.35 * rand::random::<f32>();

            total_loss += mock_loss;
            total_accuracy += mock_accuracy;
            batch_count += 1;
        }

        let avg_loss = if batch_count > 0 { total_loss / batch_count as f32 } else { 0.0 };
        let avg_accuracy = if batch_count > 0 { total_accuracy / batch_count as f32 } else { 0.0 };

        Ok((avg_loss, avg_accuracy))
    }

    /// Save a checkpoint
    async fn save_checkpoint(&self, epoch: usize) -> Result<()> {
        let checkpoint_path = format!("{}/checkpoint_epoch_{}.bin", self.config.checkpoint_dir, epoch);
        save_model(&self.model, &checkpoint_path)?;
        println!("Checkpoint saved: {}", checkpoint_path);
        Ok(())
    }

    /// Save the final trained model
    async fn save_final_model(&self) -> Result<()> {
        let final_path = format!("{}/final_model.bin", self.config.checkpoint_dir);
        save_model(&self.model, &final_path)?;
        println!("Final model saved: {}", final_path);
        Ok(())
    }

    /// Set up interrupt handler for graceful shutdown
    async fn setup_interrupt_handler(&self) -> Result<()> {
        let interrupt_flag = Arc::clone(&self.interrupt_flag);
        
        tokio::spawn(async move {
            match signal::ctrl_c().await {
                Ok(()) => {
                    println!("\nReceived interrupt signal (Ctrl+C)");
                    interrupt_flag.store(true, Ordering::Relaxed);
                }
                Err(err) => {
                    eprintln!("Failed to listen for interrupt signal: {}", err);
                }
            }
        });

        Ok(())
    }
}
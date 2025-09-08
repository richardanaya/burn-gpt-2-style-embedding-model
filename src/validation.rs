use crate::{load_model, Dataset, Gpt2Config, SimilarityCalculator};
use anyhow::Result;
use burn::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

/// Validate model accuracy on a dataset
///
/// This function evaluates how well the trained embedding model performs on a validation dataset.
/// It loads the model, processes validation examples in batches, and computes accuracy based on
/// similarity threshold classification.
///
/// ## Validation Process
/// 1. **Model Loading**: Loads the trained model from the specified path
/// 2. **Dataset Processing**: Loads validation TSV data with sentence pairs and labels
/// 3. **Batch Processing**: Processes samples efficiently in configurable batch sizes
/// 4. **Similarity Calculation**: Uses cosine similarity between sentence embeddings
/// 5. **Classification**: Applies 0.5 threshold (>0.5 = similar, ≤0.5 = different)
/// 6. **Metrics**: Computes accuracy and provides interpretation guidance
///
/// ## Parameters
/// - `model_path`: Path to the saved model checkpoint (.mpk file)
/// - `validation_data`: Path to validation TSV file with format: sentence1\tsentence2\tlabel
/// - `batch_size`: Number of samples to process at once (affects memory usage)
/// - Model architecture parameters (n_heads, n_layers, d_model, context_size)
/// - `limit`: Optional limit on validation samples for testing
/// - `device`: WebGPU device for model operations
pub async fn validate_model<B: Backend<FloatElem = f32>>(
    model_path: &PathBuf,
    validation_data: &PathBuf,
    batch_size: usize,
    n_heads: usize,
    n_layers: usize,
    d_model: usize,
    context_size: usize,
    limit: Option<usize>,
    device: B::Device,
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
    let model = load_model::<B>(config, model_path, &device)?;
    let tokenizer = crate::Gpt2Tokenizer::new_simple(context_size)?;

    // Load validation dataset
    println!("📊 Loading validation dataset...");
    let mut validation_dataset = Dataset::from_tsv(validation_data)?;

    if let Some(limit_val) = limit {
        println!(
            "⚠️  Limiting validation to {} samples for testing",
            limit_val
        );
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
            "🔍 Validating [{bar:40.cyan/blue}] {pos}/{len} samples | Accuracy: {msg}",
        )
        .unwrap()
        .progress_chars("#>-"),
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
            let similarity =
                calculator.calculate_similarity(&sample.sentence1, &sample.sentence2)?;

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

    progress_bar.finish_with_message(format!(
        "{:.1}%",
        (correct_predictions as f32 / total_processed as f32) * 100.0
    ));

    // Final results
    let final_accuracy = (correct_predictions as f32 / total_processed as f32) * 100.0;

    println!("\n\n✅ Validation Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Validation Results:");
    println!("   • Total samples processed: {}", total_processed);
    println!("   • Correct predictions: {}", correct_predictions);
    println!(
        "   • Incorrect predictions: {}",
        total_processed - correct_predictions
    );
    println!(
        "   • Accuracy: {:.2}% ({}/{} correct)",
        final_accuracy, correct_predictions, total_processed
    );
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

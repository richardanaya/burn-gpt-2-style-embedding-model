use anyhow::Result;
use burn_gpt_n_embedding_model::{train_model, Dataset};
use std::path::PathBuf;

/// Minimal runnable example that demonstrates training loop health
///
/// This example creates a tiny dataset with clearly similar/dissimilar pairs
/// and runs a short training session to verify:
/// 1. Training loop executes without errors
/// 2. Both training and validation loss curves trend downward
/// 3. Model learns to distinguish between similar and dissimilar pairs
#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 Running minimal training example...");

    // Create a small dataset with clear similar/dissimilar pairs
    let dummy_pairs = vec![
        ("What is Rust?", "Explain Rust language.", 1.0),
        ("Cat", "Dog", 0.0),
        ("Fast car", "Quick automobile", 1.0),
        ("Coffee", "Tea", 0.0),
        ("Hello world", "Hi there", 1.0),
        ("Winter cold", "Summer heat", 0.0),
        ("Programming language", "Coding language", 1.0),
        ("Blue sky", "Green grass", 0.0),
    ];

    let train_ds = Dataset::from_pairs(dummy_pairs.clone());
    let val_ds = Dataset::from_pairs(dummy_pairs);

    println!("📊 Dataset created:");
    train_ds.statistics().print();
    println!();

    let device = burn::backend::wgpu::WgpuDevice::default();

    train_model(
        train_ds,
        Some(val_ds),
        &PathBuf::from("runs/minimal_example"),
        5,    // epochs
        4,    // batch size
        1e-3, // learning rate
        2,    // n_heads (reduced for faster training)
        2,    // n_layers (reduced for faster training)
        128,  // d_model (reduced for faster training)
        32,   // context size (reduced for faster training)
        true, // no_tui - use headless mode for testing
        device,
    )
    .await?;

    println!("\n✅ Minimal training example completed successfully!");
    println!("📈 If both training and validation loss curves trended down, your training loop is healthy.");

    Ok(())
}

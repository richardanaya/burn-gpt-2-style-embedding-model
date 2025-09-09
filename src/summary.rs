use burn::train::LearnerSummary;

/// Print educational explanations of training metrics based on actual values
pub fn print_educational_metrics_explanation(summary: &LearnerSummary) {
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
}
